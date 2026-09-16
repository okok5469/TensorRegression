"""Manifest-driven LEMON download planning; transfers require explicit opt-in."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import errno
import hashlib
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError('Manifest path must be a nonempty relative POSIX path')
    parts = value.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                *(f'LPT{i}' for i in range(1, 10))}
    invalid = any(part in {'', '.', '..'} or part.endswith(('.', ' '))
                  or part.split('.')[0].upper() in reserved for part in parts)
    if invalid or re.search(r'[\\<>:"|?*\x00-\x1f]', value):
        raise ValueError('Unsafe manifest path; require a relative POSIX path without traversal')
    return PurePosixPath(value)


def _entry_problems(item: dict) -> list[str]:
    reasons = []
    if type(item.get('size')) is not int or item['size'] <= 0:
        reasons.append('invalid_size')
    algorithm = item.get('checksum_algorithm')
    lengths = {'sha256': 64, 'git': 40}
    checksum = item.get('checksum')
    if algorithm not in lengths or not isinstance(checksum, str) or not re.fullmatch(
            r'[a-fA-F0-9]{' + str(lengths.get(algorithm, 1)) + '}', checksum):
        reasons.append('invalid_checksum')
    try:
        parsed = urlsplit(item.get('bytes_url', ''))
        valid = (parsed.scheme in {'https', 'http'} and parsed.hostname
                 and not parsed.query and not parsed.fragment
                 and not parsed.username and not parsed.password)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        reasons.append('invalid_stable_bytes_url')
    return reasons


def _plan_item(item: dict, subject_id: str | None) -> dict:
    # The manifest's temporary signed `url` is deliberately never copied or used.
    return {'subject_id': subject_id, **{name: item[name] for name in
            ('path', 'size', 'checksum_algorithm', 'checksum', 'bytes_url')}}


def build_download_plan(manifest_path: Path, limit: int = 150) -> dict:
    """Select numeric-first subjects using manifest evidence, never local absence."""
    if type(limit) is not int or limit <= 0:
        raise ValueError('limit must be a positive integer')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, list) or not all(isinstance(item, dict) for item in manifest):
        raise ValueError('Manifest must be a list of file records')
    suffixes = {'eeg': '_eeg.eeg', 'vhdr': '_eeg.vhdr', 'vmrk': '_eeg.vmrk',
                'events_tsv': '_events.tsv', 'channels_tsv': '_channels.tsv',
                'eeg_json': '_eeg.json', 'events_json': '_events.json'}
    required = ('eeg', 'vhdr', 'vmrk', 'events_tsv')
    shared_names = ('participants.tsv', 'participants.json', 'dataset_description.json')
    subjects, shared = {}, {name: [] for name in shared_names}
    for item in manifest:
        path = _safe_relative_path(item.get('path'))
        if path.as_posix() in shared:
            shared[path.as_posix()].append(item)
        if len(path.parts) == 3 and re.fullmatch(r'sub-\d+', path.parts[0]) and path.parts[1] == 'eeg':
            subjects.setdefault(path.parts[0], []).append(item)
    shared_files = []
    for name, items in shared.items():
        if len(items) > 1 or (items and _entry_problems(items[0])):
            raise ValueError(f'Invalid or duplicate shared metadata: {name}')
        shared_files.extend(_plan_item(item, None) for item in items)
    eligible, excluded, metadata_missing = [], [], {}
    files_by_subject = {}
    for subject in sorted(subjects, key=lambda value: (int(value[4:]), value)):
        entries = subjects[subject]
        duplicate = [name for name, count in Counter(item['path'] for item in entries).items() if count > 1]
        reasons = ['duplicate_path' for _ in duplicate]
        grouped = {kind: [] for kind in suffixes}
        prefixes = set()
        for item in entries:
            name = PurePosixPath(item['path']).name
            for kind, suffix in suffixes.items():
                if not name.endswith(suffix) or (kind == 'events_tsv' and name.endswith('_eeg_events.tsv')):
                    continue
                grouped[kind].append(item)
                prefix = name[:-len(suffix)]
                prefixes.add(prefix)
                if not prefix.startswith(subject + '_'):
                    reasons.append('wrong_subject_prefix')
                reasons.extend(kind + ':' + reason for reason in _entry_problems(item))
        reasons.extend('missing_' + kind for kind in required if not grouped[kind])
        reasons.extend('multiple_' + kind for kind, items in grouped.items() if len(items) > 1)
        if len(prefixes) > 1:
            reasons.append('inconsistent_file_prefix')
        if reasons:
            excluded.append({'subject_id': subject, 'reasons': sorted(set(reasons))})
            continue
        eligible.append(subject)
        missing = [kind for kind in ('channels_tsv', 'eeg_json', 'events_json') if not grouped[kind]]
        if missing:
            metadata_missing[subject] = missing
        files_by_subject[subject] = [_plan_item(item, subject) for items in grouped.values() for item in items]
    selected = eligible[:limit]
    files = [item for subject in selected for item in files_by_subject[subject]] + shared_files
    size = sum(item['size'] for item in files)
    return {'selected_subjects': selected, 'excluded_subjects': excluded,
            'processing_metadata_missing': {subject: metadata_missing[subject] for subject in selected if subject in metadata_missing},
            'eligible_not_selected': eligible[limit:], 'files': files,
            'totals': {'subjects_seen': len(subjects), 'subjects_eligible': len(eligible),
                       'subjects_selected': len(selected), 'subjects_excluded': len(excluded),
                       'files': len(files), 'shared_files': len(shared_files),
                       'bytes': size, 'gib': size / 1024 ** 3}}


def _destination_paths(item: dict, eeg_root: Path) -> tuple[Path, Path]:
    relative = _safe_relative_path(item.get('path'))
    problems = _entry_problems(item)
    if problems:
        raise ValueError('Invalid download record: ' + ', '.join(problems))
    root = Path(eeg_root).resolve()
    final = root.joinpath(*relative.parts)
    part = final.with_name(final.name + '.part')
    for candidate in (final, part):
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError('Unsafe download path resolves outside destination or is a symbolic link')
        if candidate.exists() and not candidate.is_file():
            raise ValueError('Download path already exists as a non-file')
    return final, part


def _verify_checksum(path: Path, item: dict) -> None:
    size = path.stat().st_size
    if size != item['size']:
        raise ValueError(f'{item["path"]}: File size mismatch: expected {item["size"]}, found {size}')
    digest = hashlib.sha256() if item['checksum_algorithm'] == 'sha256' else hashlib.sha1()
    if item['checksum_algorithm'] == 'git':
        digest.update(b'blob ' + str(size).encode('ascii') + b'\0')
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest().lower() != item['checksum'].lower():
        raise ValueError(f'{item["path"]}: File checksum mismatch; preserve file for diagnosis before retrying')


def _response_layout(response, offset: int, total: int) -> tuple[int, str]:
    status = response.getcode()
    encoding = response.headers.get('Content-Encoding', 'identity').lower()
    if encoding != 'identity':
        raise ValueError('Unexpected Content-Encoding; byte-range integrity requires identity')
    if status == 206:
        value = response.headers.get('Content-Range', '')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
        expected = (offset, total - 1, total)
        if match is None or tuple(map(int, match.groups())) != expected:
            raise ValueError('Invalid Content-Range start, end or total for requested file')
        start, action = offset, 'resumed' if offset else 'downloaded'
    elif status == 200:
        start, action = 0, 'restarted' if offset else 'downloaded'
    else:
        raise ValueError(f'Unexpected HTTP status: {status}')
    length = response.headers.get('Content-Length')
    if (status == 206 and length is None) or (length is not None and
            (not length.isdigit() or int(length) != total - start)):
        raise ValueError('Invalid Content-Length for requested file or range')
    return start, action


class _TransientNetworkError(OSError):
    def __init__(self, message: str, bytes_transferred: int = 0):
        super().__init__(message)
        self.bytes_transferred = bytes_transferred


def _transfer_error(exc: Exception, context: str, bytes_transferred: int = 0) -> OSError:
    """Classify transport errors before sanitizing messages that may contain URLs."""
    if isinstance(exc, HTTPError):
        transient = exc.code in {408, 429, 500, 502, 503, 504}
        reason = f'HTTP {exc.code}'
        exc.close()
    else:
        cause = exc.reason if isinstance(exc, URLError) else exc
        network_errnos = {errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
                          errno.ECONNREFUSED, errno.EPIPE, errno.ENETDOWN,
                          errno.ENETUNREACH, errno.EHOSTUNREACH}
        transient = (isinstance(cause, (TimeoutError, ConnectionError, IncompleteRead,
                                       RemoteDisconnected, ssl.SSLEOFError))
                     or isinstance(cause, socket.gaierror) and cause.errno == socket.EAI_AGAIN
                     or isinstance(cause, OSError) and cause.errno in network_errnos
                     or isinstance(cause, OSError) and getattr(cause, 'winerror', None)
                     in {10050, 10051, 10052, 10053, 10054, 10060, 10061, 10064, 10065})
        reason = type(cause).__name__
    message = f'{context} ({reason}); partial file retained'
    return (_TransientNetworkError(message, bytes_transferred) if transient else OSError(message))


def _notify(progress, kind: str, item: dict, **details) -> None:
    if progress is not None:
        progress({'type': kind, 'path': item['path'], **details})


def _promote_partial(part: Path, final: Path) -> None:
    """Publish verified bytes atomically without replacing an existing name."""
    if os.name == 'nt':
        os.rename(part, final)  # Windows rename fails if final already exists.
        return
    os.link(part, final)  # POSIX rename would replace final; link fails instead.
    part.unlink()


def _write_chunk(handle, chunk: bytes, written: int, item: dict, progress) -> int:
    handle.write(chunk)
    written += len(chunk)
    if written > item['size']:
        raise ValueError('Downloaded body exceeds expected size; partial file retained')
    _notify(progress, 'progress', item, downloaded_bytes=written, total_bytes=item['size'])
    return written


def _download_once(item: dict, eeg_root: Path, *, opener, timeout: float,
                   chunk_size: int, progress) -> dict:
    final, part = _destination_paths(item, eeg_root)
    summary = {'subject_id': item.get('subject_id'), 'path': item['path'],
               'size': item['size'], 'checksum_algorithm': item['checksum_algorithm'],
               'bytes_transferred': 0}
    if final.exists():
        if final.stat().st_size != item['size']:
            raise ValueError(f'{item["path"]}: Existing final file has wrong size; it is not overwritten. '
                             'If it is a known partial download, explicitly rename it to .part before retrying')
        _verify_checksum(final, item)
        return {**summary, 'status': 'skipped'}
    offset = part.stat().st_size if part.exists() else 0
    if offset > item['size']:
        raise ValueError('Existing partial file exceeds expected size; preserve it for diagnosis')
    _notify(progress, 'progress', item, downloaded_bytes=offset, total_bytes=item['size'])
    if offset == item['size']:
        _verify_checksum(part, item)
        _promote_partial(part, final)
        return {**summary, 'status': 'completed_partial'}
    headers = {'Accept-Encoding': 'identity'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    request = Request(item['bytes_url'], headers=headers)
    try:
        response = opener(request, timeout=timeout)
    except (OSError, HTTPException, ValueError) as exc:
        raise _transfer_error(exc, 'Download connection failed') from None
    with response:
        start, action = _response_layout(response, offset, item['size'])
        final.parent.mkdir(parents=True, exist_ok=True)
        written = start
        _notify(progress, 'progress', item, downloaded_bytes=start, total_bytes=item['size'])
        with part.open('ab' if start else 'wb') as handle:
            while True:
                # Only network reads enter the retry classifier; disk writes never do.
                try:
                    chunk = response.read(chunk_size)
                except (OSError, HTTPException) as exc:
                    if isinstance(exc, IncompleteRead) and exc.partial:
                        written = _write_chunk(handle, exc.partial, written, item, progress)
                    raise _transfer_error(exc, 'Download interrupted', written - start) from None
                if not chunk:
                    break
                written = _write_chunk(handle, chunk, written, item, progress)
            handle.flush()
            os.fsync(handle.fileno())
    if written != item['size']:
        raise _TransientNetworkError(
            f'Download interrupted: received {written} of {item["size"]} bytes; partial file retained',
            written - start)
    _verify_checksum(part, item)
    _promote_partial(part, final)
    return {**summary, 'status': action, 'bytes_transferred': written - start}


def _validate_options(timeout, chunk_size, max_retries, retry_delay) -> None:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if type(max_retries) is not int or not 0 <= max_retries <= 5:
        raise ValueError('max_retries must be an integer between 0 and 5')
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0
               for value in (timeout, retry_delay)):
        raise ValueError('timeout and retry_delay must be finite and positive')


def download_file(item: dict, eeg_root: Path, *, opener=urlopen,
                  timeout: float = 180, chunk_size: int = 1024 * 1024,
                  max_retries: int = 5, retry_delay: float = 2.0,
                  sleep=time.sleep, progress=None) -> dict:
    """Retry only transient transport failures, always resuming from retained bytes."""
    _validate_options(timeout, chunk_size, max_retries, retry_delay)
    _notify(progress, 'file_start', item, total_bytes=item['size'])
    history, transferred = [], 0
    for attempt in range(1, max_retries + 2):
        try:
            result = _download_once(item, eeg_root, opener=opener, timeout=timeout,
                                    chunk_size=chunk_size, progress=progress)
        except (ValueError, OSError) as exc:
            exc.attempts, exc.retry_history = attempt, list(history)
            if not isinstance(exc, _TransientNetworkError) or attempt > max_retries:
                _notify(progress, 'file_failed', item, attempts=attempt, reason=str(exc))
                raise
            transferred += exc.bytes_transferred
            retry = {'retry': attempt, 'max_retries': max_retries,
                     'delay': retry_delay * 2 ** (attempt - 1), 'reason': str(exc)}
            history.append(retry)
            _notify(progress, 'retry', item, **retry)
            sleep(retry['delay'])
            continue
        result.update(attempts=attempt, retry_history=list(history),
                      bytes_transferred=transferred + result['bytes_transferred'])
        _notify(progress, 'file_complete', item, status=result['status'],
                downloaded_bytes=item['size'], total_bytes=item['size'], attempts=attempt)
        return result


def download_plan(plan: dict, eeg_root: Path, log_path: Path, *, opener=urlopen,
                  timeout: float = 180, chunk_size: int = 1024 * 1024,
                  max_retries: int = 5, retry_delay: float = 2.0,
                  sleep=time.sleep, progress=None) -> dict:
    """Execute files sequentially; log attempts and terminal status without URLs."""
    _validate_options(timeout, chunk_size, max_retries, retry_delay)
    files = plan['files']
    paths = [_destination_paths(item, eeg_root) for item in files]
    destinations = [final.resolve() for final, _part in paths]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Download plan contains duplicate destination paths')
    log_path = Path(log_path)
    if log_path.resolve() in {path.resolve() for pair in paths for path in pair}:
        raise ValueError('Download log must not overwrite a planned data file')
    results = []
    report = {'complete': False, 'completed_files': 0, 'files': results}
    for item in files:
        try:
            result = download_file(item, eeg_root, opener=opener, timeout=timeout,
                                   chunk_size=chunk_size, max_retries=max_retries,
                                   retry_delay=retry_delay, sleep=sleep, progress=progress)
        except (ValueError, OSError) as exc:
            results.append({'subject_id': item.get('subject_id'), 'path': item['path'],
                            'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc),
                            'attempts': getattr(exc, 'attempts', 1),
                            'retry_history': getattr(exc, 'retry_history', [])})
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            raise
        results.append({**result, 'time_utc': datetime.now(timezone.utc).isoformat()})
        report['completed_files'] += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report['complete'] = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


class _ConsoleProgress:
    def __init__(self, stream):
        self.stream = stream
        self.last_progress = -math.inf

    def __call__(self, event: dict) -> None:
        kind, path = event['type'], event['path']
        if kind == 'progress':
            now = time.monotonic()
            done, total = event['downloaded_bytes'], event['total_bytes']
            if now - self.last_progress < 1.0 and done != total:
                return
            self.last_progress = now
            message = f'PROGRESS {path}: {done}/{total} bytes ({100 * done / total:.1f}%)'
        elif kind == 'retry':
            self.last_progress = -math.inf
            message = (f'RETRY {event["retry"]}/{event["max_retries"]} {path}: '
                       f'{event["reason"]}; waiting {event["delay"]:g}s')
        elif kind == 'file_start':
            self.last_progress = -math.inf
            message = f'FILE {path}: checking existing files'
        elif kind == 'file_complete':
            message = f'VERIFIED {path}: {event["status"]}'
        else:
            message = f'FAILED {path}: {event["reason"]}'
        print(message, file=self.stream, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Plan LEMON downloads; network requires --download')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=150)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('outputs_download') / 'plan.json')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--timeout', type=float, default=180, help='Socket timeout in seconds (default: 180)')
    parser.add_argument('--max-retries', type=int, choices=range(6), default=5,
                        help='Additional retries per file, 0-5 (default: 5)')
    parser.add_argument('--retry-delay', type=float, default=2.0,
                        help='Initial retry wait in seconds; doubles each retry (default: 2)')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args(argv)
    try:
        _validate_options(args.timeout, 1024 * 1024, args.max_retries, args.retry_delay)
        plan = build_download_plan(args.manifest, limit=args.limit)
        protected = {args.manifest.resolve()}
        protected.update(path.resolve() for item in plan['files']
                         for path in _destination_paths(item, args.destination))
        log_path = args.log or args.plan.parent / 'download_log.json'
        if args.plan.resolve() in protected or log_path.resolve() in protected:
            raise ValueError('Plan/log path must not overwrite the source manifest or destination data')
        if args.plan.resolve() == log_path.resolve():
            raise ValueError('Plan and download log must use different paths')
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        totals = plan['totals']
        print(f"Plan: {totals['subjects_selected']} subjects, {totals['files']} files, {totals['gib']:.3f} GiB")
        print(f'Plan file: {args.plan.resolve()}')
        print(f'Destination: {args.destination.resolve()}')
        if args.download:
            result = download_plan(plan, args.destination, log_path, timeout=args.timeout,
                                   max_retries=args.max_retries, retry_delay=args.retry_delay,
                                   progress=_ConsoleProgress(sys.stdout))
            print(f"Completed: {result['completed_files']} files")
        else:
            print('Plan only; no network download started. Use --download to transfer files.')
    except (ValueError, OSError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
