"""Download planning and transfer tests use synthetic manifests and localhost only."""
from __future__ import annotations

import hashlib
import importlib
import contextlib
import errno
import inspect
import io
from http.client import IncompleteRead, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import ssl
import tempfile
import unittest
from unittest.mock import patch
from threading import Thread
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def manifest_item(path, body=b'example'):
    return {'path': path, 'size': len(body), 'checksum_algorithm': 'sha256',
            'checksum': hashlib.sha256(body).hexdigest(),
            'bytes_url': 'https://example.test/bytes/' + path,
            'url': 'https://example.test/temporary?signature=DO_NOT_USE'}


def subject_items(subject, optional=True):
    suffixes = ['_eeg.eeg', '_eeg.vhdr', '_eeg.vmrk', '_events.tsv']
    if optional:
        suffixes += ['_channels.tsv', '_eeg.json', '_events.json']
    return [manifest_item(f'{subject}/eeg/{subject}_task-resting{suffix}')
            for suffix in suffixes]


class DownloadPlanTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('empirical_analysis.lemon.download')
        except ModuleNotFoundError as exc:
            self.fail('The download module is not implemented: ' + str(exc))

    def plan(self, items, limit=150):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps(items), encoding='utf-8')
            return self.module().build_download_plan(path, limit=limit)

    def test_selects_numeric_order_and_includes_metadata_without_signed_url(self):
        items = subject_items('sub-20') + subject_items('sub-3')
        items += [manifest_item('participants.tsv'), manifest_item('participants.json'),
                  manifest_item('dataset_description.json'), manifest_item('README.md')]
        plan = self.plan(items, limit=1)
        self.assertEqual(plan['selected_subjects'], ['sub-3'])
        self.assertEqual(len(plan['files']), 10)
        self.assertEqual(plan['totals']['bytes'], 70)
        self.assertEqual(plan['totals']['subjects_selected'], 1)
        self.assertNotIn('DO_NOT_USE', json.dumps(plan))
        self.assertTrue(all('url' not in item for item in plan['files']))

    def test_optional_channels_and_metadata_absence_does_not_exclude_subject(self):
        plan = self.plan(subject_items('sub-1', optional=False))
        self.assertEqual(plan['selected_subjects'], ['sub-1'])
        self.assertEqual(len(plan['files']), 4)
        self.assertEqual(plan['excluded_subjects'], [])
        self.assertEqual(set(plan['processing_metadata_missing']['sub-1']),
                         {'channels_tsv', 'eeg_json', 'events_json'})

    def test_manifest_git_blob_checksums_are_supported_for_metadata(self):
        items = subject_items('sub-1')
        items[-1]['checksum_algorithm'] = 'git'
        items[-1]['checksum'] = hashlib.sha1(b'blob 7\0example').hexdigest()
        plan = self.plan(items)
        self.assertEqual(plan['selected_subjects'], ['sub-1'])
        self.assertEqual(plan['files'][-1]['checksum_algorithm'], 'git')

    def test_missing_required_file_excludes_subject_without_consulting_local_data(self):
        plan = self.plan(subject_items('sub-1')[1:] + subject_items('sub-2'))
        self.assertEqual(plan['selected_subjects'], ['sub-2'])
        self.assertIn('missing_eeg', plan['excluded_subjects'][0]['reasons'])

    def test_duplicate_path_and_multiple_same_kind_are_explicit_exclusions(self):
        first = subject_items('sub-1')
        duplicate = first + [first[0]]
        second = subject_items('sub-2')
        second.append(manifest_item('sub-2/eeg/sub-2_task-other_eeg.eeg'))
        plan = self.plan(duplicate + second + subject_items('sub-3'))
        self.assertEqual(plan['selected_subjects'], ['sub-3'])
        self.assertTrue(any('duplicate_path' in reason for reason in plan['excluded_subjects'][0]['reasons']))
        self.assertIn('multiple_eeg', plan['excluded_subjects'][1]['reasons'])

    def test_wrong_subject_prefix_is_not_assigned_to_subject(self):
        items = subject_items('sub-1')
        items[0]['path'] = 'sub-1/eeg/sub-2_task-resting_eeg.eeg'
        plan = self.plan(items)
        self.assertEqual(plan['selected_subjects'], [])
        self.assertTrue(any('prefix' in reason for reason in plan['excluded_subjects'][0]['reasons']))

    def test_invalid_integrity_or_unstable_bytes_url_excludes_subject(self):
        cases = [('size', 0), ('size', True), ('checksum_algorithm', 'md5'),
                 ('checksum', 'zz' * 32), ('bytes_url', ''),
                 ('bytes_url', 'https://example.test/file?X-Amz-Signature=secret'),
                 ('bytes_url', 'file:///secret')]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                items = subject_items('sub-1')
                items[0][field] = value
                plan = self.plan(items)
                self.assertEqual(plan['selected_subjects'], [])
                self.assertTrue(plan['excluded_subjects'][0]['reasons'])
                self.assertNotIn('secret', json.dumps(plan))

    def test_manifest_path_traversal_is_rejected_before_selection(self):
        for path in ('../escape.eeg', '/absolute.eeg', 'C:/escape.eeg',
                     'sub-1/eeg/../../escape.eeg', 'sub-1\\eeg\\escape.eeg',
                     'sub-1/eeg/a:stream.eeg', 'sub-1/eeg/a./file.eeg'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'path'):
                self.plan([manifest_item(path)])

    def test_invalid_limit_and_ambiguous_shared_metadata_are_errors(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.plan(subject_items('sub-1'), limit=limit)
        shared = manifest_item('participants.tsv')
        with self.assertRaisesRegex(ValueError, 'shared'):
            self.plan(subject_items('sub-1') + [shared, shared])

    def test_similar_non_subject_and_non_eeg_directories_do_not_add_subjects(self):
        items = subject_items('sub-1') + subject_items('sub-2x')
        items += [manifest_item('sub-3/anat/sub-3_task-resting_eeg.eeg')]
        self.assertEqual(self.plan(items)['selected_subjects'], ['sub-1'])

    def test_cli_default_only_writes_plan_and_does_not_start_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(subject_items('sub-1')), encoding='utf-8')
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                result = self.module().main(['--manifest', str(manifest), '--destination',
                                             str(root / 'data'), '--plan', str(root / 'plan.json')])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads((root / 'plan.json').read_text())['selected_subjects'], ['sub-1'])
            self.assertFalse((root / 'data').exists())
            self.assertNotIn('https://', console.getvalue())
            self.assertNotIn('DO_NOT_USE', console.getvalue())

    def test_cli_plan_cannot_overwrite_source_manifest_or_destination_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            original = json.dumps(subject_items('sub-1'))
            manifest.write_text(original, encoding='utf-8')
            destination = root / 'data'
            data_file = destination / subject_items('sub-1')[0]['path']
            data_file.parent.mkdir(parents=True)
            data_file.write_bytes(b'original')
            for unsafe_plan in (manifest, data_file):
                with self.subTest(path=unsafe_plan), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    result = self.module().main(['--manifest', str(manifest), '--destination',
                                                 str(destination), '--plan', str(unsafe_plan)])
                self.assertEqual(result, 2)
                self.assertEqual(manifest.read_text(), original)
                self.assertEqual(data_file.read_bytes(), b'original')

    def test_cli_passes_retry_defaults_and_prints_file_progress_and_retry_status(self):
        module = self.module()
        self.assertIn('max_retries', inspect.signature(module.download_plan).parameters,
                      'CLI retry configuration is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(subject_items('sub-1')), encoding='utf-8')
            captured = {}

            def fake_transfer(plan, _destination, _log_path, **kwargs):
                captured.update(kwargs)
                callback = kwargs['progress']
                path = plan['files'][0]['path']
                callback({'type': 'file_start', 'path': path})
                callback({'type': 'progress', 'path': path, 'downloaded_bytes': 1, 'total_bytes': 2})
                callback({'type': 'retry', 'path': path, 'retry': 1, 'max_retries': 5,
                          'delay': 2.0, 'reason': 'TimeoutError'})
                callback({'type': 'file_complete', 'path': path, 'status': 'downloaded'})
                return {'completed_files': 1}

            console = io.StringIO()
            with patch.object(module, 'download_plan', side_effect=fake_transfer), contextlib.redirect_stdout(console):
                result = module.main(['--manifest', str(manifest), '--destination', str(root / 'data'),
                                      '--plan', str(root / 'plan.json'), '--download'])
            self.assertEqual(result, 0)
            self.assertEqual(captured['timeout'], 180)
            self.assertEqual(captured['max_retries'], 5)
            self.assertEqual(captured['retry_delay'], 2.0)
            self.assertIn('sub-1/eeg/sub-1_task-resting_eeg.eeg', console.getvalue())
            self.assertIn('50', console.getvalue())
            self.assertIn('retry', console.getvalue().lower())
            self.assertIn('1/5', console.getvalue())
            self.assertNotIn('https://', console.getvalue())

    def test_cli_explicit_retry_settings_reach_download_plan(self):
        module = self.module()
        self.assertIn('max_retries', inspect.signature(module.download_plan).parameters,
                      'CLI retry configuration is not implemented')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(subject_items('sub-1')), encoding='utf-8')
            with patch.object(module, 'download_plan', return_value={'completed_files': 0}) as transfer:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = module.main(['--manifest', str(manifest), '--destination', str(root / 'data'),
                                          '--plan', str(root / 'plan.json'), '--download', '--timeout', '90',
                                          '--max-retries', '3', '--retry-delay', '0.25'])
            self.assertEqual(result, 0)
            self.assertEqual(transfer.call_args.kwargs['timeout'], 90)
            self.assertEqual(transfer.call_args.kwargs['max_retries'], 3)
            self.assertEqual(transfer.call_args.kwargs['retry_delay'], 0.25)


@contextlib.contextmanager
def transfer_server(body, mode='normal'):
    state = {'requests': [], 'mode': mode}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            range_header = self.headers.get('Range')
            state['requests'].append(range_header)
            start = int(range_header[6:-1]) if range_header else 0
            active_mode = state['mode'].pop(0) if isinstance(state['mode'], list) else state['mode']
            if isinstance(active_mode, int):
                self.send_response(active_mode)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            partial = bool(range_header) and active_mode != 'ignore_range'
            start = start if partial else 0
            data = body[start:]
            self.send_response(206 if partial else 200)
            if partial:
                reported_start = start + (1 if active_mode == 'bad_range' else 0)
                self.send_header('Content-Range', f'bytes {reported_start}-{len(body)-1}/{len(body)}')
            length = len(data) + (1 if active_mode == 'bad_length' else 0)
            self.send_header('Content-Length', str(length))
            self.end_headers()
            if active_mode == 'interrupt' and len(state['requests']) == 1:
                data = data[:13]
            self.wfile.write(data)
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/data', state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class DownloadTransferTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('empirical_analysis.lemon.download')
        self.assertTrue(hasattr(self.module, 'download_file'), 'download_file is not implemented')
        self.body = b'example EEG bytes\0' * 19
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'data'
        self.item = manifest_item('sub-1/eeg/sub-1_task-resting_eeg.eeg', self.body)
        self.final = self.root / self.item['path']
        self.part = self.final.with_name(self.final.name + '.part')

    def existing(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def require_retry_api(self, function=None):
        parameters = inspect.signature(function or self.module.download_file).parameters
        for name in ('max_retries', 'retry_delay', 'sleep', 'progress'):
            self.assertIn(name, parameters, f'Downloader retry API is missing {name}')

    def test_retry_defaults_are_180_seconds_and_five_extra_attempts(self):
        for function in (self.module.download_file, self.module.download_plan):
            with self.subTest(function=function.__name__):
                self.require_retry_api(function)
                parameters = inspect.signature(function).parameters
                self.assertEqual(parameters['timeout'].default, 180)
                self.assertEqual(parameters['max_retries'].default, 5)
                self.assertEqual(parameters['retry_delay'].default, 2.0)

    def test_connection_timeout_retries_then_downloads_with_status(self):
        self.require_retry_api()
        waits, events, timeouts = [], [], []

        def first_timeout(request, *, timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise URLError(TimeoutError('temporary connection timeout'))
            return urlopen(request, timeout=timeout)

        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root, opener=first_timeout,
                                               sleep=waits.append, progress=events.append)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(timeouts, [180, 180])
        self.assertEqual(waits, [2.0])
        self.assertEqual(state['requests'], [None])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(len(result['retry_history']), 1)
        retry = next(event for event in events if event['type'] == 'retry')
        self.assertEqual((retry['path'], retry['retry'], retry['max_retries'], retry['delay']),
                         (self.item['path'], 1, 5, 2.0))
        self.assertTrue(retry['reason'])

    def test_repeated_timeout_stops_after_six_attempts_and_increasing_waits(self):
        self.require_retry_api()
        self.existing(self.part, self.body[:11])
        requests, waits, events = [], [], []

        def timeout_opener(request, **_kwargs):
            requests.append(request.get_header('Range'))
            raise TimeoutError('https://example.test/private?signature=DO_NOT_USE')

        with self.assertRaises(OSError) as caught:
            self.module.download_file(self.item, self.root, opener=timeout_opener,
                                      sleep=waits.append, progress=events.append)
        self.assertEqual(requests, ['bytes=11-'] * 6)
        self.assertEqual(waits, [2.0, 4.0, 8.0, 16.0, 32.0])
        self.assertEqual(self.part.read_bytes(), self.body[:11])
        self.assertFalse(self.final.exists())
        retries = [event for event in events if event['type'] == 'retry']
        self.assertEqual([event['retry'] for event in retries], [1, 2, 3, 4, 5])
        self.assertNotIn('DO_NOT_USE', str(caught.exception) + json.dumps(events))

    def test_automatic_retry_resumes_new_partial_offset_and_reports_progress(self):
        self.require_retry_api()
        waits, events = [], []
        with transfer_server(self.body, 'interrupt') as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root, chunk_size=8,
                                               sleep=waits.append, progress=events.append)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(state['requests'], [None, 'bytes=13-'])
        self.assertEqual(waits, [2.0])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['bytes_transferred'], len(self.body))
        self.assertEqual(events[0]['type'], 'file_start')
        self.assertEqual(events[-1]['type'], 'file_complete')
        self.assertTrue(all(event['path'] == self.item['path'] for event in events))
        progress = [event for event in events if event['type'] == 'progress']
        self.assertIn(13, [event['downloaded_bytes'] for event in progress])
        self.assertEqual(progress[-1]['downloaded_bytes'], len(self.body))
        self.assertTrue(all(event['total_bytes'] == len(self.body) for event in progress))

    def test_existing_partial_progress_starts_at_resume_offset(self):
        self.require_retry_api()
        self.existing(self.part, self.body[:11])
        events = []
        with transfer_server(self.body) as (url, _state):
            self.item['bytes_url'] = url
            self.module.download_file(self.item, self.root, progress=events.append)
        progress = [event for event in events if event['type'] == 'progress']
        self.assertEqual(progress[0]['downloaded_bytes'], 11)
        self.assertEqual(progress[-1]['downloaded_bytes'], len(self.body))

    def test_incomplete_read_preserves_exception_partial_bytes_before_resume(self):
        self.require_retry_api()
        attempts, waits = [], []

        class InterruptedResponse:
            def __init__(wrapped, response):
                wrapped.response = response
                wrapped.headers = response.headers

            def __enter__(wrapped):
                return wrapped

            def __exit__(wrapped, *_args):
                wrapped.response.close()

            def getcode(wrapped):
                return wrapped.response.getcode()

            def read(wrapped, _size):
                raise IncompleteRead(self.body[:13], len(self.body) - 13)

        def first_incomplete(request, *, timeout):
            attempts.append(request.get_header('Range'))
            response = urlopen(request, timeout=timeout)
            return InterruptedResponse(response) if len(attempts) == 1 else response

        with transfer_server(self.body) as (url, _state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root, opener=first_incomplete,
                                               sleep=waits.append)
        self.assertEqual(attempts, [None, 'bytes=13-'])
        self.assertEqual(waits, [2.0])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['bytes_transferred'], len(self.body))
        self.assertEqual(self.final.read_bytes(), self.body)

    def test_timeout_during_response_read_resumes_from_flushed_bytes(self):
        self.require_retry_api()
        attempts, waits = [], []

        class TimedOutResponse:
            def __init__(wrapped, response):
                wrapped.response = response
                wrapped.headers = response.headers
                wrapped.reads = 0

            def __enter__(wrapped):
                return wrapped

            def __exit__(wrapped, *_args):
                wrapped.response.close()

            def getcode(wrapped):
                return wrapped.response.getcode()

            def read(wrapped, _size):
                wrapped.reads += 1
                if wrapped.reads == 1:
                    return wrapped.response.read(13)
                raise TimeoutError('response stalled')

        def first_read_timeout(request, *, timeout):
            attempts.append(request.get_header('Range'))
            response = urlopen(request, timeout=timeout)
            return TimedOutResponse(response) if len(attempts) == 1 else response

        with transfer_server(self.body) as (url, _state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root, opener=first_read_timeout,
                                               sleep=waits.append)
        self.assertEqual(attempts, [None, 'bytes=13-'])
        self.assertEqual(waits, [2.0])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['bytes_transferred'], len(self.body))
        self.assertEqual(self.final.read_bytes(), self.body)

    def test_transient_http_status_retries_and_keeps_range(self):
        self.require_retry_api()
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                part = (root / self.item['path']).with_suffix('.eeg.part')
                self.existing(part, self.body[:11])
                waits = []
                with transfer_server(self.body, [status, 'normal']) as (url, state):
                    item = dict(self.item, bytes_url=url)
                    result = self.module.download_file(item, root, sleep=waits.append)
                self.assertEqual(state['requests'], ['bytes=11-', 'bytes=11-'])
                self.assertEqual(waits, [2.0])
                self.assertEqual(result['attempts'], 2)
                self.assertEqual((root / self.item['path']).read_bytes(), self.body)

    def test_permanent_http_status_stops_without_retry(self):
        self.require_retry_api()
        for status in (403, 404):
            with self.subTest(status=status):
                waits, events = [], []
                with transfer_server(self.body, status) as (url, state):
                    self.item['bytes_url'] = url
                    with self.assertRaises(OSError):
                        self.module.download_file(self.item, self.root, sleep=waits.append,
                                                  progress=events.append)
                self.assertEqual(state['requests'], [None])
                self.assertEqual(waits, [])
                self.assertFalse(any(event['type'] == 'retry' for event in events))

    def test_transient_connection_and_dns_errors_are_retried(self):
        self.require_retry_api()
        faults = (ConnectionResetError('reset'), ConnectionAbortedError('abort'),
                  ConnectionRefusedError('refused'), URLError(socket.gaierror(socket.EAI_AGAIN, 'dns busy')),
                  OSError(errno.ENETUNREACH, 'network unavailable'), RemoteDisconnected('peer closed'))
        for fault in faults:
            with self.subTest(fault=type(fault).__name__), tempfile.TemporaryDirectory() as tmp:
                attempts, waits = [], []

                def first_failure(request, *, timeout):
                    attempts.append(request)
                    if len(attempts) == 1:
                        raise fault
                    return urlopen(request, timeout=timeout)

                with transfer_server(self.body) as (url, _state):
                    result = self.module.download_file(dict(self.item, bytes_url=url), Path(tmp),
                                                       opener=first_failure, sleep=waits.append)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(waits, [2.0])
                self.assertEqual(result['attempts'], 2)

    def test_certificate_permanent_dns_and_nonnetwork_os_errors_do_not_retry(self):
        self.require_retry_api()
        faults = (URLError(ssl.SSLCertVerificationError('certificate rejected')),
                  URLError(socket.gaierror(socket.EAI_NONAME, 'host does not exist')),
                  PermissionError(errno.EACCES, 'access denied'),
                  OSError(errno.ENOSPC, 'disk full'), OSError('unclassified error'))
        for fault in faults:
            with self.subTest(fault=str(fault)):
                attempts, waits = [], []

                def failing_opener(*_args, **_kwargs):
                    attempts.append(1)
                    raise fault

                with self.assertRaises(OSError):
                    self.module.download_file(self.item, self.root, opener=failing_opener,
                                              sleep=waits.append)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(waits, [])

    def test_storage_error_preserves_partial_and_is_never_retried(self):
        self.require_retry_api()
        waits = []
        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            with patch.object(self.module.os, 'fsync', side_effect=OSError(errno.ENOSPC, 'disk full')):
                with self.assertRaises(OSError):
                    self.module.download_file(self.item, self.root, sleep=waits.append)
        self.assertEqual(state['requests'], [None])
        self.assertEqual(waits, [])
        self.assertEqual(self.part.read_bytes(), self.body)
        self.assertFalse(self.final.exists())

    def test_checksum_failure_does_not_retry_and_names_specific_file(self):
        self.require_retry_api()
        waits, events = [], []
        with transfer_server(b'x' * len(self.body)) as (url, state):
            self.item['bytes_url'] = url
            with self.assertRaisesRegex(ValueError, 'checksum') as caught:
                self.module.download_file(self.item, self.root, sleep=waits.append,
                                          progress=events.append)
        self.assertIn(self.item['path'], str(caught.exception).replace('\\', '/'))
        self.assertEqual(state['requests'], [None])
        self.assertEqual(waits, [])
        self.assertFalse(any(event['type'] == 'retry' for event in events))
        self.assertEqual(self.part.read_bytes(), b'x' * len(self.body))
        self.assertFalse(self.final.exists())

    def test_existing_final_is_verified_and_skipped_without_attempting_network(self):
        self.require_retry_api()
        self.existing(self.final, self.body)
        waits, events = [], []
        result = self.module.download_file(
            self.item, self.root, sleep=waits.append, progress=events.append,
            opener=lambda *_args, **_kwargs: self.fail('Verified final requested network'))
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['retry_history'], [])
        self.assertEqual(waits, [])
        self.assertEqual(events[-1]['type'], 'file_complete')

    def test_final_appearing_at_full_progress_is_never_overwritten(self):
        self.require_retry_api()
        waits = []

        def competing_file(event):
            if event['type'] == 'progress' and event['downloaded_bytes'] == len(self.body):
                self.existing(self.final, b'other process owns this file')

        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            with self.assertRaises(FileExistsError):
                self.module.download_file(self.item, self.root, progress=competing_file,
                                          sleep=waits.append)
        self.assertEqual(state['requests'], [None])
        self.assertEqual(waits, [])
        self.assertEqual(self.final.read_bytes(), b'other process owns this file')
        self.assertEqual(self.part.read_bytes(), self.body)

    def test_invalid_retry_settings_fail_before_touching_files(self):
        self.require_retry_api()
        settings = ({'max_retries': value} for value in (-1, 6, True, 1.5))
        invalid = list(settings) + [{name: value} for name in ('timeout', 'retry_delay')
                                    for value in (0, -1, float('nan'), float('inf'))]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.module.download_file(self.item, self.root, **options,
                                          opener=lambda *_args, **_kwargs: self.fail('Invalid options reached network'))
        self.assertFalse(self.root.exists())

    def test_plan_retries_one_file_to_completion_before_requesting_the_next(self):
        self.require_retry_api(self.module.download_plan)
        attempts, waits, events = [], [], []
        other = dict(self.item, path='sub-2/eeg/sub-2_task-resting_eeg.eeg')
        log = self.root.parent / 'sequential_log.json'

        def ordered_opener(request, *, timeout):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise TimeoutError('temporary')
            if len(attempts) == 3:
                self.assertEqual(self.final.read_bytes(), self.body)
                self.assertEqual(events[-1]['path'], other['path'])
            return urlopen(request, timeout=timeout)

        with transfer_server(self.body) as (url, _state):
            self.item['bytes_url'] = url + '/first'
            other['bytes_url'] = url + '/second'
            result = self.module.download_plan({'files': [self.item, other]}, self.root, log,
                                               opener=ordered_opener, sleep=waits.append,
                                               progress=events.append)
        self.assertEqual(attempts, [url + '/first', url + '/first', url + '/second'])
        self.assertEqual(waits, [2.0])
        self.assertTrue(result['complete'])
        self.assertEqual(result['completed_files'], 2)
        saved = json.loads(log.read_text())
        self.assertEqual([item['attempts'] for item in saved['files']], [2, 1])
        self.assertEqual(len(saved['files'][0]['retry_history']), 1)

    def test_exhausted_plan_records_failure_and_does_not_start_next_file(self):
        self.require_retry_api(self.module.download_plan)
        self.existing(self.part, self.body[:11])
        attempts, waits = [], []
        other = dict(self.item, path='sub-2/eeg/sub-2_task-resting_eeg.eeg')
        log = self.root.parent / 'failed_log.json'

        def timeout_opener(request, **_kwargs):
            attempts.append(request.full_url)
            raise TimeoutError('https://example.test/private?signature=DO_NOT_USE')

        with self.assertRaises(OSError):
            self.module.download_plan({'files': [self.item, other]}, self.root, log,
                                      opener=timeout_opener, sleep=waits.append, max_retries=2)
        saved = json.loads(log.read_text())
        self.assertEqual(attempts, [self.item['bytes_url']] * 3)
        self.assertEqual(waits, [2.0, 4.0])
        self.assertFalse(saved['complete'])
        self.assertEqual(saved['completed_files'], 0)
        self.assertEqual(len(saved['files']), 1)
        self.assertEqual(saved['files'][0]['status'], 'failed')
        self.assertEqual(saved['files'][0]['path'], self.item['path'])
        self.assertEqual(saved['files'][0]['attempts'], 3)
        self.assertEqual(len(saved['files'][0]['retry_history']), 2)
        self.assertEqual(self.part.read_bytes(), self.body[:11])
        self.assertFalse((self.root / other['path']).exists())
        self.assertNotIn('DO_NOT_USE', log.read_text())

    def test_fresh_download_is_verified_before_atomic_final_name(self):
        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root, chunk_size=8)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertFalse(self.part.exists())
        self.assertEqual(result['status'], 'downloaded')
        self.assertEqual(result['bytes_transferred'], len(self.body))
        self.assertEqual(state['requests'], [None])

    def test_existing_correct_file_skips_without_network(self):
        self.existing(self.final, self.body)
        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(state['requests'], [])

    def test_existing_final_wrong_size_is_never_overwritten_or_moved(self):
        self.existing(self.final, self.body[:9])
        with self.assertRaisesRegex(ValueError, 'existing|Existing'):
            self.module.download_file(self.item, self.root)
        self.assertEqual(self.final.read_bytes(), self.body[:9])
        self.assertFalse(self.part.exists())

    def test_existing_final_wrong_checksum_is_preserved(self):
        self.existing(self.final, b'x' * len(self.body))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.module.download_file(self.item, self.root)
        self.assertEqual(self.final.read_bytes(), b'x' * len(self.body))

    def test_partial_file_resumes_with_validated_range(self):
        self.existing(self.part, self.body[:11])
        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(result['status'], 'resumed')
        self.assertEqual(result['bytes_transferred'], len(self.body) - 11)
        self.assertEqual(state['requests'], ['bytes=11-'])

    def test_ignored_range_restarts_instead_of_appending(self):
        self.existing(self.part, self.body[:11])
        with transfer_server(self.body, 'ignore_range') as (url, state):
            self.item['bytes_url'] = url
            result = self.module.download_file(self.item, self.root)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(result['status'], 'restarted')
        self.assertEqual(result['bytes_transferred'], len(self.body))
        self.assertEqual(state['requests'], ['bytes=11-'])

    def test_invalid_range_does_not_modify_partial(self):
        self.existing(self.part, self.body[:11])
        with transfer_server(self.body, 'bad_range') as (url, _state):
            self.item['bytes_url'] = url
            with self.assertRaisesRegex(ValueError, 'Content-Range'):
                self.module.download_file(self.item, self.root)
        self.assertEqual(self.part.read_bytes(), self.body[:11])
        self.assertFalse(self.final.exists())

    def test_invalid_content_length_is_not_accepted(self):
        with transfer_server(self.body, 'bad_length') as (url, _state):
            self.item['bytes_url'] = url
            with self.assertRaisesRegex(ValueError, 'Content-Length'):
                self.module.download_file(self.item, self.root)
        self.assertFalse(self.final.exists())

    def test_interruption_keeps_partial_then_next_call_resumes(self):
        self.require_retry_api()
        with transfer_server(self.body, 'interrupt') as (url, state):
            self.item['bytes_url'] = url
            with self.assertRaises(OSError):
                self.module.download_file(self.item, self.root, chunk_size=8, max_retries=0)
            self.assertEqual(self.part.read_bytes(), self.body[:13])
            self.assertFalse(self.final.exists())
            result = self.module.download_file(self.item, self.root, chunk_size=8)
        self.assertEqual(self.final.read_bytes(), self.body)
        self.assertEqual(result['status'], 'resumed')
        self.assertEqual(state['requests'], [None, 'bytes=13-'])

    def test_bad_download_checksum_keeps_part_and_never_promotes(self):
        with transfer_server(b'x' * len(self.body)) as (url, _state):
            self.item['bytes_url'] = url
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.module.download_file(self.item, self.root)
        self.assertFalse(self.final.exists())
        self.assertEqual(self.part.read_bytes(), b'x' * len(self.body))

    def test_oversized_partial_is_not_truncated(self):
        self.existing(self.part, self.body + b'overflow')
        with self.assertRaisesRegex(ValueError, 'partial|Partial'):
            self.module.download_file(self.item, self.root)
        self.assertEqual(self.part.read_bytes(), self.body + b'overflow')

    def test_already_complete_part_is_verified_and_promoted_without_network(self):
        self.existing(self.part, self.body)
        result = self.module.download_file(self.item, self.root)
        self.assertEqual(result['status'], 'completed_partial')
        self.assertEqual(self.final.read_bytes(), self.body)

    def test_git_blob_checksum_is_verified_on_transfer_and_skip(self):
        self.item['checksum_algorithm'] = 'git'
        self.item['checksum'] = hashlib.sha1(b'blob ' + str(len(self.body)).encode() + b'\0' + self.body).hexdigest()
        with transfer_server(self.body) as (url, state):
            self.item['bytes_url'] = url
            self.module.download_file(self.item, self.root)
            result = self.module.download_file(self.item, self.root)
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(len(state['requests']), 1)

    def test_traversal_download_does_not_create_outside_file(self):
        self.item['path'] = '../escape.eeg'
        with self.assertRaisesRegex(ValueError, 'path'):
            self.module.download_file(self.item, self.root)
        self.assertFalse((self.root.parent / 'escape.eeg').exists())

    def test_plan_log_records_success_and_redacts_transfer_failure(self):
        self.assertTrue(hasattr(self.module, 'download_plan'), 'download_plan is not implemented')
        log = self.root.parent / 'download_log.json'
        with transfer_server(self.body) as (url, _state):
            self.item['bytes_url'] = url
            result = self.module.download_plan({'files': [self.item]}, self.root, log)
        self.assertEqual(result['completed_files'], 1)
        content = json.loads(log.read_text())
        self.assertEqual(content['files'][0]['status'], 'downloaded')
        self.assertNotIn('bytes_url', log.read_text())

        def failing_opener(*_args, **_kwargs):
            raise OSError('https://example.test/secret?signature=DO_NOT_USE')

        other = dict(self.item, path='sub-2/eeg/sub-2_task-resting_eeg.eeg')
        with self.assertRaises(OSError):
            self.module.download_plan({'files': [other]}, self.root, log, opener=failing_opener)
        self.assertNotIn('DO_NOT_USE', log.read_text())
        self.assertEqual(json.loads(log.read_text())['files'][0]['status'], 'failed')

    def test_log_cannot_replace_active_partial_file(self):
        self.existing(self.part, self.body[:9])
        with self.assertRaisesRegex(ValueError, 'log'):
            self.module.download_plan({'files': [self.item]}, self.root, self.part,
                                      opener=lambda *_args, **_kwargs: self.fail('Unsafe log accepted before transfer'))
        self.assertEqual(self.part.read_bytes(), self.body[:9])


if __name__ == '__main__':
    unittest.main()
