"""Fixed daily Beijing forecasting data; ZIP streams stay read-only, NA stays NA."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def aggregate_four_hours(hours):
    """Input station/hour/pollutant; output day/station/pollutant/four-hour-bin."""
    values=np.asarray(hours,float)
    if values.ndim!=3 or values.shape[1]%24: raise ValueError('Expected complete daily hourly grid')
    blocked=values.reshape(values.shape[0],-1,6,4,values.shape[2])
    counts=np.isfinite(blocked).sum(axis=3)
    means=np.divide(np.nansum(blocked,axis=3),counts,out=np.full(counts.shape,np.nan),where=counts>=2)
    return means.transpose(1,0,3,2),counts.transpose(1,0,3,2)


def daily_targets(hourly_pm25):
    daily=np.asarray(hourly_pm25,float).reshape(-1,24)
    count=np.isfinite(daily).sum(1)
    means=np.divide(np.nansum(daily,axis=1),count,out=np.full(count.shape,np.nan),where=count>0)
    return dict(y=means[1:],persistence=means[:-1],target_counts=count[1:],previous_counts=count[:-1],
                eligible=(count[1:]>=18)&(count[:-1]>=18),all_daily_counts=count)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def prepare(zip_path,out_dir):
    """Verify the entire raw archive, then save deterministic observed daily features."""
    zip_path,out_dir=Path(zip_path),Path(out_dir)
    before=zip_path.stat();archive_sha=hashlib.sha256(zip_path.read_bytes()).hexdigest()
    pollutants=['PM2.5','PM10','SO2','NO2','CO','O3']
    stations=['Aotizhongxin','Changping','Dingling','Dongsi','Guanyuan','Gucheng','Huairou',
              'Nongzhanguan','Shunyi','Tiantan','Wanliu','Wanshouxigong']
    start=dt.datetime(2013,3,1);expected=[start+dt.timedelta(hours=i) for i in range(35064)]
    tables={};identity=[];audit=[]
    with ZipFile(zip_path) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC failure')
        members=[info for info in archive.infolist() if info.filename.endswith('.csv')]
        if len(members)!=12: raise ValueError('Expected exactly 12 station CSVs')
        for info in members:
            raw=archive.read(info);rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
            names={row['station'] for row in rows}
            if len(names)!=1: raise ValueError('Ambiguous station identity')
            station=names.pop()
            if station not in stations or station in tables or station not in info.filename:
                raise ValueError('Station filename/column identity mismatch')
            dates=[dt.datetime(*[int(row[key]) for key in ['year','month','day','hour']]) for row in rows]
            if dates!=expected or [int(row['No']) for row in rows]!=list(range(1,35065)):
                raise ValueError(f'Datetime/row identity mismatch: {station}')
            values=np.asarray([[np.nan if row[key]=='NA' else float(row[key]) for key in pollutants] for row in rows])
            if np.isinf(values).any() or (values<0).any():
                raise ValueError(f'Nonfinite infinity or negative pollutant: {station}')
            tables[station]=values
            identity.append(dict(station=station,member=info.filename,bytes=len(raw),crc32=f'{info.CRC:08x}',
                                 sha256=hashlib.sha256(raw).hexdigest()))
            audit.append(dict(station=station,n_hours=len(rows),duplicate_datetime=0,missing_datetime=0,
                pollutants={poll:dict(missing_count=int(np.isnan(values[:,j]).sum()),
                    missing_fraction=float(np.isnan(values[:,j]).mean()),negative_count=0,
                    minimum=float(np.nanmin(values[:,j])),maximum=float(np.nanmax(values[:,j])),
                    mean=float(np.nanmean(values[:,j])),median=float(np.nanmedian(values[:,j])))
                    for j,poll in enumerate(pollutants)}))
    if set(tables)!=set(stations): raise ValueError('Incomplete station coverage')
    after=zip_path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise ValueError('Archive changed during reading')
    hourly=np.stack([tables[s] for s in stations]);aggregated,counts=aggregate_four_hours(hourly)
    target=daily_targets(tables['Aotizhongxin'][:,0]);eligible=target['eligible']
    if eligible.mean()<.8: raise ValueError('Predeclared common-date retention below 80%')
    all_dates=np.array([(start+dt.timedelta(days=i)).date().isoformat() for i in range(len(eligible)+1)])
    dates=all_dates[1:][eligible];input_dates=all_dates[:-1][eligible]
    X=aggregated[:-1][eligible];hour_counts=counts[:-1][eligible]
    y=target['y'][eligible];persistence=target['persistence'][eligible]
    if not np.isfinite(y).all() or not np.isfinite(persistence).all(): raise ValueError('Unobserved target/persistence')
    if np.any(np.all(np.isnan(X),axis=0)): raise ValueError('Entire dataset contains unobserved feature')
    bins=['00:00-03:59','04:00-07:59','08:00-11:59','12:00-15:59','16:00-19:59','20:00-23:59']
    manifest=[]
    for i,date in enumerate(dates):
        source_index=int(np.flatnonzero(all_dates==date)[0])-1
        manifest.append(dict(target_date=date,input_date=input_dates[i],input_start=input_dates[i]+'T00:00:00',
            input_end=input_dates[i]+'T23:00:00',target_valid_hours=int(target['target_counts'][source_index]),
            previous_pm25_valid_hours=int(target['previous_counts'][source_index]),y=float(y[i]),
            persistence=float(persistence[i]),missing_input_cells=int(np.isnan(X[i]).sum())))
    excluded=[dict(target_date=all_dates[0],target_valid_hours=int(target['all_daily_counts'][0]),
                   previous_pm25_valid_hours=0,reason='no previous day in archive')]
    for i in np.flatnonzero(~eligible):
        reasons=[]
        if target['target_counts'][i]<18: reasons.append('target PM2.5 valid hours below18')
        if target['previous_counts'][i]<18: reasons.append('previous PM2.5 valid hours below18; observed persistence unavailable')
        excluded.append(dict(target_date=all_dates[i+1],target_valid_hours=int(target['target_counts'][i]),
            previous_pm25_valid_hours=int(target['previous_counts'][i]),reason='; '.join(reasons)))
    metadata=dict(status='PREPARED',zip_path=str(zip_path.resolve()),zip_bytes=after.st_size,
        zip_mtime_ns=after.st_mtime_ns,zip_sha256=archive_sha,zip_crc='PASS',csv_identity=identity,
        official_checksum_published=False,official_source='https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data',
        authenticity_limit='CRC and local SHA/shape validation are not server-byte comparison',
        stations=stations,pollutants=pollutants,bins=bins,units='ug/m^3 for all six pollutants including CO',
        tensor_shape=list(X.shape),axis_order=['sample','station','pollutant','previous_day_four_hour_bin'],flatten_order='C',
        target_station='Aotizhongxin',target_rule='observed target-day PM2.5 mean, >=18 valid hours',
        persistence_rule='observed previous-day PM2.5 mean, >=18 valid hours; never imputed',
        common_eligibility='target>=18 AND previous-day PM2.5>=18; every model uses identical dates',
        cell_rule='observed four-hour mean when count>=2, otherwise NaN; no interpolation',
        first_date=str(dates[0]),last_date=str(dates[-1]),potential_dates=len(eligible),retained_dates=int(eligible.sum()),
        retention_fraction=float(eligible.mean()),retention_gate=.8,
        target_only_eligible=int(np.sum(target['target_counts']>=18)),
        additional_persistence_exclusions=int(np.sum((target['target_counts']>=18)&~eligible)),
        missing_input_cells=int(np.isnan(X).sum()),days_with_missing_input=int(np.any(np.isnan(X),axis=(1,2,3)).sum()),
        station_audits=audit,raw_modified=False,learned_preprocessing=False,
        remaining_gate='Every outer and inner current training coordinate must have at least one finite value; check frozen splits before fitting')
    distribution=[]
    for level,keys in [('month',sorted(set(date[:7] for date in dates))),('year',sorted(set(date[:4] for date in dates)))]:
        for key in keys:
            keep=np.array([date.startswith(key) for date in dates]);values=y[keep]
            distribution.append(dict(level=level,period=key,n=len(values),mean=float(values.mean()),std=float(values.std()),
                minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max())))
    out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out_dir/'data.npz',X=X,y=y,persistence=persistence,dates=dates,input_dates=input_dates,
                        stations=np.asarray(stations),pollutants=np.asarray(pollutants),bins=np.asarray(bins),hour_counts=hour_counts)
    write_json(out_dir/'tensor_metadata.json',metadata)
    write_csv(out_dir/'sample_manifest.csv',manifest);write_csv(out_dir/'excluded_dates.csv',excluded)
    write_csv(out_dir/'response_period_statistics.csv',distribution)
    (out_dir/'BEIJING_DATA_AUDIT.md').write_text('# Beijing raw data audit\n\n'
        f'Status: PREPARED. Twelve stations, each35064 hourly rows, all identical complete datetime grids; ZIP CRC PASS. '
        f'{len(y)} common dates / {len(eligible)} candidate dates retained. No negative pollution values.\n\n'
        'Observed target and observed persistence both require >=18 hours, excluding13 additional target-valid dates. '
        'All models use identical eligibility, not selected by predictive errors. '
        'No official checksum published: local SHA/CRC/shape do not prove server-byte identity. '
        'Full raw identities and station QC: tensor_metadata.json. Original archive unchanged. '
        'Six pollutant units including CO are ug/m^3.\n',encoding='utf-8')
    (out_dir/'TENSOR_CONSTRUCTION_REPORT.md').write_text('# Fixed Beijing forecast tensor\n\n'
        f'Status: PREPARED; X shape {list(X.shape)}. Fixed axes: alphabetical station, PM2.5/PM10/SO2/NO2/CO/O3, '
        'six chronological four-hour bins of the previous calendar day. At least2 observed hours per cell; otherwise NaN. '
        'Targets use actual observed-hour means (at least18), not a divisor of24. '
        'Persistence is the observed previous-day mean (at least18), never learned or imputed.\n\n'
        f'Missing cells={np.isnan(X).sum()}, days requiring fold-only imputation={metadata["days_with_missing_input"]}. '
        'No interpolation or learned statistics are cached. Median imputation and scaling must fit current training rows only. '
        'Each current inner/outer training coordinate must be independently checked for all-NaN values before modeling. '
        'Prediction origin precedes target-day00:00; X contains only the previous day observations.\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zip-path',required=True);parser.add_argument('--out-dir',required=True)
    args=parser.parse_args();result=prepare(args.zip_path,args.out_dir)
    print(json.dumps(dict(status=result['status'],shape=result['tensor_shape'],retained_dates=result['retained_dates'])))
