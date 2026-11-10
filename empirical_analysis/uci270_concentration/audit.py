"""Independent artifact audit; no production preprocessing or metric helpers.

Artifact/metric verification never fits a model and raises ValueError on
contradictory evidence. audit_native_pilot is an explicit separately budgeted
optimizer rerun; it is never invoked by artifact or metric verification.
"""
from math import lgamma
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np


def _require(condition, label):
    if not condition:
        raise ValueError('Artifact audit failed: '+label)


def _equal(a,b,label,rtol=1e-9,atol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    _require(a.shape==b.shape and np.allclose(a,b,rtol=rtol,atol=atol),label)


def _preprocessing(X,y,indices,p):
    x=X[indices]; target=y[indices]; std=x.std(0)
    expected=dict(x_mean=x.mean(0),x_scale=np.where(std==0,1,std),
                  constant_mask=std==0,y_mean=target.mean(),y_scale=target.std())
    for key,value in expected.items():
        _equal(p[key],value,'training-only preprocessing '+key)
    _require(expected['y_scale']>0,'training response variance')
    return (x-expected['x_mean'])/expected['x_scale'],(target-expected['y_mean'])/expected['y_scale']


def _split(a,b,data):
    a,b=np.asarray(a,int),np.asarray(b,int)
    _require(len(a)>0 and len(b)>0,'nonempty split')
    _require(len(np.unique(a))==len(a) and len(np.unique(b))==len(b),'unique split rows')
    _require(not np.intersect1d(a,b).size,'split overlap')
    _require(np.max(data['batch'][a])<np.min(data['batch'][b]),'chronology')
    _require(len(np.unique(data['gas'][np.r_[a,b]]))==1,'gas membership')
    _require(not np.intersect1d(data['group96'][a],data['group96'][b]).size,'duplicate leakage')


def _alpha(X,y,p,diag,param):
    flat=((X-p['x_mean'])/p['x_scale']).reshape(len(X),-1)
    target=(y-p['y_mean'])/p['y_scale']
    alpha=float(np.max(np.abs(flat.T@target))/len(y))
    _equal(diag['alpha_max'],alpha,'Lasso alpha_max')
    _equal(diag['alpha_actual'],alpha*param['lambda_ratio'],'Lasso relative penalty')


def audit_artifact(artifact,data,saved_predictions=None):
    """Audit an already-unpickled artifact against dataset.npz array mapping.

    saved_predictions, when supplied, is a one-dimensional array in test_indices
    order. This function reads no files, changes no input and runs no new fits.
    """
    raw=np.asarray(data['raw128']); tensor=np.asarray(data['X']); y=np.asarray(data['y'])
    _require(raw.shape==(len(y),128) and tensor.shape==(len(y),16,3,2),'input dimensions')
    for sensor in range(16):
        for scale in range(3):
            for stage in range(2):
                _require(np.array_equal(tensor[:,sensor,scale,stage],raw[:,sensor*8+2+stage*3+scale]),'raw tensor mapping')
    method=artifact['model_id']; X=raw if method in ('M6','M7') else tensor
    train=np.asarray(artifact['train_indices'],int); test=np.asarray(artifact['test_indices'],int)
    _split(train,test,data)
    batch=np.asarray(data['batch']); gas=np.asarray(data['gas']); groups=np.asarray(data['group96'])
    g=int(gas[test[0]]); K=int(batch[test[0]])-1
    _require(np.all(batch[test]==K+1),'single test batch')
    full_test=np.flatnonzero((gas==g)&(batch==K+1))
    original=np.flatnonzero((gas==g)&(batch<=K))
    expected_train=original[~np.isin(groups[original],groups[full_test])]
    _equal(test,full_test,'all test records',rtol=0,atol=0)
    _equal(train,expected_train,'outer train-only duplicate purge',rtol=0,atol=0)
    pairs=artifact['inner_pairs']; _require(2<=len(pairs)<=3,'inner fold count')
    for a,b in pairs:
        _split(a,b,data)
        _require(np.isin(np.r_[a,b],train).all(),'inner membership')
    eligible_pairs=[]
    for boundary in range(1,K):
        validation=train[batch[train]==boundary+1]
        candidate=train[batch[train]<=boundary]
        candidate=candidate[~np.isin(groups[candidate],groups[validation])]
        valid=len(candidate)>=30 and len(validation)>=5 and len(np.unique(y[candidate]))>=3 and np.var(y[candidate])>0
        if valid:
            eligible_pairs.append((candidate,validation))
    _require(len(pairs)==min(3,len(eligible_pairs)),'recent eligible inner fold count')
    for (a,b),(expected_a,expected_b) in zip(pairs,eligible_pairs[-3:]):
        _equal(a,expected_a,'recent inner train/purge',0,0)
        _equal(b,expected_b,'recent inner validation',0,0)
    prep=artifact['preprocessing']; Z,target=_preprocessing(X,y,train,prep)
    coef=np.asarray(artifact['coef_standardized']); intercept=artifact['intercept_standardized']
    flat=Z.reshape(len(train),-1)
    training_rank=int(np.linalg.matrix_rank(flat))
    _require(training_rank==artifact['diagnostics']['training_rank'],'training rank')
    _require(coef.shape==(flat.shape[1],),'coefficient dimensions')
    Ztest=((X[test]-prep['x_mean'])/prep['x_scale']).reshape(len(test),-1)
    standardized=Ztest@coef+intercept
    pred=standardized*prep['y_scale']+prep['y_mean']
    _require(np.isfinite(pred).all(),'finite predictions')
    result=dict(passed=True,n_train=len(train),n_test=len(test),mapping_coordinates_checked=int(tensor.size),
                training_rank=training_rank,
                prediction_max_abs=None,native_optimizer_rerun='not_run',inner_sse_recomputed=0,
                inner_sse_unavailable=0,estimator_replay_max_abs=None)
    if saved_predictions is not None:
        _equal(pred,saved_predictions,'saved prediction replay')
        result['prediction_max_abs']=float(np.max(np.abs(pred-np.asarray(saved_predictions))))
    if 'estimator' in artifact:
        native=np.asarray(artifact['estimator'].predict(Ztest)).reshape(-1)
        _equal(native,standardized,'estimator coefficient/intercept replay')
        result['estimator_replay_max_abs']=float(np.max(np.abs(native-standardized)))
    if 'B' in artifact:
        _equal(np.asarray(artifact['B']).ravel(order='C'),coef,'tensor coefficient C-order')
    records=artifact['cv_records']
    for record in records:
        _require(len(record['folds'])==len(pairs),'candidate fold count')
        for fold,(a,b) in zip(record['folds'],pairs):
            _equal(fold['train_indices'],a,'inner train IDs',0,0)
            _equal(fold['validation_indices'],b,'inner validation IDs',0,0)
            _preprocessing(X,y,a,fold['preprocessing'])
            _require(fold['n_validation']==len(b),'inner validation denominator')
            if 'predictions' in fold:
                residual=np.asarray(fold['predictions'])-y[b]
                _equal(fold['sse'],residual@residual,'inner SSE')
                result['inner_sse_recomputed']+=1
                if 'coef_standardized' in fold:
                    p=fold['preprocessing']; z=((X[b]-p['x_mean'])/p['x_scale']).reshape(len(b),-1)
                    replay=(z@np.asarray(fold['coef_standardized'])+fold['intercept_standardized'])*p['y_scale']+p['y_mean']
                    _equal(replay,fold['predictions'],'inner coefficient prediction replay')
            else:
                result['inner_sse_unavailable']+=1
            if method in ('M3','M7'):
                _alpha(X[a],y[a],fold['preprocessing'],fold['diagnostics'],record['parameter'])
        mse=sum(f['sse'] for f in record['folds'])/sum(f['n_validation'] for f in record['folds'])
        _equal(record['mse'],mse,'candidate weighted mse')
    if records:
        best=min(r['mse'] for r in records)
        eligible=[r for r in records if r['mse']<=best+1e-12*max(1,abs(best))]
        chosen=sorted(eligible,key=lambda r:next(iter(r['parameter'].values())),reverse=method!='M5')[0]['parameter']
        _require(chosen==artifact['parameters'],'parameter tie selection')
    if method in ('M3','M7'):
        _alpha(X[train],y[train],prep,artifact['diagnostics'],artifact['parameters'])
    if method=='M1':
        diag=artifact['diagnostics']; supports=[np.asarray(s,int) for s in diag['supports']]
        mask=np.zeros((16,3,2),bool);mask[np.ix_(*supports)]=True
        q=int(np.prod([len(s) for s in supports]));_require(q==diag['q'],'ABESS rectangular q')
        _require(np.all(coef[~mask.ravel()]==0),'ABESS out-of-support zeros')
        design=flat[:,mask.ravel()]
        ols,_,rank,_=np.linalg.lstsq(design,target,rcond=None)
        _equal(design@ols,flat@coef,'ABESS OLS fitted values',rtol=1e-8,atol=1e-9)
        residual=target-flat@coef; rss=float(residual@residual)
        _equal(diag['rss'],rss,'ABESS RSS')
        n=len(train); ell=np.log(max(np.log(n),np.e))
        combinatorial=sum(lgamma(p+1)-lgamma(len(s)+1)-lgamma(p-len(s)+1) for p,s in zip((16,3,2),supports))
        parts=dict(log_rss=n*np.log(max(rss/n,np.finfo(float).tiny)),df=1.5*q*ell,combinatorial=ell*combinatorial)
        for key,value in parts.items():_equal(diag['msic_components'][key],value,'ABESS MSIC '+key)
        _equal(diag['msic'],sum(parts.values()),'ABESS total MSIC')
        result['abess']=dict(q=q,rank=int(rank),rss=rss,msic_components=parts,msic=sum(parts.values()),
                             ols_prediction_max_abs=float(np.max(np.abs(design@ols-flat@coef))),
                             solver_convergence='not_exposed_by_formal_solver')
        native=artifact.get('native_result')
        if native is not None:
            for actual,expected,label in [(native.B,artifact['B'],'B'),(native.rss,rss,'RSS'),(native.sic,sum(parts.values()),'MSIC')]:
                _equal(actual,expected,'native '+label)
            for actual,expected in zip((native.S1,native.S2,native.S3),supports):_equal(actual,expected,'native support',0,0)
            caps=tuple(int(v) for v in native.s_max)
            count=sum(i*j*k<=max(1,int(np.floor(.7*n))) for i in range(1,caps[0]+1) for j in range(1,caps[1]+1) for k in range(1,caps[2]+1))
            _require(count==diag['candidate_grid_count'],'native full cap grid count')
            bcd=native.bcd_result
            _equal(bcd.B,native.B,'native BCD selected coefficients')
            _equal(bcd.rss,native.rss,'native BCD selected RSS')
            result['abess']['candidate_grid_count']=count
            result['abess']['tolerance_selection']='not_exposed: native adaptive result does not save all candidate scores, minimum-score q, or eligible set; histories cannot reconstruct these'
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _select_candidates(records,n,scale):
    best=min(range(len(records)),key=lambda i:records[i][1])
    q_best=int(np.prod(records[best][0])); minimum=float(records[best][1])
    tolerance=float(max(scale,0)*np.sqrt(max(q_best,1)*max(np.log(max(n,2)),1)))
    eligible=[i for i,(sizes,score) in enumerate(records) if score<=minimum+tolerance]
    selected=min(eligible,key=lambda i:(int(np.prod(records[i][0])),records[i][1],sum(records[i][0])))
    return dict(selected_index=selected,min_score=minimum,q_best=q_best,tolerance=tolerance,eligible_indices=eligible)


def audit_native_pilot(root):
    """One explicitly budgeted native pilot replay with read-only return tracing."""
    from tensor_abess.tensor.abess import tensor_abess
    from tensor_abess.tensor.bcd import _refine_msic_global_candidates
    root=Path(root);contract=_json(root/'contract.json');digest=_sha(root/'contract.json')
    split=next(s for s in contract['splits'] if s['status']=='ready' and s['split']=='dev1')
    name=f"gas{split['gas_id']}_dev1_M1";folder=root/'models'/name
    _sealed_task(folder,name,digest)
    output=root/'audit_native_pilot.json';_require(not output.exists(),'native audit already exists')
    with (folder/'model.pkl').open('rb') as stream:artifact=pickle.load(stream)
    with np.load(contract['data_path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    Z,target=_preprocessing(data['X'],data['y'],np.asarray(split['train']),artifact['preprocessing'])
    captures=[]
    def tracer(frame,event,arg):
        if frame.f_code is _refine_msic_global_candidates.__code__ and event=='return':
            captures.append(dict(frame.f_locals))
        return tracer
    prior=sys.gettrace();start=time.perf_counter()
    try:
        sys.settrace(tracer)
        replay=tensor_abess(Z,target,**artifact['parameters'])
    finally:
        elapsed=time.perf_counter()-start;sys.settrace(prior)
        # Always retain elapsed fitting time, even if native fitting raises.
        (root/'audit_native_budget.json').write_text(json.dumps(dict(task_id='audit_native_refit',seconds=elapsed,contract_sha256=digest),indent=2),encoding='utf-8')
    _require(len(captures)==1,'native refine return capture count')
    capture=captures[0]; candidates=capture['candidates']
    records=[([int(c.s1),int(c.s2),int(c.s3)],float(score)) for c,score in candidates]
    independent=_select_candidates(records,len(target),capture['one_se_scale'])
    for key in ('min_score','q_best','tolerance'):_equal(independent[key],capture[key],'native tolerance '+key)
    chosen=candidates[independent['selected_index']][0]
    _equal(chosen.B,replay.B,'native independent selected B')
    _equal(replay.B,artifact['B'],'native refit original B')
    _equal(replay.rss,artifact['native_result'].rss,'native refit original RSS')
    _equal(replay.sic,artifact['native_result'].sic,'native refit original MSIC')
    for support,expected in zip((replay.S1,replay.S2,replay.S3),artifact['diagnostics']['supports']):_equal(support,expected,'native refit support',0,0)
    result=dict(status='PASS',contract_sha256=digest,task_id=name,extra_fit_count=1,fit_seconds=elapsed,
                candidate_count=len(candidates),candidate_sizes_scores=records,independent_selection=independent,
                coefficient_max_abs=float(np.max(np.abs(replay.B-artifact['B']))),
                instrumentation='sys.settrace return-event locals read only; unchanged native code/parameters/return',
                complete_sha256=_sha(folder/'COMPLETE.json'))
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def _sealed_task(folder,name,digest):
    seal=_json(folder/'COMPLETE.json')
    _require(seal.get('status')=='COMPLETE' and seal.get('task_id')==name,'completion identity')
    _require(seal.get('contract_sha256')==digest,'completion contract')
    expected={'model.pkl','predictions.csv','diagnostics.json','cv_records.json'}
    _require(set(seal['files'])==expected,'completion required files')
    for filename,value in seal['files'].items():
        _require(_sha(folder/filename)==value,'completion file hash '+filename)
    return seal


def verify_all(root,phase='pilot'):
    """Verify a completed phase; final requires accepted development first.

    Returns INCOMPLETE without issuing an acceptance lock while tasks are absent.
    Contradictions raise immediately. PASS locks are bound to contract and report.
    """
    _require(phase in ('pilot','dev','final'),'unknown audit phase')
    root=Path(root);contract=_json(root/'contract.json');digest=_sha(root/'contract.json')
    if phase=='final':
        _require((root/'DEVELOPMENT_ACCEPTED.json').exists(),'development acceptance required')
        _validated_lock(root,'DEVELOPMENT_ACCEPTED.json',digest)
    config=_json(root/'config.json')
    for path,expected in contract['source'].items():_require(_sha(path)==expected,'frozen source '+path)
    for key in ('data_path','config_path'):_require(_sha(contract[key])==contract[key+'_sha256'],'frozen '+key)
    ready=[s for s in contract['splits'] if s['status']=='ready' and s['split']!='final']
    selected=[next(s for s in ready if s['split']=='dev1')] if phase=='pilot' else ready
    if phase=='final':
        selected=[s for s in contract['splits'] if s['status']=='ready' and s['split']=='final']
        _require(len(selected)==6 and len(config['methods'])==8,'all six final tasks and eight methods required')
    tasks=[(f"gas{s['gas_id']}_{s['split']}_{m}",s,m) for s in selected for m in config['methods']]
    missing=[name for name,_,_ in tasks if not (root/'models'/name/'COMPLETE.json').exists()]
    if missing:
        return dict(status='INCOMPLETE',missing=missing,contract_sha256=digest)
    native_path=root/'audit_native_pilot.json'
    _require(native_path.exists(),'native pilot audit required before acceptance')
    native=_json(native_path)
    _require(native['status']=='PASS' and native['contract_sha256']==digest,'native pilot identity')
    with np.load(contract['data_path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    audits={}; seals={}
    for name,split,method in tasks:
        folder=root/'models'/name;_sealed_task(folder,name,digest)
        with (folder/'model.pkl').open('rb') as stream:a=pickle.load(stream)
        _require(a['contract_sha256']==digest and a['task_id']==name and a['model_id']==method,'artifact task identity')
        _require(a['split']==split,'artifact frozen split')
        with (folder/'predictions.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        _require(all(row['subset'] in ('train','test') for row in rows),'prediction subsets')
        test_rows=[row for row in rows if row['subset']=='test']
        train_rows=[row for row in rows if row['subset']=='train']
        _equal([int(row['row_index']) for row in test_rows],split['test'],'prediction test IDs',0,0)
        _equal([int(row['row_index']) for row in train_rows],split['train'],'prediction train IDs',0,0)
        low,high=np.min(data['y'][split['train']]),np.max(data['y'][split['train']])
        for row in rows:
            i=int(row['row_index']);target=float(data['y'][i])
            _equal(float(row['y']),target,'prediction response')
            _require(row['record_id']==str(data['record_id'][i]) and int(row['gas_id'])==int(data['gas'][i]) and int(row['batch'])==int(data['batch'][i]),'prediction metadata')
            _require(row['split']==split['split'] and row['method']==method,'prediction task metadata')
            expected_range='below' if target<low else 'above' if target>high else 'inside'
            _require(row['range']==expected_range,'response range stratum')
        audits[name]=audit_artifact(a,data,[float(row['prediction']) for row in test_rows])
        _require(audits[name]['inner_sse_unavailable']==0,'inner prediction evidence absent')
        seals[name]=_sha(folder/'COMPLETE.json')
    report=dict(status='PASS',phase=phase,contract_sha256=digest,tasks=audits,complete_seals=seals,
                audit_source_sha256=_sha(__file__),native_pilot_sha256=_sha(native_path),
                limits=['full candidate selection independently traced on pilot only; other saved adaptive results do not expose all candidates'])
    report_path=root/('audit_'+phase+'.json')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lock_path=root/{'pilot':'PILOT_ACCEPTED.json','dev':'DEVELOPMENT_ACCEPTED.json','final':'FINAL_ACCEPTED.json'}[phase]
    lock=dict(status='PASS',contract_sha256=digest,report=report_path.name,report_sha256=_sha(report_path),
              audit_source_sha256=_sha(__file__),native_pilot_sha256=_sha(native_path),task_count=len(tasks))
    lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    return report


def _validated_lock(root,filename,digest):
    lock=_json(root/filename)
    _require(lock.get('status')=='PASS' and lock.get('contract_sha256')==digest,'acceptance lock identity')
    _require(_sha(root/lock['report'])==lock['report_sha256'],'acceptance report hash')
    report=_json(root/lock['report'])
    _require(report.get('status')=='PASS' and report.get('contract_sha256')==digest,'acceptance report identity')
    for task,expected in report['complete_seals'].items():
        folder=root/'models'/task
        _require(_sha(folder/'COMPLETE.json')==expected,'accepted completion seal')
        _sealed_task(folder,task,digest)
    return report


def _expected_metrics(y,pred):
    y=list(map(float,y));pred=list(map(float,pred));n=len(y)
    _require(len(pred)==n,'metric prediction count')
    if n==0:return dict(n=0,rmse=None,mae=None,r2=None,bias=None)
    residual=[p-t for p,t in zip(pred,y)];sse=sum(v*v for v in residual)
    mean=sum(y)/n;sst=sum((v-mean)**2 for v in y)
    return dict(n=n,rmse=(sse/n)**.5,mae=sum(abs(v) for v in residual)/n,
                bias=sum(residual)/n,r2=1-sse/sst if n>1 and sst>0 else None)


def _csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def _check_fields(row,expected):
    for field,value in expected.items():
        _require(field in row,'missing aggregate field '+field)
        if value is None:
            _require(row[field] in ('','null','None'),'expected NA '+field)
            continue
        if isinstance(value,list):
            _require(json.loads(row[field])==value,'aggregate list '+field)
            continue
        _equal(float(row[field]),value,'aggregate '+field)


def _design_complexity(artifact,data):
    indices=np.asarray(artifact['train_indices'],int);p=artifact['preprocessing'];method=artifact['model_id']
    X=data['raw128'] if method in ('M6','M7') else data['X']
    design=((X[indices]-p['x_mean'])/p['x_scale']).reshape(len(indices),-1)
    beta=np.asarray(artifact['coef_standardized']);strict=np.flatnonzero(beta!=0).tolist()
    active=np.flatnonzero(abs(beta)>1e-8*max(1.,max(abs(beta))))
    support=artifact['diagnostics'].get('supports')
    capacity=int(np.prod([len(s) for s in support])) if support is not None else len(strict)
    native=strict if method in ('M3','M7') else None
    return dict(training_rank=int(np.linalg.matrix_rank(design)),
                effective_design_rank=int(np.linalg.matrix_rank(design[:,active])) if len(active) else 0,
                native_active_coordinates=native,native_active_count=len(native) if native is not None else None,
                support_capacity=capacity)


def verify_metrics(root):
    """Independently recompute written summaries; no calls into analysis.py."""
    root=Path(root);digest=_sha(root/'contract.json')
    contract=_json(root/'contract.json')
    _require(_sha(contract['data_path'])==contract['data_path_sha256'],'metric data hash')
    with np.load(contract['data_path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    summary=_csv(root/'summary_metrics.csv');ranges=_csv(root/'range_metrics.csv');complexity=_csv(root/'model_complexity.csv')
    # Only metadata are inspected until the appropriate phase lock passes.
    has_final=any(row['split']=='final' for row in summary)
    dev_report=_validated_lock(root,'DEVELOPMENT_ACCEPTED.json',digest)
    accepted=dict(dev_report['complete_seals'])
    if has_final:accepted.update(_validated_lock(root,'FINAL_ACCEPTED.json',digest)['complete_seals'])
    key=lambda r:f"gas{r['gas_id']}_{r['split']}_{r['method']}"
    summaries={key(r):r for r in summary}; comps={key(r):r for r in complexity}
    _require(len(summaries)==len(summary) and len(comps)==len(complexity),'aggregate unique task rows')
    _require(set(summaries)==set(accepted) and set(comps)==set(accepted),'aggregate complete task set')
    range_index={(key(r),r['range']):r for r in ranges}
    _require(len(range_index)==len(ranges)==3*len(accepted),'range unique full set')
    combined=_csv(root/'predictions.csv'); expected_combined=[]; counts=0
    for name in sorted(accepted):
        folder=root/'models'/name; rows=_csv(folder/'predictions.csv')
        train=[r for r in rows if r['subset']=='train'];test=[r for r in rows if r['subset']=='test']
        expected_combined.extend(test)
        ty=[float(r['y']) for r in train];y=[float(r['y']) for r in test]
        test_metrics=_expected_metrics(y,[r['prediction'] for r in test]);train_metrics=_expected_metrics(ty,[r['prediction'] for r in train])
        train_mean=sum(ty)/len(ty);sd=(sum((v-train_mean)**2 for v in ty)/len(ty))**.5
        fields=dict(test_metrics,n_train=len(train),n_test=len(test),train_rmse=train_metrics['rmse'],train_r2=train_metrics['r2'],
                    train_y_sd=sd,normalized_rmse=test_metrics['rmse']/sd,train_y_min=min(ty),train_y_max=max(ty),
                    test_y_min=min(y),test_y_max=max(y),outside_n=sum(v<min(ty) or v>max(ty) for v in y))
        _check_fields(summaries[name],fields)
        for group in ('inside','below','above'):
            selected=[r for r in test if r['range']==group]
            _check_fields(range_index[(name,group)],_expected_metrics([r['y'] for r in selected],[r['prediction'] for r in selected]))
        with (folder/'model.pkl').open('rb') as stream:a=pickle.load(stream)
        beta=np.asarray(a['coef_standardized']);full=a['model_id'] in ('M6','M7')
        threshold=1e-8*max(1,max(abs(beta)));selected=np.flatnonzero(abs(beta)>threshold)
        coordinates=[np.unravel_index(int(i),(16,8) if full else (16,3,2)) for i in selected]
        sensor=sorted({int(c[0])+1 for c in coordinates});scale=None if full else sorted({int(c[1]) for c in coordinates});stage=None if full else sorted({int(c[2]) for c in coordinates})
        fields=dict(strict_nnz=int(sum(v!=0 for v in beta)),effective_nnz=len(selected),threshold=threshold,
                    sensors=sensor,n_sensors=len(sensor),scales=scale,stages=stage,n_scales=None if full else len(scale),
                    n_stages=None if full else len(stage),envelope_capacity=None if full else len(sensor)*len(scale)*len(stage),
                    steady_nnz=sum(c[1]<2 for c in coordinates) if full else 0,
                    dynamic_nnz=sum(c[1]>=2 for c in coordinates) if full else len(selected))
        _check_fields(comps[name],fields)
        diag=_json(folder/'diagnostics.json')
        native=a['diagnostics'].get('supports')
        extra=dict(q=int(np.prod([len(s) for s in native])) if native is not None else None,
                   native_group_counts=[len(s) for s in native] if native is not None else None,
                   pls_components=a['parameters'].get('n_components'),
                   training_rank=a['diagnostics']['training_rank'],selected_rank=a['diagnostics'].get('ols_rank'),
                   total_seconds=diag['total_seconds'],prediction_seconds=diag['prediction_seconds'])
        _check_fields(comps[name],extra)
        _check_fields(comps[name],_design_complexity(a,data))
        _check_fields(summaries[name],dict(total_seconds=diag['total_seconds'],fit_count=diag['total_fit_count']))
        counts+=len(test)
    sorting=lambda r:(int(r['gas_id']),r['split'],r['method'],int(r['row_index']))
    _require(sorted(combined,key=sorting)==sorted(expected_combined,key=sorting),'combined predictions exact correspondence')
    filenames=['summary_metrics.csv','range_metrics.csv','model_complexity.csv','predictions.csv']
    result=dict(status='PASS',contract_sha256=digest,tasks=len(accepted),test_predictions=counts,
                files={name:_sha(root/name) for name in filenames},audit_source_sha256=_sha(__file__))
    (root/'audit_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
