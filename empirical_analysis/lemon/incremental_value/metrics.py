"""Paired OOF metrics and conditional subject-cluster bootstrap intervals."""
from __future__ import annotations
import numpy as np

MODELS = ('Covariates Only','Tensor-ABESS','Vector Lasso','Tensor-GLasso','Ridge')
BASELINE = MODELS[0]


def _performance(y, prediction):
    e = y-prediction
    mse = float(np.mean(e**2))
    tss = float(np.sum((y-y.mean())**2))
    return {'mse':mse,'rmse':float(np.sqrt(mse)),'mae':float(np.mean(np.abs(e))),
            'r2':None if tss==0 else float(1-np.sum(e**2)/tss)}


def _increment(base, model):
    b, m = float(np.mean(base)), float(np.mean(model))
    return {'delta_mse':b-m,'delta_rmse':float(np.sqrt(b)-np.sqrt(m)),
            'incremental_r2':None if b==0 else 1-m/b}


def _intervals(base, model, indices):
    # repeat x subject errors: resample each subject with all repeats together.
    b = np.mean(base.mean(axis=0)[indices],axis=1)
    m = np.mean(model.mean(axis=0)[indices],axis=1)
    draws = {'delta_mse':b-m,'delta_rmse':np.sqrt(b)-np.sqrt(m),
             'incremental_r2':np.divide(b-m,b,out=np.full_like(b,np.nan),where=b>0)}
    out = {}
    for name, values in draws.items():
        finite=values[np.isfinite(values)]
        low,high = np.quantile(finite,[.025,.975]) if len(finite) else (None,None)
        out[name+'_ci_low'] = None if low is None else float(low)
        out[name+'_ci_high'] = None if high is None else float(high)
    return out


def summarize_oof(rows, bootstrap_seed=20261009, bootstrap_samples=2000):
    rows = list(rows)
    if not rows or bootstrap_samples<1:
        raise ValueError('Nonempty OOF rows and positive bootstrap count required')
    repeats = sorted({int(r['repeat']) for r in rows})
    subjects = sorted({str(r['subject_id']) for r in rows})
    if {r['model'] for r in rows} != set(MODELS):
        raise ValueError('Exactly five configured models required')
    lookup = {}
    for row in rows:
        key=(int(row['repeat']),str(row['subject_id']),row['model'])
        if key in lookup or not np.isfinite([float(row['y_true']),float(row['y_pred'])]).all():
            raise ValueError('Duplicate or nonfinite OOF prediction')
        lookup[key]=row
    if len(lookup)!=len(repeats)*len(subjects)*len(MODELS):
        raise ValueError('Incomplete subject/repeat/model coverage')
    for repeat in repeats:
        for sid in subjects:
            base=lookup[(repeat,sid,BASELINE)]
            for name in MODELS[1:]:
                row=lookup[(repeat,sid,name)]
                if int(row['outer_fold'])!=int(base['outer_fold']) or float(row['y_true'])!=float(base['y_true']):
                    raise ValueError('Paired models disagree on fold or true response')
    y=np.array([[float(lookup[(r,s,BASELINE)]['y_true']) for s in subjects] for r in repeats])
    if not np.array_equal(y,np.broadcast_to(y[0],y.shape)):
        raise ValueError('True response changes across repeats')
    pred={m:np.array([[float(lookup[(r,s,m)]['y_pred']) for s in subjects] for r in repeats]) for m in MODELS}
    errors={m:(y-p)**2 for m,p in pred.items()}
    by_repeat,summary=[],[]
    for name in MODELS:
        records=[{'model':name,'repeat':r,'n_subjects':len(subjects),**_performance(y[i],pred[name][i])}
                 for i,r in enumerate(repeats)]
        by_repeat.extend(records)
        s={'model':name,'n_subjects':len(subjects),'n_repeats':len(repeats)}
        for metric in ('mse','rmse','mae','r2'):
            values=[r[metric] for r in records]
            s[metric+'_mean']=None if None in values else float(np.mean(values))
            s[metric+'_sd']=None if None in values or len(values)==1 else float(np.std(values,ddof=1))
        summary.append(s)
    rng=np.random.default_rng(bootstrap_seed)
    indices=rng.integers(0,len(subjects),size=(bootstrap_samples,len(subjects)))
    incremental,per_subject=[],[]
    for name in MODELS[1:]:
        b,m=errors[BASELINE],errors[name]
        d=b-m
        for i,r in enumerate(repeats):
            incremental.append({'model':name,'repeat':r,'n_subjects':len(subjects),'n_repeats':1,
                                'aggregation':'complete_oof_within_repeat',**_increment(b[i],m[i]),
                                **_intervals(b[i:i+1],m[i:i+1],indices)})
        average=d.mean(axis=0)
        absolute=np.abs(average)
        count=max(1,int(np.ceil(.1*len(subjects))))
        top=np.argsort(-absolute,kind='stable')[:count]
        incremental.append({'model':name,'repeat':'all','n_subjects':len(subjects),'n_repeats':len(repeats),
                            'aggregation':'pooled_equal_subject_repeat_mse',**_increment(b,m),**_intervals(b,m,indices),
                            'improved_subject_fraction':float(np.mean(average>0)),
                            'worsened_subject_fraction':float(np.mean(average<0)),
                            'tied_subject_fraction':float(np.mean(average==0)),
                            'mean_subject_improvement':float(average.mean()),
                            'median_subject_improvement':float(np.median(average)),
                            'top_absolute_10pct_n':count,
                            'top_absolute_10pct_share':float(absolute[top].sum()/absolute.sum()) if absolute.sum()>0 else 0.,
                            'top_absolute_10pct_signed_contribution':float(average[top].sum()/len(subjects)),
                            'bootstrap_seed':bootstrap_seed,'bootstrap_samples':bootstrap_samples,
                            'interval_interpretation':'descriptive_subject_cluster_percentile_conditional_on_saved_oof'})
        for j,sid in enumerate(subjects):
            record={'model':name,'subject_id':sid,'n_repeats':len(repeats),'y_true':float(y[0,j]),
                    'baseline_mean_squared_error':float(b[:,j].mean()),'model_mean_squared_error':float(m[:,j].mean()),
                    'mean_improvement':float(average[j]),'median_improvement':float(np.median(d[:,j])),
                    'improved_repeat_fraction':float(np.mean(d[:,j]>0))}
            record.update({f'improvement_repeat_{r}':float(d[i,j]) for i,r in enumerate(repeats)})
            per_subject.append(record)
    return {'metrics_by_repeat':by_repeat,'model_summary':summary,'incremental_metrics':incremental,
            'per_subject_improvement':per_subject}
