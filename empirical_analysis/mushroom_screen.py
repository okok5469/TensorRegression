"""CSV-only nested grouped screening: training mean, Ridge and PLS."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings
import numpy as np


def read_segment(path):
    active=False;wavelengths=None;records=[]
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for number,raw in enumerate(csv.reader(stream),1):
            row=[v.strip() for v in raw]
            while row and not row[-1]: row.pop()
            if not row: continue
            if len(row)>1 and 'dry matter' in row[1].lower():
                active=row[0].lower()=='mush no'
                if active:
                    current=np.asarray(row[2:],float)
                    if wavelengths is not None and not np.array_equal(current,wavelengths):
                        raise ValueError('Conflicting Mushroom section wavelength headers')
                    wavelengths=current
                continue
            if not active: continue
            if len(row)!=len(wavelengths)+2: raise ValueError(f'Malformed Mushroom row {number}')
            records.append((row[0],float(row[1]),np.asarray(row[2:],float),number))
    if not records: raise ValueError('Mush No section not found')
    ids=[r[0] for r in records]
    y=np.array([r[1] for r in records]);X=np.stack([r[2] for r in records])
    if len(set(ids))!=len(ids): raise ValueError('Duplicate sample ID')
    if not all(np.isfinite(a).all() for a in [y,X,wavelengths]): raise ValueError('Nonfinite input')
    if not np.all(np.diff(wavelengths)>0): raise ValueError('Nonunique or unordered wavelength axis')
    if np.any((y<0)|(y>1)): raise ValueError('Response contradicts audited mass-fraction encoding')
    return ids,y,wavelengths,X,[r[3] for r in records]


def spectrum_groups(X):
    return np.array([hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest() for row in X])


def group_folds(groups,count,seed):
    groups=np.asarray(groups).astype(str)
    unique,inverse,sizes=np.unique(groups,return_inverse=True,return_counts=True)
    if len(unique)<count: raise ValueError('Insufficient leakage groups')
    order=np.random.default_rng(seed).permutation(len(unique))
    order=order[np.argsort(-sizes[order],kind='stable')]
    loads=np.zeros(count,int);assigned=np.empty(len(unique),int)
    for index in order:
        fold=int(np.argmin(loads));assigned[index]=fold;loads[fold]+=sizes[index]
    return [(np.flatnonzero(assigned[inverse]!=k),np.flatnonzero(assigned[inverse]==k)) for k in range(count)]


def fit_model(name,X,y,parameter):
    from sklearn.linear_model import Ridge
    from sklearn.cross_decomposition import PLSRegression
    X,y=np.asarray(X,float),np.asarray(y,float)
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('Invalid training values')
    std=X.std(0,ddof=0);constant=std<=1e-12*max(1.,float(std.max()))
    y_scale=float(y.std(ddof=0))
    if y_scale<=1e-12*max(1.,float(np.max(abs(y)))): raise ValueError('Degenerate training response')
    artifact=dict(name=name,parameter=parameter,x_mean=X.mean(0),x_scale=np.where(constant,1.,std),
                  constant_mask=constant,y_mean=float(y.mean()),y_scale=y_scale)
    Z=np.where(constant,0.,(X-artifact['x_mean'])/artifact['x_scale'])
    ys=(y-artifact['y_mean'])/y_scale
    artifact['iterations']=None;artifact['converged']=True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name=='Mean': estimator=None
        elif name=='Ridge': estimator=Ridge(alpha=float(parameter),fit_intercept=False,solver='svd').fit(Z,ys)
        elif name=='PLS':
            estimator=PLSRegression(n_components=int(parameter),scale=False,max_iter=1000,tol=1e-7).fit(Z,ys)
            artifact.update(iterations=list(estimator.n_iter_),converged=all(v<1000 for v in estimator.n_iter_))
        else: raise ValueError('Unsupported screen model')
    artifact.update(estimator=estimator,warnings=[str(w.message) for w in caught])
    return artifact


def predict(artifact,X):
    Z=np.where(artifact['constant_mask'],0.,(np.asarray(X,float)-artifact['x_mean'])/artifact['x_scale'])
    if artifact['name']=='Mean': return np.full(len(Z),artifact['y_mean'])
    return np.asarray(artifact['estimator'].predict(Z)).reshape(-1)*artifact['y_scale']+artifact['y_mean']


def tune(name,X,y,splits,grid):
    records=[]
    for parameter in grid:
        sse=0.;n=0;folds=[]
        for train,valid in splits:
            model=fit_model(name,X[train],y[train],parameter)
            error=predict(model,X[valid])-y[valid]
            sse+=float(error@error);n+=len(valid)
            folds.append(dict(validation_n=len(valid),converged=model['converged'],warnings=model['warnings']))
        records.append(dict(parameter=float(parameter) if name=='Ridge' else int(parameter),mse=sse/n,folds=folds))
    selected=min(records,key=lambda r:r['mse'])['parameter']
    return selected,records


def metrics(y,pred):
    y,pred=np.asarray(y,float),np.asarray(pred,float)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
    total=float(np.sum((y-y.mean())**2))
    if total<=0: raise ValueError('Undefined R2 for constant response')
    sse=float(np.sum((y-pred)**2))
    return dict(rmse=float(np.sqrt(sse/len(y))),r2=float(1-sse/total))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(csv_path,output):
    from threadpoolctl import threadpool_limits
    thread_limiter=threadpool_limits(limits=1)
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists(): raise ValueError('Screen already frozen; refuse an unreviewed rerun')
    raw=Path(csv_path).read_bytes()
    if hashlib.md5(raw).hexdigest()!='15327c7fb3d5fe10231735a286629715': raise ValueError('Official CSV identity mismatch')
    ids,y_raw,waves,X,lines=read_segment(csv_path)
    y=100*y_raw;ids=np.asarray(ids);groups=spectrum_groups(X)
    duplicate=[]
    for group in np.unique(groups):
        members=np.flatnonzero(groups==group)
        if len(members)>1:
            duplicate.append(dict(sample_ids=ids[members].tolist(),y_pct=y[members].tolist(),
                                  response_span_pct=float(np.ptp(y[members]))))
    quality=dict(records=len(y),features=X.shape[1],wavelength_min_nm=float(waves.min()),
        wavelength_max_nm=float(waves.max()),missing_values=0,nonfinite_values=0,duplicate_ids=0,
        constant_bands=int(np.sum(X.std(0)==0)),reflectance_min=float(X.min()),reflectance_max=float(X.max()),
        exact_spectrum_groups=len(np.unique(groups)),duplicate_spectrum_groups=duplicate,
        duplicate_records=sum(len(g['sample_ids']) for g in duplicate),
        duplicate_redundant_records=len(y)-len(np.unique(groups)),
        y_min_pct=float(y.min()),y_max_pct=float(y.max()),y_mean_pct=float(y.mean()),y_sd_pct=float(y.std(ddof=1)),
        y_quantiles_pct={str(q):float(np.quantile(y,q)) for q in [0,.1,.25,.5,.75,.9,1]},
        excluded_records=[],unit='percentage points of dry matter; original fraction retained',
        unit_evidence='https://pmc.ncbi.nlm.nih.gov/articles/PMC10823050/ Table3: Mushroom5-87%,mean29%',
        entity_batch_information='Not present in CSV; exact-spectrum hashes are leakage groups, not physical entities')
    quality['id_order_blocks']=[dict(first_id=str(ids[i]),last_id=str(ids[min(i+49,len(y)-1)]),
        mean_pct=float(y[i:i+50].mean()),min_pct=float(y[i:i+50].min()),max_pct=float(y[i:i+50].max()))
        for i in range(0,len(y),50)]
    splits=[]
    for repeat,seed in enumerate([23001,23002,23003],1):
        for fold,(train,test) in enumerate(group_folds(groups,5,seed),1):
            inner=group_folds(groups[train],3,230000+100*repeat+fold)
            splits.append(dict(repeat=repeat,fold=fold,seed=seed,train=train.tolist(),test=test.tolist(),
                train_ids=ids[train].tolist(),test_ids=ids[test].tolist(),
                inner=[dict(train=a.tolist(),validation=b.tolist(),train_ids=ids[train[a]].tolist(),
                            validation_ids=ids[train[b]].tolist()) for a,b in inner]))
    config=dict(models=['Mean','Ridge','PLS'],outer_seeds=[23001,23002,23003],outer_folds=5,inner_folds=3,
        ridge_alphas=np.logspace(-4,4,17).tolist(),pls_components=[1,2,3,5,8,12,16],
        preprocessing='training-fold X/y zscore ddof0; constant-mask threshold1e-12; y inversion',
        grouping='exact-spectrum SHA256; all records retained and record-weighted',
        scalar_objective='Pooled inner validation MSE in percentage points',tie_rule='first minimum in frozen grid',
        source_csv=str(Path(csv_path).resolve()),csv_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((output/'FROZEN_PROTOCOL.md').read_bytes()).hexdigest())
    contract=dict(config=config,splits=splits)
    write_json(output/'contract.json',dict(fingerprint=fingerprint(contract),**contract))
    write_json(output/'config.json',config);write_json(output/'splits.json',splits);write_json(output/'data_quality.json',quality)
    write_csv(output/'sample_manifest.csv',[dict(sample_id=str(s),source_line=line,spectrum_group=str(g),
        y_raw=float(v),y_pct=float(p)) for s,line,g,v,p in zip(ids,lines,groups,y_raw,y)])
    all_rows=[];fold_rows=[];replays=[];fits=0;inner_fits=0;warnings_total=0
    for split in splits:
        train,test=np.array(split['train']),np.array(split['test'])
        inner=[(np.array(s['train']),np.array(s['validation'])) for s in split['inner']]
        limits=[]
        for a,b in inner:
            Z=X[train[a]];std=Z.std(0);mask=std<=1e-12*max(1.,float(std.max()))
            Z=np.where(mask,0.,(Z-Z.mean(0))/np.where(mask,1.,std))
            limits.append(min(len(a)-1,Z.shape[1],np.linalg.matrix_rank(Z)))
        for name in config['models']:
            parameter=None;cv=[]
            if name!='Mean':
                grid=config['ridge_alphas'] if name=='Ridge' else [c for c in config['pls_components'] if c<=min(limits)]
                parameter,cv=tune(name,X[train],y[train],inner,grid)
                inner_fits+=len(grid)*3
                warnings_total+=sum(len(f['warnings']) for r in cv for f in r['folds'])
            model=fit_model(name,X[train],y[train],parameter);fits+=1
            prediction=predict(model,X[test])
            directory=output/'models';directory.mkdir(exist_ok=True)
            artifact=directory/f'r{split["repeat"]}_f{split["fold"]}_{name}.pkl'
            with artifact.open('wb') as stream: pickle.dump(model,stream,protocol=5)
            with artifact.open('rb') as stream: saved=pickle.load(stream)
            rebuilt=predict(saved,X[test])
            if not np.allclose(rebuilt,prediction,rtol=1e-9,atol=1e-10): raise ValueError('Model artifact replay failed')
            warnings_total+=len(model['warnings'])
            write_json(directory/(artifact.stem+'.json'),dict(parameter=parameter,inner_cv=cv,
                converged=model['converged'],warnings=model['warnings'],iterations=model['iterations']))
            replays.append(dict(model=name,repeat=split['repeat'],fold=split['fold'],
                                max_abs_error=float(np.max(abs(rebuilt-prediction)))))
            all_rows.extend(dict(sample_id=str(ids[i]),spectrum_group=str(groups[i]),repeat=split['repeat'],
                fold=split['fold'],model=name,y_true_pct=float(y[i]),y_pred_pct=float(p),
                model_artifact=str(artifact)) for i,p in zip(test,prediction))
            fold_rows.append(dict(model=name,repeat=split['repeat'],fold=split['fold'],n_test=len(test),
                                  **metrics(y[test],prediction)))
        print('OUTER COMPLETE',split['repeat'],split['fold'],flush=True)
    keys=[(r['sample_id'],r['repeat'],r['model']) for r in all_rows]
    expected={(str(s),r,m) for s in ids for r in range(1,4) for m in config['models']}
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('OOF coverage mismatch')
    repeated=[]
    for name in config['models']:
        for repeat in range(1,4):
            selected=[r for r in all_rows if r['model']==name and r['repeat']==repeat]
            repeated.append(dict(model=name,repeat=repeat,**metrics([r['y_true_pct'] for r in selected],
                                                                   [r['y_pred_pct'] for r in selected])))
    summary=[dict(model=m,**{metric+'_'+label:float(fn([r[metric] for r in repeated if r['model']==m]))
             for metric in ['rmse','r2'] for label,fn in [('mean',np.mean),('sd',lambda v:np.std(v,ddof=1))]})
             for m in config['models']]
    for name,rows in [('oof_predictions.csv',all_rows),('fold_metrics.csv',fold_rows),
                      ('repeat_metrics.csv',repeated),('summary_metrics.csv',summary)]:write_csv(output/name,rows)
    if any(k.startswith('tensor_abess') for k in sys.modules): raise ValueError('Tensor algorithm unexpectedly imported')
    validation=dict(status='COMPLETE',records=len(ids),leakage_groups=len(np.unique(groups)),models=config['models'],
        outer_models=fits,inner_fits=inner_fits,oof_rows=len(all_rows),elapsed_seconds=time.perf_counter()-started,
        warnings_total=warnings_total,artifact_replays=replays,tensor_algorithm_imported=False,
        csv_sha256=config['csv_sha256'],source_sha256=config['source_sha256'],fingerprint=fingerprint(contract),
        caveat='Grouped by identical spectra only; physical entity/batch independence unavailable')
    write_json(output/'FINAL_VALIDATION.json',validation)
    report=['# Mushroom平均光谱：低成本预测信息检查','',f'状态：COMPLETE；仅CSV，Mean/Ridge/PLS，未导入或运行Tensor-ABESS。耗时{validation["elapsed_seconds"]:.1f}秒。','',
        '## 结果','', '|模型|完整OOF RMSE（百分点，均值±SD）|完整OOF R²（均值±SD）|','|---|---:|---:|']
    for row in summary:report.append(f'|{row["model"]}|{row["rmse_mean"]:.4f} ± {row["rmse_sd"]:.4f}|{row["r2_mean"]:.4f} ± {row["r2_sd"]:.4f}|')
    report+=['','3次重复×5外折，内3折独立调参；所有方法共享真实ID划分。每个重复先拼齐250条OOF再算RMSE/R²，之后报告重复均值和样本标准差；不是折R²平均。Ridge为17个固定alpha，PLS为7个预声明成分数（仅按内训练秩过滤），全部标准化和响应还原均在当前训练折拟合。','',
        '## 响应与数据质量','',f'250条记录，204个波段，400–998.04nm。干物质响应{y.min():.6f}%–{y.max():.6f}%，均值{y.mean():.6f}%，样本标准差{y.std(ddof=1):.6f}个百分点；原值{y_raw.min():.9f}–{y_raw.max():.9f}作为质量分数保留，按100倍转为百分点。该编码与[原作者论文Table3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10823050/)的Mushroom5%–87%、均值29%吻合；模型只使用本地CSV，不使用网页或MAT特征。',
        '无缺失/非有限值、重复ID或全体恒定波段，原始CSV官方MD5匹配；未删除任何记录或依据成绩修改响应/光谱。',
        '有6组完全相同光谱，共13条记录、7条冗余谱，且同谱响应不同。它们可能源于重复提取、记录来源或测量差异，CSV不足以判定；必须回源核验。此次保留全部响应，不平均、不修补标签；相同光谱以一个泄漏组进入外层及内层划分，防止跨折重复谱。243个谱组不等于243个已证明独立蘑菇。','']
    report.extend('- '+', '.join(g['sample_ids'])+'：'+', '.join(f'{v:.5f}%' for v in g['y_pct']) for g in duplicate)
    report+=['','## 解释边界','', '响应分布跨度很大，且沿CSV的ID顺序存在明显区段性；每连续50ID的响应摘要仅作质量描述，保存在data_quality.json，不是据此认定真实批次，也未用ID/响应分组来选择最有利划分。高总体R²可能主要反映宽范围或不同状态间差异，不能自动证明新批次泛化、鲜蘑菇窄范围精细预测或空间张量价值。',
        '本检查严格控制已知重复光谱及训练/调参泄漏，但CSV没有物理实体、时间或批次元数据，不能声称已控制未知实体或批次依赖；结论仅限这批CSV记录的平均谱预测信息筛查。Apple暂停，未启动新的张量实证或下载数据。','',
        f'全部{fits}个外层模型和{len(all_rows)}条OOF持久化重放通过rtol1e-9/atol1e-10；拟合警告总数{warnings_total}，每折参数与收敛记录保存在models/*.json。','',
        '复现：`python -m empirical_analysis.mushroom_screen --csv <本地CSV路径> --output doc/mushroom_screen`，在仓库根目录使用现有.venv；本目录已冻结、运行后拒绝无审查覆盖。协议、数据质量、划分、清单、逐样本OOF和完整OOF指标均已保存。']
    (output/'MUSHROOM_SCREEN_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--csv',required=True);parser.add_argument('--output',default='doc/mushroom_screen')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'doc/taskcard13/runtime'))
    for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    run(args.csv,args.output)


if __name__=='__main__':main()
