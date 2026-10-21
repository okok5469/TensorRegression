"""Independent saved-coefficient reconstruction, without any support search."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_digest(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256(str(array.shape).encode('ascii'))
    h.update(array.dtype.str.encode('ascii'))
    h.update(array.tobytes())
    return h.hexdigest()


def near(actual, expected, message, rtol=1e-9, atol=1e-10):
    if not np.isclose(actual,expected,rtol=rtol,atol=atol):
        raise ValueError(f'{message}: {actual} vs {expected}')


def check_task(repo, folder):
    from scripts.taskcard7.lemon import array_sha
    from numerical_simulation.data_generation import (TensorSimulationConfig,
        generate_simulation_dataset, iter_test_batches)
    from tensor_abess.tensor.sic import compute_tensor_msic_from_rss
    record=read(folder/'complete.json')
    request=record['request']
    values=dict(request['config'])
    for key in ('dims','sparsity','rho'):
        values[key]=tuple(values[key])
    config=TensorSimulationConfig(**values)
    data=generate_simulation_dataset(config,seed=request['seed'])
    for key in ('X_train','y_train','B_true'):
        if array_sha(getattr(data,key))!=record['input_fingerprints'][key]:
            raise ValueError('Input regeneration fingerprint drift: '+key)
    if [array_sha(s) for s in data.supports_true]!=record['input_fingerprints']['supports_true']:
        raise ValueError('Truth support fingerprint drift')
    for name,digest in record['artifact_sha256'].items():
        if file_sha(folder/name)!=digest:
            raise ValueError('Saved dataset artifact drift: '+name)
    trace=read(folder/'trace.json')
    choices=read(folder/'selection.json')['choices']
    with np.load(folder/'candidate_coefficients.npz',allow_pickle=False) as cache:
        coefficients=cache['B']
    if len(coefficients)!=len(trace['candidates']):
        raise ValueError('Missing coefficient candidates')
    maximum_rss=maximum_gradient=maximum_score=0.
    for index,(candidate,B) in enumerate(zip(trace['candidates'],coefficients)):
        supports=[np.asarray(s,dtype=int) for s in candidate['support']]
        sizes=tuple(map(len,supports)); q=int(np.prod(sizes))
        if q!=candidate['q'] or q>int(np.floor(.7*config.n_train)):
            raise ValueError('Illegal q')
        for s,p in zip(supports,config.dims):
            if list(s)!=sorted(set(s)) or np.any(s<0) or np.any(s>=p):
                raise ValueError('Illegal support')
        if array_sha(B)!=candidate['B_sha256'] or not np.isfinite(B).all():
            raise ValueError('Candidate B hash/nonfinite drift')
        beta=B[np.ix_(*supports)].reshape(-1)
        if np.count_nonzero(B)!=np.count_nonzero(beta):
            raise ValueError('Coefficients outside recorded support')
        Z=data.X_train[np.ix_(np.arange(config.n_train),*supports)].reshape(config.n_train,q)
        residual=data.y_train-Z@beta
        rss=float(residual@residual)
        near(rss,candidate['rss'],'Independent sparse-design RSS')
        maximum_rss=max(maximum_rss,abs(rss-candidate['rss'])/max(1.,abs(rss)))
        gradient=float(np.linalg.norm(Z.T@residual)/((1+np.linalg.norm(Z))*(1+np.linalg.norm(data.y_train))))
        maximum_gradient=max(maximum_gradient,gradient)
        if gradient>1e-10:
            raise ValueError('Candidate does not satisfy normalized OLS equations')
        for alpha,choice in zip((1.5,2.),choices):
            score=float(compute_tensor_msic_from_rss(rss,*sizes,config.n_train,*config.dims,alpha=alpha,gamma=1.))
            near(score,choice['candidate_scores'][index],'Independent MSIC')
            maximum_score=max(maximum_score,abs(score-choice['candidate_scores'][index]))
    sse=np.zeros(2); signal_sse=np.zeros(2); h=hashlib.sha256(); n=0; batches=0
    for X,y,signal in iter_test_batches(data,seed=request['test_seed']):
        for array in (X,y,signal):
            h.update(stream_digest(array).encode('ascii'))
        for j,choice in enumerate(choices):
            candidate=trace['candidates'][choice['selected_index']]
            B=coefficients[choice['selected_index']]
            supports=[np.asarray(s,dtype=int) for s in candidate['support']]
            Z=X[np.ix_(np.arange(len(y)),*supports)].reshape(len(y),candidate['q'])
            prediction=Z@B[np.ix_(*supports)].reshape(-1)
            residual=y-prediction; gap=signal-prediction
            sse[j]+=float(residual@residual); signal_sse[j]+=float(gap@gap)
        n+=len(y); batches+=1
    expected=record['test_fingerprint']
    if h.hexdigest()!=expected['sha256'] or n!=expected['n_observations'] or batches!=expected['n_batches']:
        raise ValueError('Independent shared test stream drift')
    metric_errors=[]; selected_conditions=[]
    for j,(choice,row) in enumerate(zip(choices,record['metrics'])):
        candidate=trace['candidates'][choice['selected_index']]
        B=coefficients[choice['selected_index']]
        supports=[np.asarray(s,dtype=int) for s in candidate['support']]
        Z=data.X_train[np.ix_(np.arange(config.n_train),*supports)].reshape(config.n_train,candidate['q'])
        singular=np.linalg.svd(Z,compute_uv=False)
        numerical_rank=int(np.sum(singular > singular[0]*max(Z.shape)*np.finfo(float).eps))
        if numerical_rank!=candidate['q']:
            raise ValueError('Selected design is numerically rank deficient')
        selected_conditions.append(float(singular[0]/singular[-1]))
        estimated=list(map(set,candidate['support'])); truth=list(map(set,data.supports_true))
        exact=float(estimated==truth)
        mode_f1=sum(2*len(a&b)/(len(a)+len(b)) for a,b in zip(estimated,truth))/3
        overlap=int(np.prod([len(a&b) for a,b in zip(estimated,truth)]))
        element_f1=2*overlap/(candidate['q']+int(np.prod(config.sparsity)))
        ree=float(np.linalg.norm(B-data.B_true)/np.linalg.norm(data.B_true))
        for key,value in [('pmse',sse[j]/n),('signal_mse',signal_sse[j]/n),('ree',ree),
                          ('mode_f1_macro',mode_f1),('element_f1',element_f1),('exact_recovery',exact)]:
            near(value,row[key],'Independent '+key)
            metric_errors.append(abs(float(value)-float(row[key])))
    return {'cohort':request['cohort'],'task_id':request['task_id'],'n_train':config.n_train,
            'sparsity':config.sparsity,'candidates_checked':len(coefficients),
            'maximum_rss_relative_error':maximum_rss,'maximum_normalized_OLS_gradient':maximum_gradient,
            'maximum_msic_absolute_error':maximum_score,'maximum_selected_metric_error':max(metric_errors),
            'selected_designs_full_rank':True,'selected_design_condition_numbers':selected_conditions,
            'shared_test_fingerprint_exact':True,'no_model_refitting':True}


def main():
    from scripts.taskcard7.lemon import configure_core
    repo=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=repo/'doc/taskcard8/INDEPENDENT_NUMERICAL_AUDIT.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh audit output; existing audit is preserved')
    base=repo/'numerical_simulation/outputs_selection_research_task8'
    if not ((base/'completion.json').is_file() or (base/'stopped.json').is_file()):
        raise ValueError('Wait for the frozen controller terminal record before numerical audit')
    source=configure_core('v5.2-search-final')
    # Fixed by n and truth shape; never chosen using test errors or recovery.
    declared=[0,6,36,42,72,78]
    results=[]; absent=[]
    for cohort in ('development','confirmation'):
        for task in declared:
            folder=base/cohort/f'task_{task:03d}'
            if (folder/'complete.json').is_file():
                results.append(check_task(repo,folder))
                print('NUMERICAL_AUDIT',cohort,task,flush=True)
            else:
                absent.append({'cohort':cohort,'task_id':task})
    imports={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
             if name=='tensor_abess' or name.startswith('tensor_abess.')}
    if not all(Path(path).is_relative_to(source) for path in imports.values()):
        raise ValueError('Auditor import escaped frozen v5.2 source')
    body={'scope':'First replicate at SNR0.5/rho0 in each n100/200/400 and both truth shapes; two cohorts when available; no support search or outcome-based case choice',
          'passed_on_available_cases':True,'all_predeclared_cases_available':not absent,
          'actual_core_imports':imports,
          'cases':results,'unavailable_cases':absent,
          'coverage_limit':'Not a claim of independent numerical reconstruction of every simulated dataset; all dataset hashes are checked separately.'}
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(body,handle,indent=2,ensure_ascii=False)
    print('INDEPENDENT_AUDIT_COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
