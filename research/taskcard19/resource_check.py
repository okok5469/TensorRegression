"""One fixed training-only functionality/timing check, then seal the budget."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'

import argparse
import hashlib
import json
from pathlib import Path
import time


def check(data_path,output):
    import numpy as np
    from tensor_abess_ovr.data import load_dataset
    from tensor_abess_ovr.model import TensorABESSOvR
    from tensor_abess_ovr.identity import code_identity
    data=load_dataset(data_path,Path(data_path)/'protocol.json')
    split=data['selected_splits'][0]
    lookup={str(key):i for i,key in enumerate(data['sample_ids'])}
    available=np.array([lookup[key] for key in split['train_ids']])
    classes=data['metadata']['classes']
    generator=np.random.default_rng(20261031)
    selected=np.concatenate([generator.choice(available[data['y'][available]==key],size=6,replace=False) for key in classes])
    selected.sort()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    model=TensorABESSOvR(classes=classes,axis_metadata=data['metadata']['axes'],core_timeout=900.)
    try:
        model.fit(data['X'][selected],data['y'][selected],sample_ids=data['sample_ids'][selected],deadline=time.monotonic()+7200.)
    except Exception as error:
        failure={'status':'FAILED','error_type':type(error).__name__,'error':str(error),'sample_ids':data['sample_ids'][selected].tolist(),'elapsed_seconds':time.perf_counter()-started,'fit_diagnostics':getattr(model,'fit_diagnostics_',[]),'formal_test_evaluated':False}
        (output/'FUNCTIONAL_CHECK_FAILURE.json').write_text(json.dumps(failure,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        raise
    model.save(output/'training_slice_model')
    result={'status':'PASS','sample_ids':data['sample_ids'][selected].tolist(),'sample_count':len(selected),'class_count':len(classes),'elapsed_seconds':time.perf_counter()-started,'class_diagnostics':model.fit_diagnostics_,'code_identity':code_identity(),'scope':'fixed training-only functionality/timing slice; no evaluation scores','resource_budget':{'per_class_seconds':900,'total_seconds':7200,'max_datasets':2,'processes':1,'blas_threads':1},'formal_test_evaluated':False}
    (output/'RESOURCE_BUDGET.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'sample_count':len(selected),'elapsed_seconds':result['elapsed_seconds']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    check(args.data,args.output)
