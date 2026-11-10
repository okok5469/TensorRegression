"""Deterministic software/installation fixture, not performance evidence."""
import hashlib
import json
from pathlib import Path


def run_smoke(output):
    from .identity import verify_release
    from .data import load_dataset
    from .runner import run
    from .returns import validate_return, pack_return
    from .model import TensorABESSOvR
    import numpy as np

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    identity = verify_release()
    fixture = Path(__file__).resolve().parent/'fixtures/tiny'
    config = fixture/'protocol.json'
    dataset = load_dataset(fixture, config)
    run_dir = root/'run'
    run_result = run(fixture, config, run_dir)
    if run_result['status'] != 'COMPLETE':
        raise RuntimeError(f'Smoke comparison incomplete: {run_result}')
    validation = validate_return(run_dir)
    packed = pack_return(run_dir, root/'smoke_return.zip')
    # Independently evaluate persisted coefficients, without invoking model.predict.
    model_files = sorted(p for p in (run_dir/'models').rglob('model.json') if p.parent.name == 'tensor_abess_ovr_ls')
    if not model_files:
        raise RuntimeError('Smoke did not persist an application model')
    model = TensorABESSOvR.load(model_files[0].parent)
    X = np.asarray(dataset['X'])
    transformed = (X-model.preprocessor_.mean_)/model.preprocessor_.scale_
    transformed[:,model.preprocessor_.constant_mask_] = 0.0
    independent = transformed.reshape(len(X),-1,order='C') @ model.coef_.reshape(len(model.classes_),-1,order='C').T + model.class_prior_
    actual = model.decision_function(X,axis_metadata=dataset['metadata']['axes'])
    np.testing.assert_allclose(actual,independent,rtol=1e-8,atol=1e-10)
    predictions=model.predict(X,axis_metadata=dataset['metadata']['axes'])
    np.savez(root/'fixture_signature.npz',scores=actual,classes=model.classes_,coefficients=model.coef_,predictions=predictions,supports=np.asarray(json.dumps(model.supports_,sort_keys=True)),tie_priority=model.tie_priority_)
    result = {'status':'PASS','identity':identity,'run':run_result,'return_validation':validation,'return_archive':str(root/'smoke_return.zip'),'saved_coefficient_score_max_error':float(np.max(np.abs(actual-independent))),'fixture_is_performance_evidence':False}
    (root/'SMOKE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    return result
