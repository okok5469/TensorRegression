"""Fixed batch order, X-only duplicate exclusion and training-only eligibility."""
import numpy as np


def make_splits(data):
    batch,gas,y,groups=(np.asarray(data[k]) for k in ('batch','gas','y','group96'))
    result=[]
    for g in range(1,7):
        for name,K in [('dev1',6),('dev2',7),('dev3',8),('final',9)]:
            original=np.flatnonzero((gas==g)&(batch<=K))
            test=np.flatnonzero((gas==g)&(batch==K+1))
            overlap=np.isin(groups[original],groups[test])
            train=original[~overlap]
            inner=[];candidates=[]
            for k in range(1,K):
                a=train[batch[train]<=k];b=train[batch[train]==k+1]
                excluded=a[np.isin(groups[a],groups[b])]
                a=a[~np.isin(groups[a],groups[b])]
                valid=len(a)>=30 and len(b)>=5 and len(np.unique(y[a]))>=3 and np.var(y[a])>0
                row=dict(boundary=k,train=a.tolist(),validation=b.tolist(),purged_train=excluded.tolist())
                candidates.append(dict(boundary=k,n_train=len(a),n_validation=len(b),eligible=bool(valid)))
                if valid:inner.append(row)
            inner=inner[-3:]
            reasons=[]
            if len(test)==0:reasons.append('empty_test_batch')
            if len(train)<30 or len(np.unique(y[train]))<3 or (len(train)>0 and np.var(y[train])==0):
                reasons.append('insufficient_training_variation_or_count')
            if len(inner)<2:reasons.append('fewer_than_two_eligible_inner_boundaries')
            result.append(dict(gas_id=g,split=name,K=K,train=train.tolist(),test=test.tolist(),
                               purged_train=original[overlap].tolist(),inner=inner,candidates=candidates,
                               status='not_evaluable' if reasons else 'ready',reasons=reasons))
    return result
