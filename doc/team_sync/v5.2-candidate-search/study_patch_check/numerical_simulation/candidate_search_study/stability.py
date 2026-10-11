"""Training-only, predeclared multistart support search; Oracle is evaluation only."""
from __future__ import annotations
from time import perf_counter
import numpy as np


def make_initializations(X, y, sizes, *, training_seed):
    from tensor_abess.utils.init import initial_support
    sizes=tuple(int(s) for s in sizes)
    if len(sizes)!=3 or any(s<1 or s>d for s,d in zip(sizes,X.shape[1:])):
        raise ValueError('Fixed sizes must be within tensor dimensions')
    strategies=[('marginal','marginal',0),('random_0','random',int(training_seed)+30_000_000),
                ('random_1','random',int(training_seed)+31_000_000)]
    return [{'strategy':name,'random_seed':seed,'supports':initial_support(X,y,*sizes,strategy=strategy,random_state=seed)}
            for name,strategy,seed in strategies]


def fit_multistart(X,y,sizes,*,training_seed,max_cycles=6,k_max=1,m_max=15):
    from tensor_abess.tensor.bcd import tensor_bcd
    sizes=tuple(int(s) for s in sizes)
    if int(np.prod(sizes))>max(1,int(np.floor(.7*len(y)))):
        raise ValueError('Predeclared size exceeds the candidate-level parameter capacity')
    records=[]
    for start in make_initializations(X,y,sizes,training_seed=training_seed):
        began=perf_counter()
        result=tensor_bcd(X,y,*sizes,max_cycles=max_cycles,k_max=k_max,m_max=m_max,
                         **{f'S{i}_init':S for i,S in enumerate(start['supports'],1)})
        records.append({'strategy':start['strategy'],'random_seed':start['random_seed'],
            'initial_supports':tuple(S.copy() for S in start['supports']),
            'supports':tuple(getattr(result,f'S{i}').copy() for i in (1,2,3)),
            'sizes':sizes,'B':result.B.copy(),'rss':float(2*len(y)*result.loss),
            'loss':float(result.loss),'loss_history':list(result.loss_history),
            'visited_loss_history':list(result.visited_loss_history),'best_iteration':result.best_iteration,
            'stop_reason':result.stop_reason,'n_cycles':result.n_cycles,'runtime_sec':perf_counter()-began})
    winner=min(range(len(records)),key=lambda i:(records[i]['rss'],i))
    return dict(records[winner],selected_start=winner),records
