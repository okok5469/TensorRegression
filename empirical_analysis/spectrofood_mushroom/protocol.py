"""Task15 ID-based recovery of screening CV and fixed file-block stress test."""
from __future__ import annotations

import numpy as np
from empirical_analysis.mushroom_screen import group_folds


def _disjoint(groups, train, test):
    if set(groups[train]) & set(groups[test]):
        raise ValueError('Leakage group crosses train/held-out partition')


def remap_main(data, screen_splits):
    ids=np.asarray(data['ids']).astype(str)
    lookup={sid:i for i,sid in enumerate(ids)}
    if len(lookup)!=len(ids):
        raise ValueError('Cache contains duplicate IDs')
    result=[]
    groups=np.asarray(data['groups']).astype(str)
    for old in screen_splits:
        train=np.array([lookup[sid] for sid in old['train_ids']])
        test=np.array([lookup[sid] for sid in old['test_ids']])
        if sorted(np.r_[train,test].tolist())!=list(range(len(ids))):
            raise ValueError('Screen split ID coverage differs from cache')
        _disjoint(groups,train,test)
        local={sid:i for i,sid in enumerate(old['train_ids'])}
        inner=[]
        for inside in old['inner']:
            a=np.array([local[sid] for sid in inside['train_ids']])
            b=np.array([local[sid] for sid in inside['validation_ids']])
            if sorted(np.r_[a,b].tolist())!=list(range(len(train))):
                raise ValueError('Screen inner split ID coverage differs')
            _disjoint(groups[train],a,b)
            inner.append(dict(train=a.tolist(),validation=b.tolist(),
                              train_ids=ids[train[a]].tolist(),validation_ids=ids[train[b]].tolist()))
        repeat,fold=int(old['repeat']),int(old['fold'])
        result.append(dict(protocol='main',repeat=repeat,fold=fold,
                           outer_seed=int(old['seed']),inner_seed=230000+100*repeat+fold,
                           train=train.tolist(),test=test.tolist(),train_ids=ids[train].tolist(),
                           test_ids=ids[test].tolist(),inner=inner))
    return result


def make_blocks(data):
    ids=np.asarray(data['ids']).astype(str)
    blocks=np.asarray(data['blocks'])
    groups=np.asarray(data['groups']).astype(str)
    if set(blocks.tolist())!=set(range(1,6)):
        raise ValueError('Expected five verified file blocks')
    result=[]
    for block in range(1,6):
        train=np.flatnonzero(blocks!=block);test=np.flatnonzero(blocks==block)
        _disjoint(groups,train,test)
        inner=[]
        # Four training files assigned intact to three folds (2/1/1 files).
        # The deterministic balancing uses only file sizes and a frozen seed.
        for a,b in group_folds(blocks[train].astype(str),3,250000+block):
            _disjoint(groups[train],a,b)
            _disjoint(blocks[train],a,b)
            inner.append(dict(train=a.tolist(),validation=b.tolist(),
                              train_ids=ids[train[a]].tolist(),validation_ids=ids[train[b]].tolist()))
        result.append(dict(protocol='block',repeat=1,fold=block,outer_seed=None,
                           inner_seed=250000+block,train=train.tolist(),test=test.tolist(),
                           train_ids=ids[train].tolist(),test_ids=ids[test].tolist(),inner=inner))
    return result
