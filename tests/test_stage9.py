import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score

from src.models.adaptive import AdaptiveFusion
from src.experiments.final import bootstrap_indices, bootstrap_f1, interval, verify_manifest, ablation_pairs
from src.experiments.baselines import TARGET_POLICY, sha256
from src.experiments.selection import fit_selected
from src.experiments.adaptive import predict_checkpoint


def test_equal_gate_preserves_initial_encoders_and_head():
    torch.manual_seed(42)
    adaptive=AdaptiveFusion(3,4)
    torch.manual_seed(42)
    equal=AdaptiveFusion(3,4,fusion_mode='equal')
    assert equal.gate is None
    for name,value in equal.state_dict().items():
        torch.testing.assert_close(value,adaptive.state_dict()[name],rtol=0,atol=0)
    equal.eval()
    a,f=torch.randn(6,3),torch.randn(6,4)
    logits,weights=equal(a,f)
    torch.testing.assert_close(weights,torch.full((6,2),.5))
    expected=equal.head((equal.audio_encoder(a)+equal.facial_encoder(f))*.5).squeeze(-1)
    torch.testing.assert_close(logits,expected)
    assert not any('gate' in name for name,_ in equal.named_parameters())


def test_bootstrap_preserves_classes_and_matches_sklearn():
    y=np.array([0,0,0,1,1]);p=np.array([.1,.6,.2,.8,.3])
    indices=bootstrap_indices(y,50)
    np.testing.assert_array_equal(indices,bootstrap_indices(y,50))
    assert np.all(y[indices].sum(1)==2)
    scores=bootstrap_f1(y,p,indices)
    expected=[f1_score(y[i],p[i]>=.5,zero_division=0) for i in indices]
    np.testing.assert_allclose(scores,expected)
    assert interval(scores-scores)==[0.,0.]
    with pytest.raises(ValueError):bootstrap_indices([0,0])


def test_manifest_rejects_mutated_artifact(tmp_path):
    p=tmp_path/'checkpoint';p.write_bytes(b'frozen')
    manifest={'target_policy':TARGET_POLICY,'threshold':.5,'frozen_hashes':{'checkpoint':sha256(p)}}
    verify_manifest(manifest,tmp_path)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen artifact changed'):
        verify_manifest(manifest,tmp_path)


def test_fixed_control_restores_and_does_not_learn_dev_labels(tmp_path):
    rng=np.random.default_rng(9)
    columns=['audio__a','audio__b','facial__a','facial__b']
    train=pd.DataFrame(rng.normal(size=(20,4)),columns=columns,index=range(1,21))
    dev=pd.DataFrame(rng.normal(size=(8,4)),columns=columns,index=range(101,109))
    y=pd.Series([0,1]*10,index=train.index);dy=pd.Series([0,1]*4,index=dev.index)
    settings={'hidden_dim':4,'dropout':.2,'learning_rate':.001,'weight_decay':.001,'epochs':3,'fusion_mode':'equal'}
    a,_,pa=fit_selected(train,y,dev,dy,columns,settings,42,'cpu')
    b,_,pb=fit_selected(train,y,dev,1-dy,columns,settings,42,'cpu')
    for k in a['state_dict']:torch.testing.assert_close(a['state_dict'][k],b['state_dict'][k],rtol=0,atol=0)
    np.testing.assert_array_equal(pa.probability_class_1,pb.probability_class_1)
    path=tmp_path/'equal.pt';torch.save(a,path)
    restored=predict_checkpoint(path,dev)
    np.testing.assert_array_equal(restored.probability_class_1,pa.probability_class_1)
    assert (restored[['audio_weight','facial_weight']]==.5).all().all()


def test_ablation_pairs_are_matched_single_factor():
    pairs=ablation_pairs()
    assert len(pairs)==12 and len(set(pairs))==12
    for first,second,effect in pairs:
        a,m=first.split('_');b,n=second.split('_')
        if effect in ['add_quality','add_gate']:assert a==b
        else:assert m==n and b=='all'
