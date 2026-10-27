import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch', reason='Install the neural extra for Stage 5 tests')
from src.models.adaptive import AdaptiveFusion
from src.experiments.adaptive import fit_adaptive, predict_checkpoint, seed_everything
from src.dataset.preprocessing import ParticipantPreprocessor


def dataset():
    rng=np.random.default_rng(91)
    frame=pd.DataFrame(rng.normal(size=(24,6)),index=pd.Index(range(1,25),name='Participant_ID'),columns=['audio__a','audio__b','audio__c','facial__d','facial__e','facial__f'])
    labels=pd.Series(np.tile([0,1],12),index=frame.index)
    return frame.iloc[:18],labels.iloc[:18],frame.iloc[18:],labels.iloc[18:]


def candidate():
    return {'hidden_dim':4,'dropout':.2,'epochs':3,'weight_decay':.001,'learning_rate':.01}


def test_shapes_gate_normalization_and_gradients():
    seed_everything(42)
    model=AdaptiveFusion(3,4,hidden_dim=8)
    logits,weights=model(torch.randn(7,3),torch.randn(7,4))
    assert logits.shape==(7,) and weights.shape==(7,2)
    torch.testing.assert_close(weights.sum(1),torch.ones(7))
    assert torch.all(weights>=0)
    logits.square().mean().backward()
    for block in [model.audio_encoder,model.facial_encoder,model.gate,model.head]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters())
    with pytest.raises(ValueError):model(torch.randn(2,3),torch.randn(3,4))
    with pytest.raises(ValueError):model(torch.randn(2,2),torch.randn(2,4))
    with pytest.raises(TypeError):model(torch.randn(2,3),torch.randn(2,4),torch.randn(2,1))


def test_fold_fitting_checkpoint_and_reproducibility(tmp_path,monkeypatch):
    train,y,dev,dy=dataset()
    observed=[];original=ParticipantPreprocessor.fit
    def track(self,X,y=None):
        observed.append(set(X.index))
        return original(self,X,y)
    monkeypatch.setattr(ParticipantPreprocessor,'fit',track)
    saved,result,predictions=fit_adaptive(train,y,dev,dy,42,3,[candidate()],'cpu')
    assert len(observed)==4
    assert all(ids.issubset(train.index) and not ids.intersection(dev.index) for ids in observed)
    assert sum(ids==set(train.index) for ids in observed)==1
    path=tmp_path/'model.pt';torch.save(saved,path)
    restored=predict_checkpoint(path,dev)
    for column in restored:
        np.testing.assert_array_equal(restored[column],predictions[column])
    saved2,result2,predictions2=fit_adaptive(train,y,dev,1-dy,42,3,[candidate()],'cpu')
    for key in saved['state_dict']:
        torch.testing.assert_close(saved['state_dict'][key],saved2['state_dict'][key],rtol=0,atol=0)
    assert result['cv_candidates']==result2['cv_candidates']
    assert result['training_loss_by_epoch']==result2['training_loss_by_epoch']
    assert result['test_evaluated'] is False
    assert saved['train_participant_ids']==list(train.index)
    assert result['train_positive_weight']==1.


def test_adaptive_rejects_overlap_and_forbidden_columns():
    train,y,dev,dy=dataset()
    with pytest.raises(ValueError,match='Overlapping'):
        fit_adaptive(train,y,train.iloc[:4],y.iloc[:4],42,3,[candidate()],'cpu')
    with pytest.raises(ValueError,match='Forbidden'):
        fit_adaptive(train.assign(audio__PHQ_Score=0),y,dev.assign(audio__PHQ_Score=0),dy,42,3,[candidate()],'cpu')


def test_cuda_training_smoke_when_available():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; CPU tests still cover training')
    from src.experiments.adaptive import train_network,predict
    train,y,dev,dy=dataset()
    prep=ParticipantPreprocessor().fit(train)
    model,_,history,_=train_network(prep.transform(train),y,candidate(),42,'cuda')
    probability,weights=predict(model,prep.transform(dev),'cuda')
    assert np.isfinite(history).all() and np.isfinite(probability).all()
    np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
