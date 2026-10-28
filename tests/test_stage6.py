import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip('torch')
from src.models.reliability import ReliabilityFusion
from src.dataset.quality import QUALITY_COLUMNS,validate_quality
from src.experiments.adaptive import fit_adaptive,predict_checkpoint,train_network,seed_everything
from src.dataset.preprocessing import ParticipantPreprocessor


def inputs():
    rng=np.random.default_rng(17)
    frame=pd.DataFrame(rng.normal(size=(24,4)),columns=['audio__a','audio__b','facial__c','facial__d'],index=pd.Index(range(1,25),name='Participant_ID'))
    y=pd.Series(np.tile([0,1],12),index=frame.index)
    q=pd.DataFrame(np.tile([1,0,1,1,0,.7,.8,1],(24,1)),columns=QUALITY_COLUMNS,index=frame.index)
    q['facial__confidence_mean']=np.linspace(.2,.9,24)
    return frame,y,q


def params():
    return {'hidden_dim':4,'dropout':.2,'epochs':3,'weight_decay':.001,'learning_rate':.01}


def test_quality_gate_gradients_and_missing_modality():
    seed_everything(42)
    model=ReliabilityFusion(2,2,4,dropout=0)
    x,y,q=inputs(); qt=torch.tensor(q.iloc[:6].to_numpy(),dtype=torch.float32)
    a=torch.randn(6,2);f=torch.randn(6,2)
    logits,weights=model(a,f,qt)
    torch.testing.assert_close(weights.sum(1),torch.ones(6))
    logits.square().mean().backward()
    assert model.quality_weight.grad.abs().sum()>0
    qt[:,0]=0
    out,w=model(a,f,qt)
    changed,_=model(a*1000,f,qt)
    torch.testing.assert_close(out,changed,rtol=0,atol=0)
    torch.testing.assert_close(w[:,0],torch.zeros(6))
    torch.testing.assert_close(w[:,1],torch.ones(6))
    qt[:,3]=0
    with pytest.raises(ValueError,match='available'):model(a,f,qt)


@pytest.mark.parametrize('problem',['label','reordered','nan','range','both_missing'])
def test_quality_validation(problem):
    frame,y,q=inputs()
    if problem=='label':q=q.assign(PHQ_Score=1)
    elif problem=='reordered':q=q.iloc[::-1]
    elif problem=='nan':q.iloc[0,6]=np.nan
    elif problem=='range':q.iloc[0,6]=1.1
    else:q.iloc[0,[0,3]]=0
    with pytest.raises(ValueError):validate_quality(q,frame.index)


def test_constant_quality_reduces_to_stage5_training():
    frame,y,q=inputs()
    q[:]=[1,0,1,1,0,.7,.8,1]
    normalized=ParticipantPreprocessor().fit_transform(frame)
    base,_,loss,_=train_network(normalized,y,params(),42,'cpu')
    quality,_,qloss,_=train_network(normalized,y,params(),42,'cpu',q)
    np.testing.assert_allclose(loss,qloss,atol=1e-7,rtol=1e-7)
    for key,value in base.state_dict().items():
        torch.testing.assert_close(value,quality.state_dict()[key],atol=1e-7,rtol=1e-7)


def test_quality_training_only_checkpoint_and_dev_independence(tmp_path,monkeypatch):
    frame,y,q=inputs();train=frame.iloc[:18];dev=frame.iloc[18:]
    seen=[];original=train_network
    def tracked(X,target,parameters,seed,device,quality=None):
        assert set(X.index).issubset(train.index)
        assert quality.index.equals(X.index)
        model,config,history,weight=original(X,target,parameters,seed,device,quality)
        np.testing.assert_allclose(config['quality_mean'],quality.mean())
        seen.append(set(X.index))
        return model,config,history,weight
    monkeypatch.setattr('src.experiments.adaptive.train_network',tracked)
    saved,result,predictions=fit_adaptive(train,y.iloc[:18],dev,y.iloc[18:],42,3,[params()],'cpu',q.iloc[:18],q.iloc[18:])
    assert len(seen)==4 and seen[-1]==set(train.index)
    path=tmp_path/'quality.pt';torch.save(saved,path)
    restored=predict_checkpoint(path,dev,'cpu',q.iloc[18:])
    for c in restored:np.testing.assert_array_equal(restored[c],predictions[c])
    with pytest.raises(ValueError,match='requires'):predict_checkpoint(path,dev)
    altered=q.iloc[18:].copy();altered['facial__confidence_mean']=.01
    again,result2,_=fit_adaptive(train,y.iloc[:18],dev,1-y.iloc[18:],42,3,[params()],'cpu',q.iloc[:18],altered)
    assert result['cv_candidates']==result2['cv_candidates']
    for k in saved['state_dict']:torch.testing.assert_close(saved['state_dict'][k],again['state_dict'][k],rtol=0,atol=0)
    assert result['experiment']=='reliability_fusion'
    assert saved['quality_columns']==QUALITY_COLUMNS
