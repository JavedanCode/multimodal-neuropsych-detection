import numpy as np
import pytest
import torch
from src.models.utterance_pooling import UtterancePool,scale_fit,scale_transform


@pytest.mark.parametrize('attention',[False,True])
def test_masked_padding_and_permutation_do_not_change_predictions(attention):
    torch.manual_seed(42);model=UtterancePool(attention).eval()
    x=torch.randn(2,5,64);mask=torch.tensor([[True,True,False,False,False],[True,True,True,False,False]])
    with torch.no_grad():
        expected=model(x,mask);changed=x.clone();changed[~mask]=500
        torch.testing.assert_close(expected,model(changed,mask))
        order=[4,2,0,3,1]
        torch.testing.assert_close(expected,model(x[:,order],mask[:,order]))


def test_scaler_ignores_padding_and_has_no_heldout_fit():
    x=np.ones((2,3,64),np.float32);mask=np.array([[True,False,False],[True,True,False]])
    x[0,0]=2;x[~mask]=999
    stats=scale_fit(x,mask)
    np.testing.assert_allclose(stats[0],4/3)
    transformed=scale_transform(x,mask,stats)
    assert (transformed[~mask]==0).all()
    np.testing.assert_allclose(transformed[mask].mean(0),0,atol=1e-6)
