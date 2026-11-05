"""Common fast fold-only preprocessing and matched compact fusion controls."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from src.dataset.preprocessing import ParticipantPreprocessor
from src.models.adaptive import AdaptiveFusion


class BranchPreprocessor(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        ParticipantPreprocessor._validate(x);self.columns_=list(x)
        a=x.to_numpy(dtype=float,copy=True);a[~np.isfinite(a)]=np.nan
        nonempty=np.flatnonzero(np.isfinite(a).any(0))
        medians=np.nanmedian(a[:,nonempty],axis=0);v=np.where(np.isnan(a[:,nonempty]),medians,a[:,nonempty])
        variable=np.flatnonzero(v.var(0)>0);v=v[:,variable]
        if not len(variable):raise ValueError('No nonconstant fitting features')
        _,unique=np.unique(v.T,axis=0,return_index=True);unique=np.sort(unique)
        self.keep_=nonempty[variable[unique]];self.medians_=medians[variable[unique]]
        self.scale_=StandardScaler().fit(v[:,unique]);return self

    def transform(self,x):
        if list(x)!=self.columns_:raise ValueError('Feature schema changed')
        a=x.to_numpy(dtype=float,copy=True)[:,self.keep_];a[~np.isfinite(a)]=np.nan
        return self.scale_.transform(np.where(np.isnan(a),self.medians_,a))


class BranchGate(AdaptiveFusion):
    def __init__(self,audio_dim,face_dim,mode):
        super().__init__(audio_dim,face_dim,16,.2,'equal' if mode=='equal' else 'adaptive')
        self.mode=mode
        if mode=='quality':self.quality_weight=nn.Parameter(torch.zeros(8,9))

    def forward(self,audio,face,quality,available):
        if not (available.sum(1)>0).all():raise ValueError('No primary modality available')
        a=self.audio_encoder(torch.where(available[:,:1].bool(),audio,torch.zeros_like(audio)))*available[:,:1]
        f=self.facial_encoder(torch.where(available[:,1:].bool(),face,torch.zeros_like(face)))*available[:,1:]
        if self.mode=='equal':weights=available/available.sum(1,keepdim=True)
        else:
            hidden=self.gate[0](torch.cat([a,f],1))
            if self.mode=='quality':hidden=hidden+F.linear(quality,self.quality_weight)
            logits=self.gate[2](self.gate[1](hidden));weights=torch.softmax(logits.masked_fill(~available.bool(),float('-inf')),1)
        return self.head(weights[:,:1]*a+weights[:,1:]*f).squeeze(-1),weights


def fit_network(audio,face,quality,available,y,mode,seed):
    torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=BranchGate(audio.shape[1],face.shape[1],mode)
    torch.manual_seed(seed+100000)
    tensors=[torch.as_tensor(a,dtype=torch.float32) for a in [audio,face,quality,available]]
    labels=torch.as_tensor(np.asarray(y),dtype=torch.float32)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((np.asarray(y)==0).sum()/(np.asarray(y)==1).sum())))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1);history=[];model.train()
    for epoch in range(40):
        optimizer.zero_grad(set_to_none=True);score,_=model(*tensors);loss=criterion(score,labels)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();optimizer.step();history.append(float(loss.detach()))
    return {k:v.detach().numpy().copy() for k,v in model.state_dict().items()},history


def neural_scores(bundle,audio,face,quality,available):
    model=BranchGate(audio.shape[1],face.shape[1],bundle['mode']);model.load_state_dict({k:torch.from_numpy(v) for k,v in bundle['state'].items()});model.eval()
    with torch.no_grad():return model(*[torch.as_tensor(v,dtype=torch.float32) for v in [audio,face,quality,available]])[0].numpy()
