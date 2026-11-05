"""Regularized pooling with matched initialization and saved learning curves."""
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

EPOCHS=(5,10,20,40,100)


class PoolingV2(nn.Module):
    def __init__(self,attention=False):
        super().__init__();self.encoder=nn.Linear(64,16);self.head=nn.Linear(16,1)
        self.attention=nn.Linear(16,1) if attention else None
        self.dropout=nn.Dropout(.2)

    def forward(self,x,mask):
        if not mask.any(1).all():raise ValueError('Empty participant')
        h=self.dropout(torch.tanh(self.encoder(x)))
        weights=(mask.float()/mask.sum(1,keepdim=True) if self.attention is None else
                 torch.softmax(self.attention(h).squeeze(-1).masked_fill(~mask,-torch.inf),1))
        return self.head((h*weights[:,:,None]).sum(1)).squeeze(-1)


def train(x,mask,y,attention,seed,decay,epochs,valid=None):
    torch.manual_seed(seed);model=PoolingV2(attention).cuda();torch.manual_seed(seed+100000)
    z=torch.as_tensor(x,device='cuda');m=torch.as_tensor(mask,device='cuda');t=torch.as_tensor(np.asarray(y),dtype=torch.float32,device='cuda')
    loss=nn.BCEWithLogitsLoss(pos_weight=(len(t)-t.sum())/t.sum())
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=decay);history=[];scores={}
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad();value=loss(model(z,m),t)
        if not torch.isfinite(value):raise ValueError('Nonfinite training loss')
        value.backward();opt.step()
        if epoch in EPOCHS or epoch==epochs:
            model.eval()
            with torch.no_grad():
                fitted=model(z,m);record={'epoch':epoch,'fit_loss':float(loss(fitted,t)),
                    'fit_f1':float(f1_score(y,fitted.cpu().numpy()>=0))}
                if valid is not None:
                    v,vm,vy=valid;held=model(torch.as_tensor(v,device='cuda'),torch.as_tensor(vm,device='cuda'))
                    vt=torch.as_tensor(np.asarray(vy),dtype=torch.float32,device='cuda')
                    record.update(validation_loss=float(loss(held,vt)),validation_f1=float(f1_score(vy,held.cpu().numpy()>=0)))
                    scores[epoch]=held.cpu().numpy()
            history.append(record)
    return {k:v.detach().cpu() for k,v in model.state_dict().items()},scores,history


def predict(bundle,x,mask):
    from src.models.utterance_pooling import scale_transform
    model=PoolingV2(bundle['attention']).cuda();model.load_state_dict(bundle['state']);model.eval()
    z=scale_transform(x,mask,bundle['stats'])
    with torch.no_grad():return model(torch.as_tensor(z,device='cuda'),torch.as_tensor(mask,device='cuda')).cpu().numpy()
