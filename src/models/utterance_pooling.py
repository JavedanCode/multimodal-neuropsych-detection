"""Small masked bag-of-utterances classifiers."""
import numpy as np
import torch
from torch import nn


class UtterancePool(nn.Module):
    def __init__(self, attention=False):
        super().__init__()
        self.encoder=nn.Linear(64,16)
        self.attention=nn.Linear(16,1) if attention else None
        self.head=nn.Linear(16,1)

    def forward(self,x,mask):
        if not mask.any(1).all():raise ValueError('Empty participant')
        h=torch.tanh(self.encoder(x))
        if self.attention is None:
            weights=mask.float()/mask.sum(1,keepdim=True)
        else:
            logits=self.attention(h).squeeze(-1).masked_fill(~mask,-torch.inf)
            weights=torch.softmax(logits,dim=1)
        return self.head((h*weights[:,:,None]).sum(1)).squeeze(-1)


def scale_fit(x,mask):
    values=x[mask].astype(np.float64)
    mean=values.mean(0);sd=values.std(0)
    return mean.astype(np.float32),np.where(sd>1e-6,sd,1).astype(np.float32)


def scale_transform(x,mask,stats):
    mean,sd=stats
    return np.where(mask[:,:,None],(x-mean)/sd,0).astype(np.float32)


def train(x,mask,y,attention,seed,epochs,valid=None):
    torch.manual_seed(seed)
    model=UtterancePool(attention).to('cuda')
    z=torch.as_tensor(x,device='cuda');m=torch.as_tensor(mask,device='cuda')
    t=torch.as_tensor(np.asarray(y),dtype=torch.float32,device='cuda')
    weight=(len(t)-t.sum())/t.sum()
    loss=nn.BCEWithLogitsLoss(pos_weight=weight)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    scores={}
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad();error=loss(model(z,m),t)
        if not torch.isfinite(error):raise ValueError('Nonfinite training loss')
        error.backward();opt.step()
        if valid is not None and epoch in (40,100):
            model.eval()
            with torch.no_grad():scores[epoch]=model(torch.as_tensor(valid[0],device='cuda'),torch.as_tensor(valid[1],device='cuda')).cpu().numpy()
    return {k:v.detach().cpu() for k,v in model.state_dict().items()},scores


def predict(bundle,x,mask):
    model=UtterancePool(bundle['attention']).to('cuda')
    model.load_state_dict(bundle['state']);model.eval()
    z=scale_transform(x,mask,bundle['stats'])
    with torch.no_grad():return model(torch.as_tensor(z,device='cuda'),torch.as_tensor(mask,device='cuda')).cpu().numpy()
