import math, torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
class PositionalEncoding(nn.Module):
 def __init__(self,d,max_len=512):
  super().__init__();p=torch.arange(max_len).unsqueeze(1);f=torch.exp(torch.arange(0,d,2)*(-math.log(10000.)/d));x=torch.zeros(max_len,d);x[:,0::2]=torch.sin(p*f);x[:,1::2]=torch.cos(p*f);self.register_buffer('pe',x.unsqueeze(0))
 def forward(self,x): return x+self.pe[:,:x.size(1)]
class MarketTransformer(nn.Module):
 def __init__(self,n,d=64,h=4,l=2):
  super().__init__();self.p=nn.Linear(n,d);self.e=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,h,192,.15,batch_first=True,norm_first=True),l);self.pos=PositionalEncoding(d);self.out=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,1))
 def forward(self,x): return self.out(self.e(self.pos(self.p(x)))[:,-1]).squeeze(-1)
def fit_classical(x,y):
 a=LogisticRegression(max_iter=2000,class_weight='balanced');b=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.);a.fit(x,y);b.fit(x,y);return a,b
def fit_transformer(x,y,xv,yv,epochs=20):
 m=MarketTransformer(x.shape[-1]);o=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4);loss=nn.BCEWithLogitsLoss();tx,ty=torch.tensor(x),torch.tensor(y)
 for _ in range(epochs):
  o.zero_grad();v=loss(m(tx),ty);v.backward();o.step()
 return m
def transformer_proba(m,x):
 m.eval()
 with torch.no_grad(): return torch.sigmoid(m(torch.tensor(x))).numpy()
