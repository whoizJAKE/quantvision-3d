import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score
from quantvision.data import load_ohlcv
from quantvision.features import FEATURES,build_features,make_sequences
from quantvision.models import fit_classical,fit_transformer,transformer_proba
def run_training(ticker='BTC-USD',period='730d',interval='1h',horizon=12,epochs=20):
 df=build_features(load_ohlcv(ticker,period,interval),horizon);xs,y,idx=make_sequences(df,FEATURES,64)
 if len(y)<300: raise ValueError('Not enough usable rows; choose a longer period or interval.')
 n=len(y);tr,va=int(n*.6),int(n*.8);s=StandardScaler().fit(xs[:tr].reshape(-1,len(FEATURES)));x=s.transform(xs.reshape(-1,len(FEATURES))).reshape(xs.shape);flat=x[:,-1,:];a,b=fit_classical(flat[:tr],y[:tr]);m=fit_transformer(x[:tr],y[:tr],x[tr:va],y[tr:va],epochs);z=slice(va,None);p=pd.DataFrame(index=idx[z]);p['logistic']=a.predict_proba(flat[z])[:,1];p['boosting']=b.predict_proba(flat[z])[:,1];p['transformer']=transformer_proba(m,x[z]);p['ensemble']=p.mean(axis=1);truth=y[z];metrics={k:{'accuracy':accuracy_score(truth,(p[k]>=.5).astype(int)),'auc':roc_auc_score(truth,p[k])}for k in p};return {'predictions':p,'metrics':metrics}
