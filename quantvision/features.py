import numpy as np
import pandas as pd

FEATURES=['ret_1','ret_5','ret_20','vol_20','range_pct','volume_z','ma_gap_10','ma_gap_30','rsi_14','atr_pct','hour_sin','hour_cos','dow_sin','dow_cos']
def build_features(ohlcv,horizon=12):
    df=ohlcv.copy().sort_index(); c,h,l=df.Close,df.High,df.Low
    df['ret_1']=np.log(c/c.shift());df['ret_5']=np.log(c/c.shift(5));df['ret_20']=np.log(c/c.shift(20));df['vol_20']=df.ret_1.rolling(20).std();df['range_pct']=(h-l)/c
    vm=df.Volume.rolling(30).mean();vs=df.Volume.rolling(30).std().replace(0,np.nan);df['volume_z']=(df.Volume-vm)/vs;df['ma_gap_10']=c/c.rolling(10).mean()-1;df['ma_gap_30']=c/c.rolling(30).mean()-1
    d=c.diff();rs=d.clip(lower=0).rolling(14).mean()/(-d.clip(upper=0)).rolling(14).mean().replace(0,np.nan);df['rsi_14']=100-100/(1+rs)
    p=c.shift();tr=pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1);df['atr_pct']=tr.rolling(14).mean()/c
    hr,dow=df.index.hour,df.index.dayofweek;df['hour_sin']=np.sin(2*np.pi*hr/24);df['hour_cos']=np.cos(2*np.pi*hr/24);df['dow_sin']=np.sin(2*np.pi*dow/7);df['dow_cos']=np.cos(2*np.pi*dow/7)
    df['future_return']=c.shift(-horizon)/c-1;df['target']=(df.future_return>0).astype(int);return df.dropna().copy()
def make_sequences(df,features=FEATURES,lookback=64):
    x=df[features].to_numpy(dtype=np.float32);y=df.target.to_numpy(dtype=np.float32);return np.asarray([x[i-lookback+1:i+1] for i in range(lookback-1,len(df))]),y[lookback-1:],df.index[lookback-1:]
