import streamlit as st
import plotly.graph_objects as go
from train import run_training
from quantvision.data import load_ohlcv
from quantvision.features import build_features
from quantvision.backtest import long_flat_backtest
st.set_page_config(page_title='QuantVision 3D',page_icon='◈',layout='wide')
st.markdown('<style>.stApp{background:#07111f}.stButton>button{background:#12b8a6;color:#06111f;border-radius:8px;border:0;font-weight:700}</style>',unsafe_allow_html=True)
st.title('◈ QuantVision 3D');st.caption('Local quantitative research • Three-model probability ensemble • Paper trading only')
with st.sidebar:
 st.header('Research controls');ticker=st.text_input('Ticker','BTC-USD').upper();period=st.selectbox('History',['180d','365d','730d','max'],index=2);interval=st.selectbox('Interval',['1h','1d','15m','5m']);horizon=st.slider('Prediction horizon (bars)',1,48,12);threshold=st.slider('Long threshold',.50,.80,.55,.01);cost=st.slider('Cost per exposure change (bps)',0,100,10);train=st.button('Train all 3 models',use_container_width=True)
try: raw=load_ohlcv(ticker,period,interval);data=build_features(raw,horizon)
except Exception as e: st.error(str(e));st.stop()
if train:
 with st.spinner('Training three models...'): st.session_state.result=run_training(ticker,period,interval,horizon)
if 'result' not in st.session_state: st.info('Download complete. Click Train all 3 models.');st.line_chart(raw.Close);st.stop()
preds=st.session_state.result['predictions'];latest=preds.iloc[-1]
for col,label,val in zip(st.columns(4),['Logistic','Boosting','Transformer','Ensemble'],[latest.logistic,latest.boosting,latest.transformer,latest.ensemble]):col.metric(label,f'{val:.1%}','bullish probability')
st.subheader(f'{ticker} • held-out test predictions');fig=go.Figure(go.Scatter(x=data.index,y=data.Close,name='Close',line=dict(color='#60a5fa')));fig.update_layout(template='plotly_dark',height=420,paper_bgcolor='#07111f',plot_bgcolor='#07111f');st.plotly_chart(fig,use_container_width=True)
backtest,stats=long_flat_backtest(data,preds.ensemble,threshold,cost)
for col,(name,val) in zip(st.columns(5),stats.items()):col.metric(name,str(val) if name=='Trades' else f'{val:.2%}')
st.subheader('Equity curve after estimated costs');st.line_chart(backtest.equity);st.subheader('Model predictions');st.dataframe(preds.tail(250).sort_index(ascending=False),use_container_width=True);st.warning('Research and paper trading only. Do not treat results as a promise or instruction to trade.')
