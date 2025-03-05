import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go

# Load data (assuming df has 'ds' as date and 'y' as target variable)
df = pd.read_csv("your_data.csv")
df['ds'] = pd.to_datetime(df['ds'])

# Prophet model with adjusted parameters
model = Prophet(
    seasonality_mode='additive',  # Reduce extreme seasonal fluctuations
    changepoint_prior_scale=0.01,  # Lower sensitivity to rapid changes
    interval_width=0.8  # Reduce uncertainty range
)
model.fit(df)

# Create future dataframe (weekly instead of daily)
future = model.make_future_dataframe(periods=52, freq='W')  # Weekly prediction
forecast = model.predict(future)

# Plot results
fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y'],
    mode='lines', name='Actual Prices (Last Year)',
    line=dict(color='blue')
))

# Forecasted prices (weekly)
fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    mode='lines', name='Forecast Prices',
    line=dict(color='red', dash='dot')
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
    fill='toself', fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Uncertainty Interval'
))

fig.update_layout(
    title='1-Year Price Forecast with Uncertainty',
    xaxis_title='Date',
    yaxis_title='Close Price',
    template='plotly_dark'
)

st.plotly_chart(fig)
