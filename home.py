import os
import pandas as pd
import streamlit as st
from utils import check_login_status
import plotly.graph_objects as go
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scraper import scrape_stock_data  # Import the scraping function

st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")

st.markdown(
    """
    <style>
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation bar
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Home"):
        st.switch_page("home.py")

with col2:
    if st.button("🔑 Login"):
        st.switch_page("/pages/login.py")

with col3:
    if st.button("📝 Register"):
        st.switch_page("/pages/register.py")



def get_available_stocks():
    stock_dir = "Stock/"
    return [f.split("_price_history.csv")[0] for f in os.listdir(stock_dir) if f.endswith("_price_history.csv")]

def ensure_stock_data(stock_symbol):
    file_path = f"Stock/{stock_symbol}_price_history.csv"
    if stock_symbol not in ["Select a stock...", "Search for a new stock..."] and not os.path.exists(file_path):
        st.warning(f"File {file_path} not found. Attempting to scrape data...")
        success = scrape_stock_data(stock_symbol)
        if not success or not os.path.exists(file_path):
            st.error("Failed to retrieve stock data. Please check the stock symbol.")
            return None
    return file_path

st.title("Stock Price Analysis")

available_stocks = get_available_stocks()
selected_stock = st.selectbox("Select a stock:", ["Select a stock..."] + available_stocks + ["Search for a new stock..."])

if selected_stock == "Search for a new stock...":
    stock_symbol = st.text_input("Enter new stock symbol:").upper()
else:
    stock_symbol = selected_stock

if stock_symbol and stock_symbol not in ["Select a stock...", "Search for a new stock..."]:
    file_path = ensure_stock_data(stock_symbol)
    if file_path:
        last_modified = os.path.getmtime(file_path)
        last_fetched_date = pd.to_datetime(last_modified, unit='s').tz_localize('UTC').tz_convert('Asia/Kathmandu').strftime('%Y-%m-%d %H:%M:%S')
        st.write(f"Last fetched: {last_fetched_date}")
        
        if st.button("Update to latest data"):
            st.write("Fetching latest data...")
            scrape_stock_data(stock_symbol)
            file_path = ensure_stock_data(stock_symbol)
            st.rerun()
        
        data = pd.read_csv(file_path, index_col="published_date", parse_dates=True)
        required_columns = {"open", "high", "low", "close", "traded_quantity"}
        
        if not required_columns.issubset(data.columns):
            st.error("CSV file is missing required columns.")
        else:
            data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "traded_quantity": "Volume"}, inplace=True)
            data = data.sort_index()
            predictors = ["Open", "High", "Low", "Close", "Volume"]
            
            train = data.iloc[:-3]
            test = data.iloc[-3:]
            
            models = {}
            predictions = {}
            for target in ["Open", "Close"]:
                model = RandomForestRegressor(n_estimators=200, random_state=1)
                model.fit(train[predictors], train[target])
                models[target] = model
                predictions[target] = model.predict(test[predictors])
            
            predictions_df = pd.DataFrame(predictions, index=test.index)
            
            # Main stock price graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
            fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig)
            
            # Biggest trend change
            st.subheader("Biggest Trend Change")
            data.loc[:, 'Daily Change'] = data['Close'].diff()
            max_trend_change = data['Daily Change'].abs().idxmax()
            max_trend_value = data.loc[max_trend_change, 'Daily Change']
            st.write(f"Biggest Trend Change: {max_trend_change.date()} with a change of {max_trend_value:.2f}")
            
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            trend_fig.add_vline(x=max_trend_change, line=dict(color='red', width=2))
            trend_fig.update_layout(title="Biggest Trend Change Highlight", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(trend_fig)
            
            # Highest and lowest stock values
            highest_value_date = data['Close'].idxmax()
            highest_value = data.loc[highest_value_date, 'Close']
            lowest_value_date = data['Close'].idxmin()
            lowest_value = data.loc[lowest_value_date, 'Close']
            st.write(f"Highest Stock Value Recorded: {highest_value:.2f} on {highest_value_date.date()}")
            st.write(f"Lowest Stock Value Recorded: {lowest_value:.2f} on {lowest_value_date.date()}")
            if not check_login_status():
                st.info("Please log in to see more content.")

            # Volatility Chart
            if check_login_status():
                st.subheader("Daily Volatility")
                data.loc[:, 'Daily Volatility'] = (data['Close'] - data['Open']) / data['Open'] * 100
                volatility_fig = go.Figure()
                volatility_fig.add_trace(go.Bar(x=data.index, y=data['Daily Volatility'], name='Volatility %'))
                volatility_fig.update_layout(title="Daily Volatility Chart", xaxis_title="Date", yaxis_title="% Change")
                st.plotly_chart(volatility_fig)
            
            # Moving Averages
                st.subheader("Moving Averages")
                data.loc[:, 'MA_7'] = data['Close'].rolling(window=7).mean()
                data.loc[:, 'MA_30'] = data['Close'].rolling(window=30).mean()
                ma_fig = go.Figure()
                ma_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA_7'], mode='lines', name='7-Day MA'))
                ma_fig.add_trace(go.Scatter(x=data.index, y=data['MA_30'], mode='lines', name='30-Day MA'))
                ma_fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(ma_fig)
            
            # Volume Spike Detection
                st.subheader("Volume Spike Detection")
                volume_threshold = data['Volume'].quantile(0.95)
                spikes = data[data['Volume'] > volume_threshold]
                st.write(f"Detected {len(spikes)} volume spikes.")
                st.dataframe(spikes[['Volume']])

                comparison = test[["Open", "Close" ]].copy()
                comparison["Predicted_Open"] = predictions_df["Open"]
                comparison["Predicted_Close"] = predictions_df["Close"]
                st.write("### Predicted vs Actual Prices for the Last 3 Days")
                st.dataframe(comparison)
        
                accuracy_open = 100 - mean_absolute_percentage_error(test["Open"], predictions_df["Open"]) * 100
                accuracy_close = 100 - mean_absolute_percentage_error(test["Close"], predictions_df["Close"]) * 100
                overall_accuracy = 100 - (mean_absolute_percentage_error(test[["Open", "Close"]], predictions_df[["Open", "Close"]]) * 100)
        
                st.write("### Model Accuracy")
                st.write(f"- Open Price Prediction Accuracy: {accuracy_open:.2f}%")
                st.write(f"- Close Price Prediction Accuracy: {accuracy_close:.2f}%")
                st.write(f"- Overall Model Accuracy: {overall_accuracy:.2f}%")
        
                st.write("### Price Trend Predictions")
                previous_close = data.iloc[-4]["Close"] if len(data) > 3 else None
                for date, row in predictions_df.iterrows():
                    predicted_close = row["Close"]
                    actual_close = test.loc[date, "Close"]
                    if previous_close is not None:
                        trend = "Increase" if predicted_close > previous_close else "Decrease"
                        correct_prediction = (trend == "Increase" and actual_close > previous_close) or (trend == "Decrease" and actual_close < previous_close)
                        correctness = "✅ Correct" if correct_prediction else "❌ Incorrect"
                        st.write(f"**{date.date()}**: Predicted Close: {predicted_close:.2f}, Trend: {trend}, Actual Close: {actual_close:.2f}, Prediction: {correctness}")
                    previous_close = actual_close
                if check_login_status():
                    st.subheader("3-Month Price Forecast with Adjusted Smoothing")
                    
                    # Prepare data for Prophet
                    prophet_data = data[['Close']].reset_index()
                    prophet_data.columns = ['ds', 'y']
                    
                    # Create and fit the model with adjustments to reduce fluctuation
                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.02,  # Lower changepoint_prior_scale for less sensitivity
                        seasonality_prior_scale=10,    # Increase seasonality prior scale for smoother seasonal fitting
                        interval_width=0.95            # Increase uncertainty interval to capture more reasonable bounds
                    )
                    
                    # Add monthly seasonality (if necessary)
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    
                    # Fit the model
                    model.fit(prophet_data)
                    
                    # Generate future dates (90 days ahead for 3 months forecast)
                    future = model.make_future_dataframe(periods=90)  # Forecast for 3 months
                    
                    # Make predictions
                    forecast = model.predict(future)
                    
                    # Create Plotly figure
                    forecast_fig = go.Figure()
                    
                    # Plot historical data (last 365 days or available data)
                    last_year = data.iloc[-365:] if len(data) > 365 else data
                    forecast_fig.add_trace(go.Scatter(
                        x=last_year.index,
                        y=last_year['Close'],
                        mode='lines',
                        name='Actual Prices (Last Year)',
                        line=dict(color='blue')
                    ))
                    
                    # Plot only the single forecasted value per day (yhat)
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast['ds'].iloc[-90:],  # Last 90 days forecast
                        y=forecast['yhat'].iloc[-90:],  # Single forecast values for each day
                        mode='lines',
                        name='Forecast Prices',
                        line=dict(color='red', dash='dot')
                    ))
                    
                    # Add uncertainty interval (just as a shaded region for the 90 days)
                    forecast_fig.add_trace(go.Scatter(
                        x=forecast['ds'].iloc[-90:].tolist() + forecast['ds'].iloc[-90:].tolist()[::-1],
                        y=forecast['yhat_upper'].iloc[-90:].tolist() + forecast['yhat_lower'].iloc[-90:].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Uncertainty Interval'
                    ))
                    
                    # Update layout for better visualization
                    forecast_fig.update_layout(
                        title='3-Month Price Forecast with Adjusted Smoothing',
                        xaxis_title='Date',
                        yaxis_title='Close Price',
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    # Plot the chart
                    st.plotly_chart(forecast_fig)

                    # Create a DataFrame to display the forecasted values and dates
                    forecast_table = forecast[['ds', 'yhat']].iloc[-90:]
                    forecast_table.columns = ['Date', 'Predicted Price']

                    # Display the table below the plot
                    st.subheader("Predicted Prices for the Next 3 Months")
                    st.table(forecast_table)

