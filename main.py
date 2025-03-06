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
            # Moving Average Calculation
            data['50_MA'] = data['Close'].rolling(window=50).mean()
            data['200_MA'] = data['Close'].rolling(window=200).mean()

            # Main stock price graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data["50_MA"], mode='lines', name='50-Day MA', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=data.index, y=data["200_MA"], mode='lines', name='200-Day MA', line=dict(dash='dot')))
            fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig)
            
            # Highest and Lowest Prices
            highest_price = data['High'].max()
            lowest_price = data['Low'].min()
            
            # Biggest Trend Change
            data['Daily Change'] = data['Close'].diff()
            biggest_trend_change = data['Daily Change'].abs().max()
            
            st.write(f"**Highest Price:** {highest_price:.2f}")
            st.write(f"**Lowest Price:** {lowest_price:.2f}")
            st.write(f"**Biggest Trend Change:** {biggest_trend_change:.2f}")
            
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
            if check_login_status():
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

                st.write("### Model Accuracy Measurement")
                st.write(f"- Open Price Prediction Accuracy: {accuracy_open:.2f}%")
                st.write(f"- Close Price Prediction Accuracy: {accuracy_close:.2f}%")
                st.write(f"- Overall Model Accuracy: {overall_accuracy:.2f}%")

                st.write("### Price Trend Predictions Accuracy measurement")
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
                        
                # Predict next number of days
                            # Select number of days for prediction
                num_days = st.slider("Select number of days", min_value=1, max_value=30, value=5)
                num_days = st.number_input("Or enter number of days", min_value=1, max_value=30, value=num_days)
                
                # Predict next number of days
                future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_days)
                future_predictions = {}
                for target in ["Open", "Close"]:
                    future_predictions[target] = models[target].predict(data[predictors].iloc[-3:].mean().values.reshape(1, -1))
                
                st.write("### Price Trend Predictions for Next selected number of Days") 
                previous_close = data.iloc[-1]["Close"]
                previous_open = data.iloc[-1]["Open"]
                for i, date in enumerate(future_dates):
                    predicted_open = future_predictions["Open"][0] + (i * 0.005 * previous_open)  # Slight variation simulation
                    predicted_close = future_predictions["Close"][0] + (i * 0.01 * previous_close)  # Slight variation simulation
                    trend = "Increase" if predicted_close > previous_close else "Decrease"
                    st.write(f"**{date.date()}**: Predicted Open: {predicted_open:.2f}, Predicted Close: {predicted_close:.2f}, Trend: {trend}")
                    previous_open = predicted_open
                    previous_close = predicted_close
            if not check_login_status():
                st.info("Please log in to see more content.")

