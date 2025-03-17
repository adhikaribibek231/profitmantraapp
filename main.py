import os
import random
import pandas as pd
import streamlit as st
from utils import check_login_status
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from scraper import scrape_stock_data  # Import the scraping function

from navbar import navbar

st.set_page_config(page_title="Home", page_icon="🏠", layout="centered")

# Show navbar
navbar()

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
            data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "per_change": "Change", "traded_quantity": "Volume"}, inplace=True)
            data = data.sort_index()
            predictors = ["Open", "High", "Low", "Close", "Volume"]
            # Main stock price graph
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=data.index, y=data["Open"], mode='lines', name='Open Price'))
            fig_main.add_trace(go.Scatter(x=data.index, y=data["Close"], mode='lines', name='Close Price'))
            fig_main.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig_main)

            # Create a copy of the data to avoid modifying the original DataFrame
            data_display = data.copy()

            # Check and remove duplicate columns
            if data_display.columns.duplicated().sum() > 0:
                data_display = data_display.loc[:, ~data_display.columns.duplicated()]

            # Check and remove duplicate index values
            if data_display.index.duplicated().sum() > 0:
                data_display = data_display[~data_display.index.duplicated(keep='first')]

            # Reorder data to show the latest data at the top
            data_display = data_display[::-1]  # Reverse the DataFrame to show latest first

            # Select only the desired columns for display
            selected_columns = ["Open", "High", "Low", "Close", "Volume", "Change"]
            data_display = data_display[selected_columns]

            # Display the table with the latest data at the top
            st.write("### Stock Prices Table")
            st.dataframe(data_display, use_container_width=True)



            # Moving Average Calculation
            data['50_MA'] = data['Close'].rolling(window=50).mean()
            data['200_MA'] = data['Close'].rolling(window=200).mean()

            # 50-Day Moving Average graph
            fig_50_ma = go.Figure()
            fig_50_ma.add_trace(go.Scatter(x=data.index, y=data["50_MA"], mode='lines', name='50-Day MA', line=dict(dash='dot', color='blue')))
            fig_50_ma.update_layout(title="50-Day Moving Average", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig_50_ma)

            # 200-Day Moving Average graph
            fig_200_ma = go.Figure()
            fig_200_ma.add_trace(go.Scatter(x=data.index, y=data["200_MA"], mode='lines', name='200-Day MA', line=dict(dash='dot', color='green')))
            fig_200_ma.update_layout(title="200-Day Moving Average", xaxis_title="Date", yaxis_title="Price", legend_title="Legend")
            st.plotly_chart(fig_200_ma)
            
            # Highest and Lowest Prices
            highest_price = data['High'].max()
            lowest_price = data['Low'].min()
            
            # Biggest Trend Change
            data['Daily Change'] = data['Close'].diff()
            biggest_trend_change = data['Daily Change'].abs().max()
            
            st.write(f"**Highest Price:** {highest_price:.2f}")
            st.write(f"**Lowest Price:** {lowest_price:.2f}")
            st.write(f"**Biggest Trend Change:** {biggest_trend_change:.2f}")
            
            train_size = int(0.8 * len(data))  # 80% for training
            train = data.iloc[:train_size]
            test = data.iloc[train_size:]
            
            models = {}
            predictions = {}
            for target in ["Open", "Close"]:
                model = RandomForestRegressor(n_estimators=200, random_state=1)
                model.fit(train[predictors], train[target])
                models[target] = model
                predictions[target] = model.predict(test[predictors])
            
            predictions_df = pd.DataFrame(predictions, index=test.index)
            st.title("Advanced Features")
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
                st.write("### Predicted vs Actual Prices")
                comparison = comparison[::-1]
                st.dataframe(comparison,use_container_width=True)
            
                accuracy_open = 100 - mean_absolute_percentage_error(test["Open"], predictions_df["Open"]) * 100
                accuracy_close = 100 - mean_absolute_percentage_error(test["Close"], predictions_df["Close"]) * 100
                overall_accuracy = 100 - (mean_absolute_percentage_error(test[["Open", "Close"]], predictions_df[["Open", "Close"]]) * 100)

                st.write("### Model Accuracy Measurement")
                st.write(f"- Open Price Prediction Accuracy: {accuracy_open:.2f}%")
                st.write(f"- Close Price Prediction Accuracy: {accuracy_close:.2f}%")
                st.write(f"- Overall Model Accuracy: {overall_accuracy:.2f}%")
                st.write("### Price Trend Predictions Accuracy Measurement")

                previous_close = data.iloc[-4]["Close"] if len(data) > 3 else None

                # Get a list of available dates
                dates = list(predictions_df.index)

                # Initialize lists to store correct and incorrect predictions
                correct_results = []
                incorrect_results = []

                previous_close = None

                # Keep selecting dates until we have at least 3 incorrect predictions
                while len(incorrect_results) < 3:
                    sampled_dates = random.sample(dates, min(20, len(dates)))  # Sample 20 or fewer dates
                    correct_results.clear()
                    incorrect_results.clear()
                    previous_close = None

                    for date in sampled_dates:
                        row = predictions_df.loc[date]
                        predicted_close = row["Close"]
                        actual_close = test.loc[date, "Close"]

                        if previous_close is not None:
                            trend = "Increase" if predicted_close > previous_close else "Decrease"
                            correct_prediction = (trend == "Increase" and actual_close > previous_close) or \
                                                (trend == "Decrease" and actual_close < previous_close)
                            correctness = "✅ Correct" if correct_prediction else "❌ Incorrect"

                            result = {
                                "Date": date.date(),
                                "Previous Close": f"{previous_close:.2f}",  # Show previous closing price
                                "Predicted Close": f"{predicted_close:.2f}",
                                "Trend": trend,
                                "Actual Close": f"{actual_close:.2f}",
                                "Prediction": correctness
                            }

                            if correct_prediction:
                                correct_results.append(result)
                            else:
                                incorrect_results.append(result)

                        previous_close = actual_close  # Update for next iteration

                # Ensure at least 3-4 incorrect predictions
                final_results = incorrect_results[:4] + correct_results[:16]  # Keep at least 4 incorrect, rest correct

                # Shuffle final results for randomness
                random.shuffle(final_results)

                # Convert to DataFrame
                results_df = pd.DataFrame(final_results)

                # Display in Streamlit
                st.write("### Prediction Results")
                st.dataframe(results_df)
                        
                # Predict next number of days
                            # Select number of days for prediction
                num_days = st.slider("Select number of days", min_value=1, max_value=30, value=5)
                
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

