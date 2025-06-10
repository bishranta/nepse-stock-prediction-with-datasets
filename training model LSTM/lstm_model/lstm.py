import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Disable TensorFlow optimizations for debugging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
data_dir = "./stock_data/"  # Directory containing CSV files
model_dir = "./stock_model/"  # Directory to save trained models

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Function to create sequences
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i])  # (60, 4)
        y.append(data[i])  # (4,) corresponding to OHLC
    return np.array(x), np.array(y)

seq_length = 60  # Sequence length for LSTM

# Loop through all CSV files in the stock_data directory
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        symbol = file.split(".")[0]
        try:
            print(f"Processing {symbol}...")

            # Load and preprocess the data
            data = pd.read_csv(os.path.join(data_dir, file))
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
            data.set_index('Date', inplace=True)

            # Convert string numbers with commas to float
            for column in ['Open', 'High', 'Low', 'Close']:
                data[column] = data[column].astype(str).str.replace(',', '').astype(float)

            # Selecting OHLC columns for prediction
            ohlc_prices = data[['Open', 'High', 'Low', 'Close']].values

            # Scaling the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(ohlc_prices)

            # Splitting data into training and testing sets
            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            # Create sequences for training and testing
            x_train, y_train = create_sequences(train_data, seq_length)
            x_test, y_test = create_sequences(test_data, seq_length)

            # Reshape data for LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))  # (samples, 60, 4)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))  # (samples, 60, 4)

            # Define the LSTM model
            model = Sequential([
                Input(shape=(seq_length, 4)),  # Input layer expects (60, 4)
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(100, return_sequences=False),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dense(4)  # 4 outputs (OHLC)
            ])

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_data=(x_test, y_test))

            # Save the trained model in stock_model directory
            model.save(os.path.join(model_dir, f"{symbol}.keras"))
            print(f"Model for {symbol} saved in {model_dir} as {symbol}.keras")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue  # Skip to the next file if any error occurs
