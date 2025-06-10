import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Disable TensorFlow optimizations for debugging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to create sequences
def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Directory containing stock data CSV files
data_dir = "./stock_data/"  # Adjust path as needed
seq_length = 60

# Loop through all CSV files in the directory
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

            # Selecting the 'Close' column for price prediction
            close_prices = data['Close'].values.reshape(-1, 1)

            # Scaling the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Splitting data into training and testing sets
            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            # Create sequences for training and testing
            x_train, y_train = create_sequences(train_data, seq_length)
            x_test, y_test = create_sequences(test_data, seq_length)

            # Reshape data for LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Define the LSTM model
            model = Sequential([
                Input(shape=(seq_length, 1)),
                LSTM(50, return_sequences=True),
                LSTM(50),
                Dense(1)
            ])

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=0, validation_data=(x_test, y_test))

            # Save the trained model
            model.save(os.path.join(data_dir, f"{symbol}.keras"))
            print(f"Model for {symbol} saved as {symbol}.keras")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue  # Skip to the next file if any error occurs
