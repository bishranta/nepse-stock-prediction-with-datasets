import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

symbol = "ADBL" #input('Enter NEPSE symbol:')
days = 60 #input('Enter no of days for future prediction:')

# Load the CSV file
data = pd.read_csv(F"{symbol}.csv")  # Replace with your actual file path

# Sorting data according to date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.set_index('Date', inplace=True)

# Selecting the 'Close' column for price prediction
close_prices = data['Close'].values.reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Splitting training and testing data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_sequences(data, seq_length=60):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Create sequences for training and testing
x_train, y_train = create_sequences(train_data)
x_test, y_test = create_sequences(test_data)

# Reshape Data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Set timesteps and features
timesteps = 60  # Match this with the sequence length in create_sequences
features = 1    # Number of features (e.g., closing price)

# Define the model (if not previously trained)
# model = Sequential([
#     Input(shape=(timesteps, features)),
#     LSTM(50, return_sequences=True),
#     LSTM(50),
#     Dense(1)
# ])
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))
# model.save(F"{symbol}_model.keras")  # Saves the model in keras format

# Load the trained model
model = load_model(F"{symbol}.keras") # Adjust to your saved model filename

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Inverse Transform the Actual Values: Scale back the test data for comparison
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Number of future days to predict
future_days = days

# Start with the last `timesteps` days from the data you used to train
last_sequence = x_train[-1]  # Get the last sequence used in training

# Container for predictions
future_predictions = []

# Generate future predictions one at a time
for _ in range(future_days):
    # Predict the next value
    next_price = model.predict(last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1]))

    # Append prediction to future predictions list
    future_predictions.append(next_price[0, 0])

    # Update last sequence for the next prediction (remove first element, add predicted value)
    last_sequence = np.append(last_sequence[1:], next_price, axis=0)

# Inverse transform the predictions if you used a scaler
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Offset future predictions to make them consistent with the last predicted price
# Get the last predicted price from the test data
last_predicted_price = predictions[-1, 0]

# Calculate the offset
offset = last_predicted_price - future_predictions[0, 0]

# Apply the offset to the entire future predictions array
future_predictions += offset

# Create a DataFrame for future predictions
future_days_range = np.arange(1, future_days + 1)  # Days from 1 to 30
future_prices_df = pd.DataFrame({
    'Day': future_days_range,
    'Predicted Price': future_predictions.flatten()  # Flatten to 1D array for DataFrame
})

# Format the 'Predicted Price' column to 2 decimal points
future_prices_df['Predicted Price'] = future_prices_df['Predicted Price'].map('{:.2f}'.format)

# Print the future prices DataFrame
print(future_prices_df)

# Plot the actual vs. predicted prices on the test data
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test_actual, color='blue', label='Actual Price')
plt.plot(data.index[-len(y_test):], predictions, color='red', label='Predicted Price')

# Generate future date range starting from the last date in the dataset
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1)[1:]  # Only future days

# Plot future predictions after the last date in test data
plt.plot(future_dates, future_predictions, color='red', label='Future Predicted Price')

# Finalize the graph
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with Future Forecast')
plt.legend()
plt.show()
