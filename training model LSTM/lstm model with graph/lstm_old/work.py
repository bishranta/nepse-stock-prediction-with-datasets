# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional



# Step 1: Load and Preprocess Data
# Read the CSV file containing stock data
gstock_data = pd.read_csv('NABIL.csv')

# Rename columns to standardized names for easier use
gstock_data = gstock_data.rename(columns={'Date': 'date', 'Open': 'open', 'Close': 'close'})

# Display the first few rows of the dataframe to confirm data is loaded correctly
gstock_data.head()

# Select only the relevant columns: date, open, and close
gstock_data = gstock_data[['date', 'open', 'close']]

# Convert the date column to datetime format, handling possible errors
gstock_data['date'] = pd.to_datetime(gstock_data['date'].apply(lambda x: x.split()[0] if isinstance(x, str) else x),
                                     errors='coerce')

# Set the 'date' column as the index and drop the original date column
gstock_data.set_index('date', drop=True, inplace=True)

# Check the first few rows after preprocessing
print(gstock_data.head())

# Step 2: Plotting Open and Close prices with date formatting
fg, ax = plt.subplots(1, 2, figsize=(20, 7))  # Create two subplots

# Plot 'open' prices with green color
ax[0].plot(gstock_data['open'], label='Open', color='green')
ax[0].set_xlabel('Date', size=15)
ax[0].set_ylabel('Price', size=15)
ax[0].legend()

# Format the x-axis to show months and years
ax[0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Plot 'close' prices with red color
ax[1].plot(gstock_data['close'], label='Close', color='red')
ax[1].set_xlabel('Date', size=15)
ax[1].set_ylabel('Price', size=15)
ax[1].legend()

# Format the x-axis to show months and years
ax[1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Display the plot
plt.show()

# Step 3: Data Normalization using MinMaxScaler
# Initialize the MinMaxScaler to scale data between 0 and 1
Ms = MinMaxScaler()

# Select only numeric columns (open, close) for scaling
numeric_columns = gstock_data.select_dtypes(include=['float64', 'int64']).columns

# Apply MinMax scaling to numeric columns
gstock_data[numeric_columns] = Ms.fit_transform(gstock_data[numeric_columns])
print(gstock_data)
# Step 4: Split Data into Training and Testing Sets
# Split the data into 80% training and 20% testing sets, with a fixed random seed for reproducibility
train_data, test_data = train_test_split(gstock_data, test_size=0.20, random_state=42)

# 2. Fit the scaler only on the training data
Ms.fit(train_data[numeric_columns])

# 3. Transform both the training and test data
train_data[numeric_columns] = Ms.transform(train_data[numeric_columns])
test_data[numeric_columns] = Ms.transform(test_data[numeric_columns])

# Step 5: Define a function to create sequences for time series forecasting
def create_sequence(dataset, sequence_length=50):
    """
    Creates sequences of a given length from the dataset.
    Each sequence is followed by the next value as the label.

    Parameters:
    dataset (DataFrame): The dataset to create sequences from
    sequence_length (int): Length of each sequence (default is 50)

    Returns:
    sequences (np.array): Array of sequences of size [num_samples, sequence_length, num_features]
    labels (np.array): Array of labels of size [num_samples, num_features]
    """
    sequences = []
    labels = []

    for i in range(len(dataset) - sequence_length):
        # Extract a sequence of length `sequence_length`
        sequences.append(dataset.iloc[i:i + sequence_length].values)
        # Label is the next value after the sequence
        labels.append(dataset.iloc[i + sequence_length].values)

    return np.array(sequences), np.array(labels)


# Step 6: Create sequences for training and testing datasets
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# Step 7: (Optional) Plot Open and Close prices again to visualize the data after normalization
fig, ax = plt.subplots(1, 2, figsize=(20, 7))  # Create two subplots

# Plot normalized 'open' prices in green
ax[0].plot(gstock_data['open'], label='Open', color='green')
ax[0].set_xlabel('Date', size=15)
ax[0].set_ylabel('Price', size=15)
ax[0].legend()

# Plot normalized 'close' prices in red
ax[1].plot(gstock_data['close'], label='Close', color='red')
ax[1].set_xlabel('Date', size=15)
ax[1].set_ylabel('Price', size=15)
ax[1].legend()

plt.show()  # Display the plot

# Initialize the model
model = Sequential()

# Add the first LSTM layer with 50 units and return_sequences=True
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))

# Add a dropout layer to prevent overfitting
model.add(Dropout(0.1))

# Add the second LSTM layer with 50 units
model.add(LSTM(units=50))

# Add a Dense layer with 2 units for the output (open and close prices)
model.add(Dense(2))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# Display the model summary
model.summary()

# Train the model
model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)

# Make predictions on the test set
test_predicted = model.predict(test_seq)

# Inverse transform the predictions to original scale
test_inverse_predicted = Ms.inverse_transform(test_predicted)

# Optionally, if you want to visualize or further analyze the predictions
# You can compare test_inverse_predicted with the actual values from test_label


# Get the number of predictions made
num_predictions = test_inverse_predicted.shape[0]  # This should be the same as the number of test sequences

# Merging actual and predicted data for better visualization
# Ensure that you slice only the required number of last rows based on the predictions
gs_slic_data = pd.concat([
    gstock_data.iloc[-num_predictions:].copy(),
    pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'], index=gstock_data.iloc[-num_predictions:].index)
], axis=1)

# Correctly inverse transforming actual values for open and close prices
gs_slic_data[['open', 'close']] = Ms.inverse_transform(gs_slic_data[['open', 'close']])

# Display the merged DataFrame
print(gs_slic_data.tail())  # Print the last few rows to confirm the merge and transformation

# Optional: Plot Actual vs Predicted Prices
plt.figure(figsize=(14, 7))
plt.plot(gs_slic_data.index, gs_slic_data['open'], label='Actual Open Price', color='green')
plt.plot(gs_slic_data.index, gs_slic_data['open_predicted'], label='Predicted Open Price', color='red', linestyle='--')
plt.plot(gs_slic_data.index, gs_slic_data['close'], label='Actual Close Price', color='blue')
plt.plot(gs_slic_data.index, gs_slic_data['close_predicted'], label='Predicted Close Price', color='orange', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
