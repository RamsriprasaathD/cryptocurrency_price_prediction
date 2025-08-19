import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from google.colab import files

# Upload the CSV file
uploaded = files.upload()
filename = next(iter(uploaded))

# Load and preprocess the data
df = pd.read_csv(filename)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Extract 'Close' prices and scale the data
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 365):  # 365 days for 1 year prediction
        X.append(data[i:(i + seq_length), 0])
        y.append(data[(i + seq_length):(i + seq_length + 365), 0])
    return np.array(X), np.array(y)

seq_length = 90  # 90 days of historical data for each prediction
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the Bi-LSTM model
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True, activation='relu'), input_shape=(seq_length, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(100, return_sequences=False, activation='relu')),
    Dropout(0.2),
    Dense(365)  # Output 365 days (1 year) prediction
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions for test set
test_predict = model.predict(X_test)

# Inverse transform predictions and actual values
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# Calculate MSE
mse = mean_squared_error(y_test, test_predict)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, test_predict) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Prepare data for plotting
train_dates = df['Date'][seq_length:train_size+seq_length]
test_dates = df['Date'][train_size+seq_length:-365]
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=365, freq='D')

# Plot the results
plt.figure(figsize=(20, 10))
plt.plot(df['Date'], scaler.inverse_transform(scaled_data), label='Actual Price', color='blue')
plt.plot(train_dates, test_predict[:, -1], label='Train Predict', color='green')
plt.plot(test_dates, test_predict[:, -1], label='Test Predict', color='red')

# Plot future predictions
last_sequence = scaled_data[-seq_length:]
last_sequence = last_sequence.reshape((1, seq_length, 1))
future_predict = model.predict(last_sequence)
future_predict = scaler.inverse_transform(future_predict)
plt.plot(future_dates, future_predict[0], label='Future Predict', color='orange')


plt.title('Cryptocurrency Price Prediction (1 Year Forecast)', fontsize=20)
plt.xlabel('Date', fontsize=17)
plt.ylabel('Price', fontsize=17)
plt.legend(fontsize=17)
plt.show()

print("Future predictions for the next year:")
for date, price in zip(future_dates[::30], future_predict[0][::30]):  # Print every 30 days
    print(f"{date.date()}: {price:.2f}")
