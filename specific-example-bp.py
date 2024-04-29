import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Load data
data = pd.read_csv('synthetic_uwb_data.csv')
X = data[['anchor1_dist', 'anchor2_dist', 'anchor3_dist']]
y = data[['smartphone_x', 'smartphone_y', 'smartphone_z']]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Build and train the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(3,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y, epochs=100, batch_size=32)

# Given input data
anchor1_dist = 9.61201413421608
anchor2_dist = 9.606368479035721
anchor3_dist = 5.8270497371371475

# Scale the input data
input_data = np.array([[anchor1_dist, anchor2_dist, anchor3_dist]])
input_data_scaled = scaler.transform(input_data)

# Predict the location
predicted_location = model.predict(input_data_scaled)

# Actual location
actual_location = [5, 8, 2]

# Calculate percentage accuracy
percent_accuracy = np.mean(1 - np.abs((predicted_location - actual_location) / actual_location)) * 100

print("Actual Location:\n X: 5\n Y: 8\n Z: 2")
print("Predicted Location:")
print("X:", predicted_location[0][0])
print("Y:", predicted_location[0][1])
print("Z:", predicted_location[0][2])
print("Percentage Accuracy:", percent_accuracy)
