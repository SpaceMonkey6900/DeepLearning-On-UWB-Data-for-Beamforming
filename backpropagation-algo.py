import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Load and preprocess the data
data = pd.read_csv('synthetic_uwb_data.csv')
X = data[['anchor1_dist', 'anchor2_dist', 'anchor3_dist']]
y = data[['smartphone_x', 'smartphone_y', 'smartphone_z']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(3,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='linear'))

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print('Test loss:', loss)

# Verify the authenticity of the algorithm
num_samples = 10  # Number of samples to verify
random_indices = np.random.choice(len(X_test), num_samples, replace=False)
X_verify = X_test.iloc[random_indices]
y_verify = y_test.iloc[random_indices]
X_verify_scaled = scaler.transform(X_verify)

# Make predictions on the verification samples
predictions = model.predict(X_verify_scaled)

# Print the actual and predicted coordinates with accuracy
print("Verification Results:")
for i in range(num_samples):
    actual = y_verify.iloc[i].values
    predicted = predictions[i]
    distance = np.linalg.norm(actual - predicted)  # Calculate Euclidean distance
    accuracy = (1 - distance) * 100  # Convert distance to percentage accuracy
    print(f"Sample {i+1}:")
    print(f"Actual Coordinates: {actual}")
    print(f"Predicted Coordinates: {predicted}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
