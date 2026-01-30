import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# Input data
X_train = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
], dtype=np.float32)
# Target values
y_train = np.array([
    [0.2],
    [0.4],
    [0.6],
    [0.8]
], dtype=np.float32)
# Create Feed Forward Neural Network
model = Sequential()
model.add(Dense(4, input_shape=(3,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='sgd', loss='mse')
# Train the model
model.fit(X_train, y_train, epochs=3000, verbose=0)
# Predict output
output = model.predict(X_train)
print("Predicted Output:\n", output)
