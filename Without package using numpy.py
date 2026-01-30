import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input vector (1 × 3)
input_data = np.array([[1, 2, 3]])

# Weight matrix (3 × 2)
weights = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

# Bias vector (1 × 2)
bias = np.array([0.1, 0.2])

# Linear combination (Z = XW + b)
z = np.dot(input_data, weights) + bias
print("Z value:", z)

# Apply activation function
activated_output = sigmoid(z)

print("Output after sigmoid:", activated_output)
