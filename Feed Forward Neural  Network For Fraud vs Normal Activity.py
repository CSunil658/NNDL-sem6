import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. Load MNIST Dataset
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==============================
# 2. Convert to Binary Classes
# Normal (0–4) -> 0
# Fraud (5–9)  -> 1
# ==============================
y_train = (y_train >= 5).astype(int)
y_test  = (y_test >= 5).astype(int)

# ==============================
# 3. Flatten Images (28x28 -> 784)
# ==============================
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# ==============================
# 4. Normalize Data
# ==============================
x_train = x_train / 255.0
x_test  = x_test / 255.0

# ==============================
# 5. Build Feed Forward Neural Network
# ==============================
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # Binary output
])

# ==============================
# 6. Compile Model
# ==============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 7. Train Model
# ==============================
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# ==============================
# 8. Evaluate Model
# ==============================
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# ==============================
# 9. Predict New Sample
# ==============================
sample = x_test[0].reshape(1, 784)
prediction = model.predict(sample)

if prediction[0][0] > 0.5:
    print("Prediction: FRAUD activity")
else:
    print("Prediction: NORMAL activity")
