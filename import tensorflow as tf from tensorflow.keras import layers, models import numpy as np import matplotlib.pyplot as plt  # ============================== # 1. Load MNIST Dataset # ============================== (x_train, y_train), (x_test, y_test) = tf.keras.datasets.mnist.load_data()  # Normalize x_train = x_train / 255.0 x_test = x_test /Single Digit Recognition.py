import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Load MNIST Dataset
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# ==============================
# 2. Build Feed Forward Model
# ==============================
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),     # Input layer
    layers.Dense(128, activation='relu'),     # Hidden layer 1
    layers.Dense(64, activation='relu'),      # Hidden layer 2
    layers.Dense(10, activation='softmax')    # Output layer
])

# ==============================
# 3. Compile Model
# ==============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 4. Train Model
# ==============================
model.fit(x_train, y_train, epochs=5, batch_size=32)

# ==============================
# 5. Evaluate Model
# ==============================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# ==============================
# 6. Predict and Display Result
# ==============================
index = 5
image = x_test[index]

prediction = model.predict(image.reshape(1, 28, 28))
predicted_value = np.argmax(prediction)

plt.imshow(image, cmap='gray')
plt.title(f"Predicted Value: {predicted_value}")
plt.axis('off')
plt.show()

print("Predicted Value:", predicted_value)
