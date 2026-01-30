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
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
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
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

# ==============================
# 5. Evaluate Model
# ==============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", test_acc)

# ==============================
# 6. Predict and Display TWO Results
# ==============================
indices = [5, 12]   # choose any two test samples

plt.figure(figsize=(6, 3))

for i, idx in enumerate(indices):
    image = x_test[idx]
    prediction = model.predict(image.reshape(1, 28, 28), verbose=0)
    predicted_value = np.argmax(prediction)

    plt.subplot(1, 2, i + 1) 
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_value}")
    plt.axis('off')

    print(f"Image {i+1} Predicted Value:", predicted_value)

plt.tight_layout()
plt.show()
