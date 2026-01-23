import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([[0,0], [0,1], [1,0], [1,1]])
y=np.array([[0], [1], [1], [0]])

model = Sequential()    
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10000, verbose=0)
loss, accuracy = model.evaluate(x, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

ouput:

Loss: ~0.001 to 0.02
Accuracy: 1.0
