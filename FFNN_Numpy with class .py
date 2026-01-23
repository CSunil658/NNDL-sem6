import numpy as np

class FeedForward:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.w1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.w2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        self.w2 += self.a1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.lr
        self.w1 += X.T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr

    def compute_loss(self, y, output):
        return np.mean((y - output) ** 2)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")   

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = FeedForward(input_size=2, hidden_size=3, output_size=1, lr=0.1)
nn.train(X, y)

ouput:

Epoch 0, Loss: 0.2975023794196643
Epoch 1000, Loss: 0.24903347007437893
Epoch 2000, Loss: 0.23947182586572824
Epoch 3000, Loss: 0.1528877188600279
Epoch 4000, Loss: 0.033612198494107896
Epoch 5000, Loss: 0.012690504371444712
Epoch 6000, Loss: 0.007248742225658798
Epoch 7000, Loss: 0.004947528244616218
Epoch 8000, Loss: 0.003712440630159579
Epoch 9000, Loss: 0.0029521685283091332
