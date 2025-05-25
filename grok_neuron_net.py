import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        error = y - output
        delta2 = error * self.sigmoid_derivative(output)
        error_hidden = np.dot(delta2, self.W2.T)
        delta1 = error_hidden * self.sigmoid_derivative(self.a1)
        self.W2 += lr * np.dot(self.a1.T, delta2)
        self.b2 += lr * np.sum(delta2, axis=0, keepdims=True)
        self.W1 += lr * np.dot(X.T, delta1)
        self.b1 += lr * np.sum(delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(2, 4, 1)
nn.train(X, y, 10000, 0.1)
print("\nPredictions:")
for i, x in enumerate(X):
    pred = nn.forward(x.reshape(1, -1))[0][0]
    print(f"Input: {x}, Predicted: {pred:.4f}, Actual: {y[i][0]}")
