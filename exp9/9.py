import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def forward(self, X):
        self.hidden_output = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights2) + self.bias2)
        return self.output
    def backward(self, X, y, learning_rate):
        # Calculate the derivative of the cost function with respect to the output
        d_output = (self.output - y) * self.output * (1 - self.output)

        # Calculate the derivative of the cost function with respect to the hidden layer
        d_hidden_output = np.dot(d_output, self.weights2.T) * self.hidden_output * (1 - self.hidden_output)

        # Update the weights and biases
        self.weights2 -= learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias2 -= learning_rate * np.sum(d_output, axis=0)
        self.weights1 -= learning_rate * np.dot(X.T, d_hidden_output)
        self.bias1 -= learning_rate * np.sum(d_hidden_output, axis=0)

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            output = self.forward(X)
            cost = np.mean((y - output) ** 2)
            self.backward(X, y, learning_rate)
            # if i % 100 == 0:
        print(f"Epoch: {i}, Cost: {cost:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(2, 4, 1)
mlp.train(X, y, 10000, 0.1)

# Test the MLP on new data
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_output = mlp.predict(test_input)

print(test_output)