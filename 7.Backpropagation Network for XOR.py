import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(x))

def sigmoid_derivative(x):
    return x * (1-x)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)

weights_0 = 2 * np.random.random((2,4)) - 1
weights_1 = 2 * np.random.random((4,1)) - 1

for i in range(10000):
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0,weights_0))
    layer_2 = sigmoid(np.dot(layer_1,weights_1))

    error = y - layer_2

    delta2 = sigmoid_derivative(layer_2) * error
    delta1 = delta2.dot(weights_1.T) * sigmoid_derivative(layer_1)

    weights_1 += layer_1.T.dot(delta2)
    weights_0 += layer_0.T.dot(delta1)

output = sigmoid(np.dot(sigmoid(np.dot(x,weights_0)),weights_1))
print('predicted output = ',output)