from NeuralNetwork import NeuralNetwork
import numpy as np


x = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
NN = NeuralNetwork()
yHat = NN.forward(x)
print(yHat)
