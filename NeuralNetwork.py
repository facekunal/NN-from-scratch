# neural network
import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):
        # hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yhat = self.sigmoid(self.z3)
        return yhat

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
