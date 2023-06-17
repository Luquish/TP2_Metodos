"""
Para entrenar el regresor neuronal utilizaremos los mismos datos del TP1. Para poder
evaluar la performance de nuestra red, dividiremos el dataset en dos conjuntos: uno que utiliza-
remos únicamente para el entrenamiento (315 muestras), y otro para la evaluación (99 muestras)
"""

import numpy as np


class NeuralNet:
    def __init__(self, X_train, y_train):

        self.W1 = np.random.random((1, X_train.shape[0]))
        self.b1 = np.random.random((5, 1))
        self.W2 = np.random.random((1, 5))
        self.b2 = np.random.random((1, 1))

        self.loss_acum = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        return self.z2

    def loss(self, eps, lr, x, y):
        # Numerical Gradient
        dW1 = (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)
        db1 = (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)
        dW2 = (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)
        db2 = (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)

        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return np.sum((self.forward(x) - y) ** 2) / 2

    def fit(self, x, y, lr=0.01, epochs=1000):

        eps = 1e-3

        for _ in range(epochs):

            self.loss_acum.append(self.loss(eps, lr, x, y))

        return self.loss_acum

    def predict(self, x):
        return self.forward(x)

    def get_weights(self):
        return self.W1, self.b1, self.W2, self.b2

    def get_loss(self):
        return self.loss_acum
