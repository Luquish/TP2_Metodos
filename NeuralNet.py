from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
import networkx as nx


class NeuralNet:
    def __init__(self):
        """
        Inicializa los parámetros de la red neuronal.
        2 capas con 5 neuronas cada una.
        """

        self.W1: ndarray = np.random.random((5, 6))
        self.b1: ndarray = np.random.random((5, 1))

        self.W2: ndarray = np.random.random((1, 5))
        self.b2: ndarray = np.random.random((1, 1))

        self.loss_acum: List[float] = []

    def sigmoid(self, x: ndarray) -> ndarray:
        """
        Función de activación sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x: ndarray, dropout: bool = False) -> ndarray:
        """
        Calcula la salida de la red neuronal.
        """
        self.z1 = self.W1 @ x.T + self.b1
        self.a1 = self.sigmoid(self.z1)

        # if dropout:
        #    self.a1 = self.dropout(self.a1, 0.5)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.z2
        return self.a2.reshape(-1, 1)

    def numerical_gradient(self, x: ndarray, y: ndarray, eps: float) -> ndarray:
        """
        Una estrategia para calcular estas derivadas parciales,
        consiste en calcular el promedio de los cocientes incrementales
        a derecha e izquierda.
        Para obtener la siguiente aproximación, para
        cada parámetro de la red calculamos:

        $$
        \frac{\partial L}{\partial p} \sim \frac{L(\theta_t, p + \epsilon) - L(\theta_t, p - \epsilon)}{2 \epsilon}
        $$

        donde usamos $p$ de forma genérica para referirnos a cada elemento $w^1_{i,j}$, $b^1_{j}$, $w^2_{i,j}$, $b^2_{j}$
        """

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        temp_w1 = self.W1.copy()
        self.W1 = self.dropout(self.W1, 0.3)
        temp_b1 = self.b1.copy()
        self.b1 = self.dropout(self.b1, 0.3)
        temp_w2 = self.W2.copy()
        self.W2 = self.dropout(self.W2, 0.3)
        temp_b2 = self.b2.copy()
        self.b2 = self.dropout(self.b2, 0.3)

        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                self.W1[i, j] += eps
                loss1 = self.loss(x, y, dropout=True)
                self.W1[i, j] -= 2 * eps
                loss2 = self.loss(x, y, dropout=True)
                dW1[i, j] = (loss1 - loss2) / (2 * eps)
                self.W1[i, j] += eps

        for i in range(self.b1.shape[0]):
            for j in range(self.b1.shape[1]):
                self.b1[i, j] += eps
                loss1 = self.loss(x, y, dropout=True)
                self.b1[i, j] -= 2 * eps
                loss2 = self.loss(x, y, dropout=True)
                db1[i, j] = (loss1 - loss2) / (2 * eps)
                self.b1[i, j] += eps

        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                self.W2[i, j] += eps
                loss1 = self.loss(x, y, dropout=True)
                self.W2[i, j] -= 2 * eps
                loss2 = self.loss(x, y, dropout=True)
                dW2[i, j] = (loss1 - loss2) / (2 * eps)
                self.W2[i, j] += eps

        for i in range(self.b2.shape[0]):
            for j in range(self.b2.shape[1]):
                self.b2[i, j] += eps
                loss1 = self.loss(x, y, dropout=True)
                self.b2[i, j] -= 2 * eps
                loss2 = self.loss(x, y, dropout=True)
                db2[i, j] = (loss1 - loss2) / (2 * eps)
                self.b2[i, j] += eps

        self.W1 = temp_w1
        self.b1 = temp_b1
        self.W2 = temp_w2
        self.b2 = temp_b2

        return dW1, db1, dW2, db2

    def dropout(self, x: ndarray, p: float) -> ndarray:
        """
        Aplica dropout a la capa de entrada.
        x: ndarray de entrada
        p: probabilidad de dropout
        """
        mask = np.random.binomial(1, p, size=x.shape)
        return x * mask

    def update_weights(
        self,
        lr: float,
        dW1: ndarray,
        db1: ndarray,
        dW2: ndarray,
        db2: ndarray,
    ):
        """
        Actualiza los pesos de la red neuronal usando
        numerical gradient descent.
        """

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def loss(self, x: ndarray, y: ndarray, dropout: bool = False) -> float:
        """
        Calcula el error cuadrático medio.
        """

        return np.power((self.forward(x, dropout=dropout) - y), 2).mean(axis=0) / 2

    def record_metrics(self, x: ndarray, y: ndarray):
        self.loss_acum.append(self.loss(x, y))

    def fit(
        self,
        x: ndarray,
        y: ndarray,
        x_test: ndarray,
        y_test: ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        eps: float = 1e-3,
    ) -> List[float]:
        """
        Entrena la red neuronal usando gradient descent.
        """

        # for _ in tqdm(range(epochs)):
        for _ in range(epochs):
            self.record_metrics(x, y)

            dW1, db1, dW2, db2 = self.numerical_gradient(x, y, eps)

            self.update_weights(lr, dW1, db1, dW2, db2)

        return self.loss_acum

    def predict(self, x: ndarray) -> ndarray:
        """
        Infiere la salida de la red neuronal.
        """
        return self.forward(x)

    def get_weights(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Devuelve los pesos de la red neuronal.
        """

        return self.W1, self.b1, self.W2, self.b2

    def get_loss(self) -> List[float]:
        """
        Devuelve el error cuadrático medio acumulado.
        """

        return self.loss_acum

    def mse(self, y_true: ndarray, x_test: ndarray) -> float:
        """
        Calcula el error cuadrático medio.
        """

        y_pred = self.predict(x_test)
        return np.mean(np.power(y_true - y_pred, 2))

    def plot_network_graph(self, ax: plt.Axes):
        """
        Visualiza la red neuronal y sus conexiones
        """
        G = nx.DiGraph()
        G.add_nodes_from(["x1", "x2", "x3", "x4", "x5"])

        G.add_nodes_from(["z1_x1", "z1_x2", "z1_x3", "z1_x4", "z1_x5"])

        G.add_nodes_from(["y"])

        # Add edges with weights
        for i in range(5):
            for j in range(5):
                G.add_edge(f"x{i+1}", f"z1_x{j+1}", weight=self.W1[i, j])
                G.add_edge(f"z1_x{j+1}", f"y", weight=1)

        pos = {
            "x1": (0, 0),
            "x2": (0, 1),
            "x3": (0, 2),
            "x4": (0, 3),
            "x5": (0, 4),
            "z1_x1": (1, 0),
            "z1_x2": (1, 1),
            "z1_x3": (1, 2),
            "z1_x4": (1, 3),
            "z1_x5": (1, 4),
            "y": (2, 2),
        }

        # Edge labels with weights

        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap("jet"), node_size=2000, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(
            G, pos, edgelist=G.edges(), edge_color="k", arrows=True, ax=ax
        )
