from perceptron import perceptron
from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


DIMENSION_OF_X = 2
TOTAL_SAMPLES = 1000

itr = data_generator(
    total_classes=2,
    total_dimensions=DIMENSION_OF_X,
    list_of_means=[1, 10],
    list_of_std=[1, 1]
)

X = []
Y = []
for i in range(TOTAL_SAMPLES):
    x, y = next(itr.generate(i % 2))
    X.append(x)
    Y.append([y])

X = np.array(X)
Y = np.array(Y)

node = perceptron(
    total_features=DIMENSION_OF_X,
    epochs=100,
    learning_rate=0.05
)
node.train(X, Y)

X_test = []
Y_test = []
for i in range(TOTAL_SAMPLES):
    x, y = next(itr.generate(i % 2))
    X_test.append(x)
    Y_test.append([y])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

Y_test_hat = node.predict(X_test)

print("Test Loss : {}".format(np.sum(Y_test - Y_test_hat)))

if DIMENSION_OF_X == 2:
    decision_boundary_x = [
        min(np.min(X[:, 0]-2), np.min(X_test[:, 0]-2)),
        max(np.max(X[:, 0]+2), np.max(X_test[:, 0]+2))
    ]
    intercept = node.weights[2]
    decision_boundary_y = np.dot(
        (-1/node.weights[1]),
        (
            np.dot(
                node.weights[0], decision_boundary_x
            ) + node.weights[2]
        )
    )
    plt.scatter(
        X[:, 0],
        X[:, 1],
        color=['red' if y == 0 else 'darkblue' for y in Y]
    )
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        color=['orange' if y == 0 else 'skyblue' for y in Y]
    )
    plt.plot(
        decision_boundary_x,
        decision_boundary_y,
        label='Decision Boundary'
    )
    plt.show()
