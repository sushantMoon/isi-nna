from perceptron import perceptron
from data_generator import data_generator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

DIMENSION_OF_X = 2
TOTAL_SAMPLES = 1000
total_classes = 3
EPOCHS = 100
LEARNING_RATE = 0.1


def data_set_create(total_classes=total_classes, total_samples=TOTAL_SAMPLES):
    itr = data_generator(
        total_classes=total_classes,
        total_dimensions=DIMENSION_OF_X,
        list_of_means=[1, 10, 20],
        list_of_std=[1, 1, 1]
    )
    X = []
    Y = []
    for i in range(total_samples):
        x, y = next(itr.generate(i % total_classes))
        X.append(x)
        Y.append([y])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def tweak_Y_for_multiclass(Y, class_index):
    return np.equal(Y, class_index).astype(int)


X, Y = data_set_create(3, TOTAL_SAMPLES)

# One VS All strategy

# For Class 0
Y_class0 = tweak_Y_for_multiclass(Y, 0)
node_0 = perceptron(
    total_features=DIMENSION_OF_X,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
)
node_0.train(X, Y_class0)

# For Class 2,
Y_class2 = tweak_Y_for_multiclass(Y, 2)
node_2 = perceptron(
    total_features=DIMENSION_OF_X,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
)
node_2.train(X, Y_class2)


X_test, Y_test = data_set_create(3, TOTAL_SAMPLES)

Y_test_hat_class0 = node_0.predict(X_test)
Y_test_hat_class2 = node_2.predict(X_test)
Y_test_hat = np.zeros((TOTAL_SAMPLES, 1))

for i in range(TOTAL_SAMPLES):
    if (Y_test_hat_class0[i] == 1) and (Y_test_hat_class2[i] == 1):
        Y_test_hat[i][0] = random.choice([0, 2])
    elif Y_test_hat_class0[i] == 1:
        Y_test_hat[i][0] = 0
    elif Y_test_hat_class2[i] == 1:
        Y_test_hat[i][0] = 2
    else:
        Y_test_hat[i][0] = 1

print("Test Loss : {}".format(np.sum(np.absolute(Y_test - Y_test_hat))))


if DIMENSION_OF_X == 2:
    decision_boundary_x = [
        min(np.min(X[:, 0]-2), np.min(X_test[:, 0]-2)),
        max(np.max(X[:, 0]+2), np.max(X_test[:, 0]+2))
    ]

    decision_boundary_y_0 = np.dot(
        (-1/node_0.weights[1]),
        (
            np.dot(
                node_0.weights[0], decision_boundary_x
            ) + node_0.weights[2]
        )
    )

    decision_boundary_y_2 = np.dot(
        (-1/node_2.weights[1]),
        (
            np.dot(
                node_2.weights[0], decision_boundary_x
            ) + node_2.weights[2]
        )
    )
    color_train = ['red', 'darkblue', 'darkgreen']
    color_test = ['yellow', 'cyan', 'lightgreen']
    plt.scatter(
        X[:, 0],
        X[:, 1],
        color=[color_train[int(y)] for y in Y]
    )
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        color=[color_test[int(y)] for y in Y_test]
    )
    plt.plot(
        decision_boundary_x,
        decision_boundary_y_0,
        label='Decision Boundary For class 0'
    )
    plt.plot(
        decision_boundary_x,
        decision_boundary_y_2,
        label='Decision Boundary For class 2'
    )
    plt.legend(loc='upper left')

    plt.show()
