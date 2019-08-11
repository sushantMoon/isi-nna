import numpy as np


class perceptron:
    """
        Perceptron Node
        Training Method is Stochastic
        Training Data has to be provided in correct format,
        i.e.    X = np.array(np.array())
        and     Y = np.array()

        Assumtions :
            1. 2 classes
            2. Data is linearly seperable
        Capacities :
            1. Multi Dimensional X i.e X of any dimension would work
    """
    def __init__(
        self, total_features, epochs=100, learning_rate=0.05, print_loss=False
    ):
        """
        Arguments:
            total_features {int} -- total number of inputs to the perceptron

        Keyword Arguments:
            epochs {int} -- total times the total traingin set will be seen by
            the perceptron (default: {100})
            learning_rate {float} -- learning rate for weight updation when
            training (default: {0.05})
            print_loss {bool} -- printing the loss while training(simple total
            misclassifications)
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(total_features + 1)
        self.print_loss = print_loss

    def predict(self, X):
        """Preceptron Predict Function

        Arguments:
            X {numpy.array(numpy.array)} -- numpy array of numpy array,
            each of the inner numpy array contains
            features/independent variables for a single sample
            Shape : m (number of sample) x n (number of features)
        Returns :
            Y_hat {numpy.array(numpy.array())} -- Predictions results
            Each row in Y corresponds to predictions for each row in X
        """
        # appending one as a column to X for bais
        ones = np.ones((X.shape[0], 1))
        with_bais_X = np.hstack((X, ones))
        dot_product = np.dot(with_bais_X, self.weights)
        Y_hat = np.where(dot_product > 0, 1, 0)
        return np.reshape(Y_hat, (X.shape[0], 1))

    def train(self, X, Y):
        """Perceptron Training Function

        Arguments:
            X {numpy.array(numpy.array())} -- numpy array of numpy array,
            each of the inner numpy array contains
            features/independent variables for a single sample
            Shape : m (number of sample) x n (number of features)

            Y {numpy.array(numpy.array())} -- Ground Truth values for
            each of the samples, stating the class they belong to.
            Shape: m (number of samples) x 1
        """
        for epoch in range(self.epochs):
            input_X = X
            Y_hat = self.predict(input_X)
            ones = np.ones((X.shape[0], 1))
            with_bais_X = np.hstack((X, ones))
            if self.print_loss:
                print("Epoch={}, Loss={}".format(epoch, np.sum(Y-Y_hat)))
            self.weights = self.weights + np.sum(
                self.learning_rate * (Y - Y_hat) * with_bais_X,
                axis=0
            )
"""
### For Testing
import numpy as np
X = np.array([[1,1,1], [-1,1,-1]])
Y = np.reshape(np.array([1,0]),(2,1))
from perceptron import perceptron
p = perceptron(3, 1)
p.train(X, Y)
"""
