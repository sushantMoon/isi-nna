import numpy as np


class data_generator:
    """
    Class for generating data Uniformly at random from a
    Normal Distribution with pre-specified mean and variance
    """
    def __init__(
        self,
        total_classes,
        total_dimensions,
        list_of_means,
        list_of_std
    ):
        """Initialization for the data generator
        Arguments:
            total_classes {int} -- total number of classes for which data
            is being generated
            total_dimensions {int} -- total number of
            dimensions/features/independent variables for a given sample
            list_of_means {list} -- list of the means for the different
            classes, Note: the index of any given mean corresponds to the
            class to which the mean belongs
            list_of_std {list} -- list of the standard deviation for the
            different classes, Note: the index of any given standard deviation
            corresponds to
            the class to which the standard deviation belongs
        """
        self.total_classes = total_classes
        self.list_of_means = list_of_means
        self.list_of_std = list_of_std
        self.total_dimensions = total_dimensions

    def generate(self, y):
        """Data Generator Iterator Function
        Arguments:
            y {int} -- Class for which we are generating the data
            Note : The value supplied should be less than the total number of
            class earlier defined
        """
        if y < self.total_classes:
            X = [
                    x_i for x_i in np.random.normal(
                        self.list_of_means[y],
                        self.list_of_std[y],
                        self.total_dimensions
                    )
                ]
            yield X, y
        else:
            print(
                "The class supplied should be less than {}".format(
                    self.total_classes
                )
            )
            yield None, None

"""
from data_generator import data_generator
import matplotlib.pyplot as plt
import numpy as np
d = data_generator(2, 2, [1, 10], [1,1])

X1 = []
Y1 = []
for i in range(10):
    x, _ = next(d.generate(0))
    X1.append(x)
    Y1.append(0)

X1 = np.array(X1)
Y1 = np.array(Y1)
plt.scatter(X1[:,0], X1[:, 1], marker='o')

X2 = []
Y2 = []
for i in range(10):
    x, _ = next(d.generate(1))
    X2.append(x)
    Y2.append(1)

X2 = np.array(X2)
Y2 = np.array(Y2)
plt.scatter(X2[:,0], X2[:, 1], marker='^')

plt.show()
"""
