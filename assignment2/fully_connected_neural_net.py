import time
import numpy as np
import matplotlib.pyplot as plt
from neuralnet.fc_net import *
from neuralnet.data_utils import get_CIFAR10_data
from neuralnet.gradient_check import eval_numerical_gradient
from neuralnet.gradient_check import eval_numerical_gradient_array
from neuralnet.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)    # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

