import time
import numpy as np
import os
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
from neuralnet.fc_net import FullyConnectedNet
from neuralnet.data_utils import get_CIFAR10_data
from neuralnet.gradient_check import eval_numerical_gradient
from neuralnet.gradient_check import eval_numerical_gradient_array
from neuralnet.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)    # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# WORKING_DIR = "/user1/student/mtc/mtc2018/mtc1807/SEM3/NNA/assignment2/"
WORKING_DIR = "/mnt/Alice/ISI/SEM3/NNA/Assignments/assignment2/"


def main():
    data = get_CIFAR10_data(dir_path=WORKING_DIR)
    learning_rate = 10**(np.random.uniform(-4, -1))
    weight_scale = 10**(np.random.uniform(-6, -1))
    model = FullyConnectedNet(
        [100, 200, 200, 100],
        weight_scale=weight_scale,
        dtype=np.float64,
        reg=0.1
    )
    solver = Solver(
                model,
                data,
                print_every=100,
                num_epochs=40,
                batch_size=1000,
                update_rule='sgd',
                optim_config={
                    'learning_rate': learning_rate,
                    }
             )
    solver.train()
    print(
        "Worked with Learning rate : {}\n"
        "Initial weight Standard Deviation : {}".format(
            learning_rate,
            weight_scale
            )
        )
    plt.plot(solver.train_acc_history, marker='o', label="Training Acc")
    plt.plot(solver.val_acc_history, marker='x', label="Validation Acc")
    plt.legend(loc='upper left')
    plt.title('Accuracy history')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig(WORKING_DIR+'accuracy.png')


if __name__ == "__main__":
    main()
