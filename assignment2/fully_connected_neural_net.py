import time
import numpy as np
import os
from neuralnet.fc_net import FullyConnectedNet
from neuralnet.data_utils import get_CIFAR10_data, get_iris_data
from neuralnet.gradient_check import eval_numerical_gradient
from neuralnet.gradient_check import eval_numerical_gradient_array
from neuralnet.solver import Solver
from neuralnet.save_load import SaveLoad
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0)
# set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# WORKING_DIR = "/user1/student/mtc/mtc2018/mtc1807/SEM3/NNA/assignment2/"
WORKING_DIR = "/mnt/Alice/ISI/SEM3/NNA/Assignments/assignment2/"
SOLVER_PICKLE = "solver_iris.pkl"
LAYER_PICKLE = "fc_iris.pkl"


def train_fc_net():
    path_to_save_model = WORKING_DIR + LAYER_PICKLE
    path_to_save_solver = WORKING_DIR + SOLVER_PICKLE

    # data = get_CIFAR10_data(dir_path=WORKING_DIR)
    data = get_iris_data()
    # learning_rate = 10**(np.random.uniform(-4, -1))
    # weight_scale = 10**(np.random.uniform(-6, -1))
    learning_rate = 1e-3
    weight_scale = 5e-2
    model = FullyConnectedNet(
        [100, 100, 100],                # for IRIS
        input_dim=4,                    # for IRIS
        num_classes=3,                  # for IRIS
        # [100, 100, 100, 100, 100],    # for CIFAR10
        weight_scale=weight_scale,
        dtype=np.float64,
        reg=0.1
    )
    solver = Solver(
                model,
                data,
                print_every=1,
                num_epochs=20,
                batch_size=1000,
                update_rule='adam',
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

    SaveLoad.save_object(path_to_save_model, model)
    SaveLoad.save_object(path_to_save_solver, solver)


def test_model():
    path_to_save_model = WORKING_DIR + LAYER_PICKLE
    path_to_save_solver = WORKING_DIR + SOLVER_PICKLE
    _ = SaveLoad.load_object(path_to_save_model)
    solver = SaveLoad.load_object(path_to_save_solver)

    # data = get_CIFAR10_data(dir_path=WORKING_DIR)
    data = get_iris_data()

    testing_accuracy = solver.check_accuracy(
        data['X_test'],
        data['y_test']
    )
    print("Testing Accuracy : {}".format(testing_accuracy))


if __name__ == "__main__":
    train_fc_net()
    test_model()
