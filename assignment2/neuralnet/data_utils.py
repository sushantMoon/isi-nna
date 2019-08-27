import _pickle as pickle
import numpy as np
import os
from sklearn import datasets
from sklearn.metrics import confusion_matrix

WORKING_DIR = '/mnt/Alice/ISI/SEM3/NNA/Assignments/assignment2/'


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    dir_path=WORKING_DIR, num_training=49000,
    num_validation=1000, num_test=1000
        ):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    # print(dir_path)
    cifar10_dir = os.path.join(dir_path, "data_dir/cifar-10-batches-py")
    # print(cifar10_dir)
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def get_iris_data(
        ratio_training=0.7,
        ratio_validation=0.1,
        ratio_test=0.2
        ):
    """
    Load the IRIS dataset using the sklearn.dataset
    """
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    target = target[indices]
    train_n = int(iris.data.shape[0] * ratio_training)
    test_n = int(iris.data.shape[0] * ratio_test)
    val_n = int(iris.data.shape[0] * ratio_validation)

    mask = range(train_n)
    X_train = data[mask]
    y_train = target[mask]

    mask = range(train_n, train_n + val_n)
    X_val = data[mask]
    y_val = target[mask]

    mask = range(train_n + val_n, train_n + val_n + test_n)
    X_test = data[mask]
    y_test = target[mask]

    # Normalize the data
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_val -= mean
    X_test -= mean

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will
    be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)