import numpy as np
import os 
import six.moves.urllib.request as ul
import urllib
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from tensorflow import keras

def __check_datasets_folder(datasets_folder='datasets/'):
    """ Checks if the datasets folder exists and creates it if not.
    Parameters
    ----------
    datasets_folder : str, optional
        The path to the datasets folder used.
    """

    if not os.path.isdir(datasets_folder):
        os.makedirs(datasets_folder)

def format_y_if_necessary(y_train, y_test):
    if len(y_train.shape) > 1:
        y_train = np.squeeze(y_train)
    if len(y_test.shape) > 1:
        y_test = np.squeeze(y_test)

    # The lowest class should be 0
    min_class = np.min(y_train)
    y_train -= min_class
    y_test -= min_class

    return y_train, y_test

def format_x_if_necessary(X_train, X_test):
    if len(X_train.shape) == 4 and X_train.shape[-1] == 3:
        X_train = np.transpose(X_train, [0, 3, 1, 2])
    if len(X_test.shape) == 4 and X_test.shape[-1] == 3:
        X_test = np.transpose(X_test, [0, 3, 1, 2])

    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, 1)
    if len(X_test.shape) == 3:
        X_test = np.expand_dims(X_test, 1)
    return X_train, X_test

def get_dataset_name_for_loading_function(loading_function):
    if loading_function == load_pendigits_data:
        return "pendigits"
    elif loading_function == load_hand_posture_data:
        return "hand_posture"
    elif loading_function == load_mnist_data:
        return "mnist"
    elif loading_function == load_cifar10_data:
        return "cifar10"
    else:
        raise NotImplementedError("Loading function not recognized: " + str(loading_function))

def is_image_dataset(loading_function):
    if loading_function == load_pendigits_data:
        return False
    elif loading_function == load_hand_posture_data:
        return False
    elif loading_function == load_mnist_data:
        return True
    elif loading_function == load_cifar10_data:
        return True
    else:
        raise NotImplementedError("Loading function not recognized: " + str(loading_function))

def load_pendigits_data(url='https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/',
                         datasets_folder='datasets/'):
    __check_datasets_folder(datasets_folder=datasets_folder)

    pendigits_folder = os.path.join(datasets_folder, "pendigits")
    
    if not os.path.isdir(pendigits_folder):
        if not os.path.isdir(pendigits_folder):
            os.mkdir(pendigits_folder)
        ul.urlretrieve(urllib.parse.urljoin(url, "pendigits.tra") , os.path.join(pendigits_folder, 'pendigits.tra'))
        ul.urlretrieve(urllib.parse.urljoin(url, "pendigits.tes") , os.path.join(pendigits_folder, 'pendigits.tes'))

    train_data = pd.read_csv(os.path.join(pendigits_folder, 'pendigits.tra'), delimiter=",", header=None)
    test_data = pd.read_csv(os.path.join(pendigits_folder, 'pendigits.tes'), delimiter=",", header=None)

    X_train = train_data.values[:, :-1]
    y_train = train_data.values[:, -1]

    X_test = test_data.values[:, :-1]
    y_test = test_data.values[:, -1]

    y_train, y_test = format_y_if_necessary(y_train, y_test)
    X_train, X_test = format_x_if_necessary(X_train, X_test)

    return (X_train, y_train), (X_test, y_test)

def load_hand_posture_data(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00405/Postures.zip',
                         datasets_folder='datasets/'):
    __check_datasets_folder(datasets_folder=datasets_folder)

    handposturefolder = os.path.join(datasets_folder, "HandPostures")

    if not os.path.isdir(handposturefolder):
        ul.urlretrieve(url, os.path.join(datasets_folder, 'Postures.zip'))
        with ZipFile(os.path.join(datasets_folder, 'Postures.zip'), 'r') as zip_ref:
            zip_ref.extractall(handposturefolder)
        os.remove(os.path.join(datasets_folder, 'Postures.zip'))

    data = pd.read_csv(os.path.join(handposturefolder, "Postures.csv"), skiprows=(1,))
    data.replace('?', 0., inplace=True)
    y_data = data.Class.values
    X_data = data.copy()
    X_data.drop(columns=["Class", "User"], inplace=True)
    X_data = X_data.values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

    y_train, y_test = format_y_if_necessary(y_train, y_test)
    X_train, X_test = format_x_if_necessary(X_train, X_test)

    return (X_train, y_train), (X_test, y_test)

def load_mnist_data(datasets_folder='datasets/'):
    __check_datasets_folder(datasets_folder=datasets_folder)

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    y_train, y_test = format_y_if_necessary(y_train, y_test)
    X_train, X_test = format_x_if_necessary(X_train, X_test)

    return (X_train, y_train), (X_test, y_test)

def load_cifar10_data(datasets_folder='datasets/'):
    __check_datasets_folder(datasets_folder=datasets_folder)

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    y_train, y_test = format_y_if_necessary(y_train, y_test)
    X_train, X_test = format_x_if_necessary(X_train, X_test)

    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_hand_posture_data()
    print("Hand Posture: # Train Samples: " + str(X_train.shape[0]) + " # Test Samples: " + str(X_test.shape[0]) + " Inp Shape: " + str(X_train.shape) + " Classes: " + str(np.max(y_train) + 1))
    (X_train, y_train), (X_test, y_test) = load_pendigits_data()
    print("Pendigits: # Train Samples: " + str(X_train.shape[0]) + " # Test Samples: " + str(X_test.shape[0]) + " Inp Shape: " + str(X_train.shape) + " Classes: " + str(np.max(y_train) + 1))
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print("MNIST: # Train Samples: " + str(X_train.shape[0]) + " # Test Samples: " + str(X_test.shape[0]) + " Inp Shape: " + str(X_train.shape) + " Classes: " + str(np.max(y_train) + 1))
    (X_train, y_train), (X_test, y_test) = load_cifar10_data()
    print("Cifar10: # Train Samples: " + str(X_train.shape[0]) + " # Test Samples: " + str(X_test.shape[0]) + " Inp Shape: " + str(X_train.shape) + " Classes: " + str(np.max(y_train) + 1))

    halt = 1