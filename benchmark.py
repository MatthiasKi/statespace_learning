import pickle
import numpy as np
import torch.nn as nn
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from datasets import is_image_dataset, load_hand_posture_data, load_pendigits_data, get_dataset_name_for_loading_function, load_mnist_data, load_cifar10_data
from model_training import train_model, get_accuracy_score, get_nb_model_parameters, get_loss
from standardmodels import StandardTHLFFNN, StandardTHLConvNet
from semiseparablemodels import TwoHiddenLayeredSSNet, SSSTHLConvNet
from rank1models import Rank1THLFFNN, Rank1THLConvNet

def get_loss_and_accuracy(model: nn.Module, X: np.ndarray, y_true: np.ndarray):
    y_pred_t = model(torch.tensor(X).float())
    y_pred = y_pred_t.detach().numpy()

    mse = get_loss(y_true=torch.tensor(y_true).float(), y_pred=y_pred_t)
    accuracy = get_accuracy_score(y_true, y_pred)

    return mse, accuracy

def get_nb_model_parameters(model: nn.Module, count_gradientless_parameters=True):
    nb_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad or count_gradientless_parameters:
            if isinstance(param, torch.sparse.FloatTensor):
                nb_parameters += param._nnz()
            else:
                nb_parameters += param.numel()
    return nb_parameters

# Hyperparamters
dataset_loading_functions = [load_pendigits_data]
nb_iterations = 5
nb_neurons_list = [50]
training_set_shares = [0.1]
patience = 20
min_accuracy_improvement = 5e-4
batch_size = 100000
# ---

dataset_names = [get_dataset_name_for_loading_function(fun) for fun in dataset_loading_functions]

results = dict()
results["dataset_names"] = dataset_names
results["nb_iterations"] = nb_iterations
results["nb_neurons_list"] = nb_neurons_list
results["batch_size"] = batch_size
results["min_accuracy_improvement"] = min_accuracy_improvement
results["patience"] = patience
results["training_set_shares"] = training_set_shares

for dataset_loading_function, dataset_name in zip(dataset_loading_functions, dataset_names):
    print("Starting computations for " + dataset_name)

    (X_train_whole, y_train_whole), (X_test, y_test) = dataset_loading_function()
    results[dataset_name] = dict()
    input_size = X_train_whole.shape[1]
    output_size = np.max(y_train_whole) + 1

    # NOTE: I shuffle before splitting so that in larger splits the smaller splits are contained
    X_train_whole, y_train_whole = shuffle(X_train_whole, y_train_whole)

    for training_set_share in training_set_shares:
        print("Starting Share " + str(training_set_share))
        results[dataset_name][training_set_share] = dict()
        if training_set_share < 1:
            X_train, _, y_train, _ = train_test_split(X_train_whole, y_train_whole, test_size=(1.0-training_set_share), shuffle=False)
        else:
            X_train = X_train_whole.copy()
            y_train = y_train_whole.copy()

        start_time = time.time()

        # Standard NN
        results[dataset_name][training_set_share]["standard_nn"] = dict()
        for neuron_nb in nb_neurons_list:
            results[dataset_name][training_set_share]["standard_nn"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = StandardTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb)
                else:
                    model = StandardTHLFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb)

                # Only used for debugging
                # print("Layer Share of the Parameters: " + str(neuron_nb * neuron_nb / get_nb_model_parameters(model)))

                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["standard_nn"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"standard_nn done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # SSS 1 Dim
        results[dataset_name][training_set_share]["ss_1_dim"] = dict()
        for neuron_nb in nb_neurons_list:
            print(f"Starting ss_1_dim with {neuron_nb} neurons")
            results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = SSSTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=1)
                else:
                    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=1)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["ss_1_dim"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"ss_1_dim done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # SSS 2 Dim
        results[dataset_name][training_set_share]["ss_2_dim"] = dict()
        for neuron_nb in nb_neurons_list:
            print(f"Starting ss_2_dim with {neuron_nb} neurons")
            results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = SSSTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=2)
                else:
                    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=2)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["ss_2_dim"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"ss_2_dim done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # SSS 3 Dim
        results[dataset_name][training_set_share]["ss_3_dim"] = dict()
        for neuron_nb in nb_neurons_list:
            print(f"Starting ss_3_dim with {neuron_nb} neurons")
            results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = SSSTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=3)
                else:
                    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=3)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["ss_3_dim"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"ss_3_dim done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # SSS 4 Dim
        results[dataset_name][training_set_share]["ss_4_dim"] = dict()
        for neuron_nb in nb_neurons_list:
            print(f"Starting ss_4_dim with {neuron_nb} neurons")
            results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = SSSTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=4)
                else:
                    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=4)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["ss_4_dim"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"ss_4_dim done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # SSS 5 Dim
        results[dataset_name][training_set_share]["ss_5_dim"] = dict()
        for neuron_nb in nb_neurons_list:
            print(f"Starting ss_5_dim with {neuron_nb} neurons")
            results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = SSSTHLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=5)
                else:
                    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=neuron_nb, statespace_dim=5)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["ss_5_dim"][neuron_nb][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"ss_5_dim done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()

        # rank_1
        results[dataset_name][training_set_share]["rank_1"] = dict()
        for nb_neurons in nb_neurons_list:
            print(f"Starting rank_1 with {nb_neurons} neurons")
            results[dataset_name][training_set_share]["rank_1"][nb_neurons] = dict()
            for iter in range(nb_iterations):
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter] = dict()
                if is_image_dataset(dataset_loading_function):
                    model = Rank1THLConvNet(input_shape=X_train[0].shape, output_size=output_size, hidden_layer_size=nb_neurons)
                else:
                    model = Rank1THLFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=nb_neurons)
                training_loss_history, accuracy_history, val_accuracy_history = train_model(model, X_train_np=X_train, Y_train_np=y_train, batch_size=batch_size, patience=patience, min_accuracy_improvement=min_accuracy_improvement)
                test_loss, test_accuracy = get_loss_and_accuracy(model, X_test, y_test)
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["training_loss_history"] = training_loss_history
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["accuracy_history"] = accuracy_history
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["val_accuracy_history"] = val_accuracy_history
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["test_loss"] = test_loss
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["test_accuracy"] = test_accuracy
                results[dataset_name][training_set_share]["rank_1"][nb_neurons][iter]["nb_model_parameters"] = get_nb_model_parameters(model)

        print(f"rank_1 done ({time.time()-start_time} seconds)")
        pickle.dump(results, open("benchmark_results.p", "wb"))
        start_time = time.time()