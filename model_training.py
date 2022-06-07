import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Local imports required for unit testing
from standardmodels import StandardTHLFFNN
from semiseparablemodels import TwoHiddenLayeredSSNet

def get_nb_model_parameters(model: nn.Module, count_gradientless_parameters=True):
    nb_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad or count_gradientless_parameters:
            nb_parameters += param.numel()
    return nb_parameters

def get_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    assert len(y_true.shape) == 1, "True labels should be given as vector"
    assert len(y_pred.shape) == 2, "Predicted labels should be number-of-classes-dimensional"
    return accuracy_score(y_true.astype("int"), y_pred.argmax(axis=1).astype("int"))

def get_loss(y_true: torch.tensor, y_pred: torch.tensor):
    cross_entropy = nn.CrossEntropyLoss()
    return cross_entropy(y_pred, target=y_true.long()).detach().numpy()

def get_batch(X: np.ndarray, y: np.ndarray, batch_size: int, batch_i: int):
    X_batch = X[batch_i*batch_size:min((batch_i+1)*batch_size, X.shape[0])]
    y_batch = y[batch_i*batch_size:min((batch_i+1)*batch_size, y.shape[0])]

    X_batch_t, y_batch_t = map(torch.tensor, (X_batch, y_batch))
    X_batch_t = X_batch_t.float()
    y_batch_t = y_batch_t.long()
    return X_batch_t, y_batch_t

def train_model(model: nn.Module, X_train_np: np.array, Y_train_np: np.array, learning_rate=1e-3, batch_size=1000, use_Adam=True, min_accuracy_improvement=5e-4, patience=10, min_number_training_epochs=200):
    X = X_train_np.copy()
    Y = Y_train_np.copy()
    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(X, Y, test_size=0.15, shuffle=True)
    X_val_t = torch.tensor(X_val_np).float()

    if use_Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    training_loss_history = []
    accuracy_history = []
    val_accuracy_history = []
    nb_batches_per_epoch = np.ceil(X_train_np.shape[0] / batch_size).astype("int")

    continue_training = True
    while continue_training:
        X_train_shuffled, y_train_shuffled = shuffle(X_train_np, Y_train_np)

        for batch_i in range(nb_batches_per_epoch):
            X_batch_t, y_batch_t = get_batch(X_train_shuffled, y_train_shuffled, batch_size=batch_size, batch_i=batch_i)
            outputs_train = model(X_batch_t)
            loss_train = loss_function(outputs_train, target=y_batch_t)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        X_train_shuffled_t = torch.tensor(X_train_shuffled).float()
        y_train_shuffled_t = torch.tensor(y_train_shuffled).float()
        y_pred = model(X_train_shuffled_t)
        training_loss_history.append(get_loss(y_train_shuffled_t, y_pred))
        accuracy_history.append(get_accuracy_score(y_train_shuffled, y_pred.detach().numpy()))

        y_val_pred = model(X_val_t)
        val_accuracy_history.append(get_accuracy_score(Y_val_np, y_val_pred.detach().numpy()))

        if len(val_accuracy_history) > min_number_training_epochs \
            and len(val_accuracy_history) > 2*patience \
            and np.max(val_accuracy_history[-patience:]) < np.max(val_accuracy_history[:-patience]) + min_accuracy_improvement:
            continue_training = False
    
    return training_loss_history, accuracy_history, val_accuracy_history

if __name__ == "__main__":
    input_size = 10
    output_size = 4
    hidden_layer_size = 100

    standard_test_model = StandardTHLFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size)
    standard_expected_nb_parameters = input_size*hidden_layer_size+hidden_layer_size*hidden_layer_size+output_size*hidden_layer_size + 2*hidden_layer_size + output_size
    standard_calculated_nb_parameters = get_nb_model_parameters(standard_test_model)
    assert standard_calculated_nb_parameters == standard_expected_nb_parameters, "Calculated the wrong number of parameters for the standard model"

    ss_test_model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size, statespace_dim=1, input_output_dim=1)
    ss_expected_nb_parameters = input_size*hidden_layer_size+hidden_layer_size \
        + hidden_layer_size*output_size + output_size \
        + hidden_layer_size * 7
    ss_calculated_nb_parameters = get_nb_model_parameters(ss_test_model)
    assert ss_expected_nb_parameters == ss_calculated_nb_parameters, "Calculated the wrong number of parameters for the ss model"

    halt = 1