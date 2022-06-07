import torch
import torch.nn as nn
import numpy as np
import scipy.linalg

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

from standardmodels import ConvFeatureExtractorBase

class SemiseparableLayer(nn.Module):
    def __init__(self, input_size, statespace_dim=10, inputoutput_dim=1):
        super(SemiseparableLayer, self).__init__()

        self.input_size = input_size
        self.statespace_dim = statespace_dim
        self.inputoutput_dim = inputoutput_dim

        assert np.mod(input_size, inputoutput_dim) == 0, "Input Size must be dividable by the Input / Output size"
        assert self.statespace_dim > 0, "State Space Dimension must be positive."
        assert self.inputoutput_dim > 0, "Input / Output Dimension must be positive"
        assert self.input_size > 0, "Number of input features must be > 0."

        self.nb_states = int(self.input_size / self.inputoutput_dim)
        self.init_state_matrices()

        self.bias = nn.parameter.Parameter(torch.tensor(get_random_glorot_uniform_matrix((input_size,))).float(), requires_grad=True)

        self.last_forward_causal_states = [] # Only used for debugging
        self.last_forward_anticausal_states = [] # Only used for debugging

    def init_state_matrices(self):
        T_full = get_random_glorot_uniform_matrix(shape=(self.input_size, self.input_size))
        dims_in = self.inputoutput_dim * np.ones((self.nb_states,), dtype='int32')
        dims_out = self.inputoutput_dim * np.ones((self.nb_states,), dtype='int32')
        T_operator = ToeplitzOperator(T_full, dims_in, dims_out)
        S = SystemIdentificationSVD(toeplitz=T_operator, max_states_local=self.statespace_dim)
        system_approx = MixedSystem(S)

        self.initial_T = system_approx.to_matrix()

        A = [stage.A_matrix for stage in system_approx.causal_system.stages]
        B = [stage.B_matrix for stage in system_approx.causal_system.stages]
        C = [stage.C_matrix for stage in system_approx.causal_system.stages]
        D = [stage.D_matrix for stage in system_approx.causal_system.stages]
        E = [stage.A_matrix for stage in system_approx.anticausal_system.stages]
        F = [stage.B_matrix for stage in system_approx.anticausal_system.stages]
        G = [stage.C_matrix for stage in system_approx.anticausal_system.stages]
        

        self.A = nn.ParameterList([nn.Parameter(torch.tensor(A_k).float(), requires_grad=True) for A_k in A])
        self.B = nn.ParameterList([nn.Parameter(torch.tensor(B_k).float(), requires_grad=True) for B_k in B])
        self.C = nn.ParameterList([nn.Parameter(torch.tensor(C_k).float(), requires_grad=True) for C_k in C])
        self.D = nn.ParameterList([nn.Parameter(torch.tensor(D_k).float(), requires_grad=True) for D_k in D])
        self.E = nn.ParameterList([nn.Parameter(torch.tensor(E_k).float(), requires_grad=True) for E_k in E])
        self.F = nn.ParameterList([nn.Parameter(torch.tensor(F_k).float(), requires_grad=True) for F_k in F])
        self.G = nn.ParameterList([nn.Parameter(torch.tensor(G_k).float(), requires_grad=True) for G_k in G])

    def get_index_range_according_to_state(self, state_i):
        return range(state_i*self.inputoutput_dim,(state_i+1)*self.inputoutput_dim)

    def forward(self, U):
        causal_state = torch.zeros((0, U.shape[0]))
        anticausal_state = torch.zeros((0, U.shape[0]))
        self.last_forward_causal_states.append(causal_state)
        self.last_forward_anticausal_states.append(anticausal_state)

        y_pred = torch.zeros((U.shape[1], U.shape[0]))
        for causal_state_i in range(self.nb_states):
            causal_state_index_range = self.get_index_range_according_to_state(causal_state_i)
            u_causal = torch.transpose(U, 1, 0)[causal_state_index_range, :]
            y_pred[causal_state_index_range, :] += torch.matmul(self.C[causal_state_i], causal_state) + torch.matmul(self.D[causal_state_i], u_causal)
            causal_state = torch.matmul(self.A[causal_state_i], causal_state) + torch.matmul(self.B[causal_state_i], u_causal)

            anticausal_state_i = self.nb_states-1-causal_state_i
            anti_causal_index_range = self.get_index_range_according_to_state(anticausal_state_i)
            u_anticausal = torch.transpose(U, 1, 0)[anti_causal_index_range, :]
            y_pred[anti_causal_index_range, :] += torch.matmul(self.G[anticausal_state_i], anticausal_state)
            anticausal_state = torch.matmul(self.E[anticausal_state_i], anticausal_state) + torch.matmul(self.F[anticausal_state_i], u_anticausal)
        
            self.last_forward_causal_states.append(causal_state)
            self.last_forward_anticausal_states.append(anticausal_state)

        y_pred = torch.transpose(y_pred, 1, 0)
        y_pred += self.bias
        return y_pred

class TwoHiddenLayeredSSNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int, statespace_dim=1, input_output_dim=1):
        super(TwoHiddenLayeredSSNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = SemiseparableLayer(hidden_layer_size, statespace_dim=statespace_dim, inputoutput_dim=input_output_dim)
        self.hidden_layer_activation = nn.ReLU()

        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.output_layer_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.input_layer_activation(out)
        out = self.hidden_layer(out)
        out = self.hidden_layer_activation(out)
        out = self.output_layer(out)
        out = self.output_layer_activation(out)
        return out

class SSSTHLConvNet(nn.Module):
    def __init__(self, input_shape: tuple, output_size: int, hidden_layer_size: int, statespace_dim=1):
        super(SSSTHLConvNet, self).__init__()

        assert len(input_shape) == 3, "Expecting the input shape passed to StandardTHLConvNet to not contain the batch dimension"
        assert input_shape[0] <= 3, "The channels must be in the first dimension [C, H, W]"

        sample_input_shape = [1,] + [tmp for tmp in input_shape]
        sample_input = torch.randn(size=sample_input_shape)

        self.feature_extractor = ConvFeatureExtractorBase(sample_input=sample_input)
        self.classifier = TwoHiddenLayeredSSNet(input_size=self.feature_extractor.out_dim, output_size=output_size, hidden_layer_size=hidden_layer_size, statespace_dim=statespace_dim)
    
    def forward(self, x):
        res = self.feature_extractor(x)
        res = self.classifier(res)
        return res

def get_random_glorot_uniform_matrix(shape: tuple):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, size=shape)

def unit_test_ss_layer_with_random_T():
    nb_samples = 51
    nb_hidden_neurons = 100
    statespace_dim = 3
    inputputoutput_dim = 10

    random_input = np.random.uniform(-1,1,size=(nb_samples, nb_hidden_neurons))

    layer = SemiseparableLayer(input_size=nb_hidden_neurons, statespace_dim=statespace_dim, inputoutput_dim=inputputoutput_dim)
    layer_output = layer(torch.tensor(random_input).float()).detach().numpy()

    T = layer.initial_T
    system_output = random_input @ T.T

    assert np.allclose(layer_output, system_output, atol=1e-5), "The layer output does not match the system output"

if __name__ == "__main__":
    #unit_test_ss_layer_with_random_T()
    
    input_size = 10
    output_size = 4
    hidden_layer_size = 50

    model = TwoHiddenLayeredSSNet(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size)

    test_inputs = np.random.uniform(-1, 1, size=(30, input_size))
    res = model(torch.tensor(test_inputs).float())

    halt = 1