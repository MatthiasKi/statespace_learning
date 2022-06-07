import torch
import torch.nn as nn
import numpy as np

from standardmodels import ConvFeatureExtractorBase
from semiseparablemodels import get_random_glorot_uniform_matrix

class Rank1THLFFNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int):
        super(Rank1THLFFNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.input_layer_activation = nn.ReLU()
        
        self.hidden_layer_vec1 = nn.parameter.Parameter(torch.tensor(get_random_glorot_uniform_matrix((hidden_layer_size, 1))).float(), requires_grad=True)
        self.hidden_layer_vec2 = nn.parameter.Parameter(torch.tensor(get_random_glorot_uniform_matrix((1, hidden_layer_size))).float(), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.tensor(get_random_glorot_uniform_matrix((hidden_layer_size,))).float(), requires_grad=True)

        self.hidden_layer_activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.output_layer_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.input_layer_activation(out)
        out = torch.matmul(out, self.hidden_layer_vec1)
        out = torch.matmul(out, self.hidden_layer_vec2)
        out = out + self.bias
        out = self.hidden_layer_activation(out)
        out = self.output_layer(out)
        out = self.output_layer_activation(out)
        return out

class Rank1THLConvNet(nn.Module):
    def __init__(self, input_shape: tuple, output_size: int, hidden_layer_size: int):
        super(Rank1THLConvNet, self).__init__()

        assert len(input_shape) == 3, "Expecting the input shape passed to StandardTHLConvNet to not contain the batch dimension"
        assert input_shape[0] <= 3, "The channels must be in the first dimension [C, H, W]"

        sample_input_shape = [1,] + [tmp for tmp in input_shape]
        sample_input = torch.randn(size=sample_input_shape)

        self.feature_extractor = ConvFeatureExtractorBase(sample_input=sample_input)
        self.classifier = Rank1THLFFNN(input_size=self.feature_extractor.out_dim, output_size=output_size, hidden_layer_size=hidden_layer_size)
    
    def forward(self, x):
        res = self.feature_extractor(x)
        res = self.classifier(res)
        return res


if __name__ == "__main__":
    input_size = 10
    output_size = 4
    hidden_layer_size = 50

    model = Rank1THLFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size)

    test_inputs = np.random.uniform(-1, 1, size=(30, input_size))
    res = model(torch.tensor(test_inputs).float())

    halt = 1