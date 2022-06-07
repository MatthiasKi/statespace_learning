import torch
import torch.nn as nn

class StandardSLFFNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int):
        super(StandardSLFFNN, self).__init__()
        self.model = StandardFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size, nb_hidden_layers=1)
    
    def forward(self, x):
        return self.model(x)

class StandardTHLFFNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int):
        super(StandardTHLFFNN, self).__init__()
        self.model = StandardFFNN(input_size=input_size, output_size=output_size, hidden_layer_size=hidden_layer_size, nb_hidden_layers=2)
    
    def forward(self, x):
        return self.model(x)

class StandardTHLConvNet(nn.Module):
    def __init__(self, input_shape: tuple, output_size: int, hidden_layer_size: int):
        super(StandardTHLConvNet, self).__init__()

        assert len(input_shape) == 3, "Expecting the input shape passed to StandardTHLConvNet to not contain the batch dimension"
        assert input_shape[0] <= 3, "The channels must be in the first dimension [C, H, W]"

        sample_input_shape = [1,] + [tmp for tmp in input_shape]
        sample_input = torch.randn(size=sample_input_shape)

        self.feature_extractor = ConvFeatureExtractorBase(sample_input=sample_input)
        self.classifier = StandardFFNN(input_size=self.feature_extractor.out_dim, output_size=output_size, hidden_layer_size=hidden_layer_size, nb_hidden_layers=2)
    
    def forward(self, x):
        res = self.feature_extractor(x)
        res = self.classifier(res)
        return res

class ConvFeatureExtractorBase(nn.Module):
    def __init__(self, sample_input: torch.tensor):
        super(ConvFeatureExtractorBase, self).__init__()

        nb_middle_channels = 5
        nb_out_channels = 10
        self.conv1 = nn.Conv2d(in_channels=sample_input.shape[1], out_channels=nb_middle_channels, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=nb_middle_channels, out_channels=nb_out_channels, kernel_size=5, stride=1)

        self.activation = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2)

        sample_forward_res = self.forward(sample_input)
        self.out_dim = sample_forward_res.shape[1]

    def forward(self, x):
        res = self.conv1(x)
        res = self.activation(res)
        res = self.pool(res)
        res = self.conv2(res)
        res = self.activation(res)
        res = self.pool(res)
        res = torch.flatten(res, start_dim=1)
        return res

class StandardFFNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_size: int, nb_hidden_layers: int):
        super(StandardFFNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.input_layer_activation = nn.ReLU()
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(nb_hidden_layers-1)])
        self.hidden_layers_activations = [nn.ReLU() for _ in range(nb_hidden_layers-1)]
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.output_layer_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.input_layer(x)
        out = self.input_layer_activation(out)
        for hidden_layer, hidden_layer_activation in zip(self.hidden_layers, self.hidden_layers_activations):
            out = hidden_layer(out)
            out = hidden_layer_activation(out)
        out = self.output_layer(out)
        out = self.output_layer_activation(out)
        return out

if __name__ == "__main__":
    sample_input = torch.randn(size=(1, 3, 32, 32))
    conv_base = ConvFeatureExtractorBase(sample_input=sample_input)
    out_shape = conv_base.out_dim

    standard_two_hidden_layer_conv_net = StandardTHLConvNet(input_shape=(3, 32, 32), output_size=4, hidden_layer_size=100)

    halt = 1