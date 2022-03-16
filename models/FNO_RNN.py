import torch
from .layers import FourierConv1d

class FNO_RNN_1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes, width, depth = 4, activation = torch.nn.ReLU()):
        super(FNO_RNN_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.depth = depth 

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv1d(self.width, self.width, self.modes))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv1d(self.width, self.width, 1))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        x = self.in_mapping(x)
        output = torch.zeros(x.shape[0], num_time_steps,x.shape[1],self.out_channels).to(x.device)
        for i in range(num_time_steps):
          x = self.cell(x)
          output[:,i,:,:] = self.out_mapping(x)

        return output
    def cell(self, x):
        x = x.permute(0, 2, 1)
        for i in range(self.depth):
          x1 = self.fourier_conv_layers[i](x)
          x2 = self.w[i](x)
          x = self.activation(x1) + x2
        x = x.permute(0,2,1)
        return x