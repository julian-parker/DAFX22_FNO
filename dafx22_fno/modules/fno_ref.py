import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .layers import FourierConv1d, FourierConv2d

"""
Code adapted from https://github.com/zongyi-li/fourier_neural_operator/
"""

class FNO_Markov_1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, width, depth = 4):
        super(FNO_Markov_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.spatial_size = spatial_size 

        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels + 1, self.width) # input channel is 2: (a(x), x)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(SpectralConv1d(self.width, self.width, modes = 20))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv1d(self.width, self.width, 1))

    def forward(self, x, num_time_steps):
        output = torch.zeros(x.shape[0], num_time_steps,self.spatial_size,self.out_channels).to(x.device)
        for i in range(num_time_steps):
          x = self.cell(x)
          output[:,i,:,:] = x
        return output
    def cell(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        for i in range(self.depth):
            x1 = self.fourier_conv_layers[i](x)
            x2 = self.w[i](x)
            x = x1 + x2
            x = F.gelu(x)
        x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class FNO_Markov_2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size_x, spatial_size_y, width, depth = 4):
        super(FNO_Markov_2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.spatial_size_x = spatial_size_x
        self.spatial_size_y = spatial_size_y 

        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channels + 2, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(SpectralConv2d(self.width, self.width, modes1 = 20, modes2 = 20))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv2d(self.width, self.width, (1, 1)))

    def forward(self, x, num_time_steps):
        output = torch.zeros(x.shape[0], num_time_steps,self.spatial_size_x, self.spatial_size_y,self.out_channels).to(x.device)
        for i in range(num_time_steps):
          x = self.cell(x)
          output[:,i,:,:,:] = x
        return output
    def cell(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])
        for i in range(self.depth):
            x1 = self.fourier_conv_layers[i](x)
            x2 = self.w[i](x)
            x = x1 + x2
            x = F.gelu(x)
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
