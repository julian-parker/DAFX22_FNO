import torch
from .layers import FourierConv1d, FourierConv2d

class FNO_RNN_1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, width, depth = 4, activation = torch.nn.ReLU()):
        super(FNO_RNN_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.spatial_size = spatial_size 

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv1d(self.width, self.width, spatial_size ))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv1d(self.width, self.width, 1))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        x = self.in_mapping(x)
        output = torch.zeros(x.shape[0], num_time_steps,self.spatial_size,self.out_channels).to(x.device)
        for i in range(num_time_steps):
          x = self.cell(x)
          output[:,i,:,:] = self.out_mapping(x)

        return output
    def cell(self, x):
        x_out = x.permute(0, 2, 1)
        for i in range(self.depth):
          x1 = self.fourier_conv_layers[i](x_out)
          x2 = self.w[i](x_out)
          x_out = self.activation(x1) + x2
        return x_out.permute(0,2,1)

class FNO_RNN_1d_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, width, depth = 4, block_size = 16, activation = torch.nn.ReLU()):
        super(FNO_RNN_1d_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth 
        self.block_size = block_size

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv2d(self.width, self.width, block_size, spatial_size ))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv2d(self.width, self.width, (1,1)))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        num_blocks = num_time_steps // self.block_size
        x = self.in_mapping(x)
        output = torch.zeros(x.shape[0], num_time_steps,x.shape[2],self.out_channels).to(x.device)
        for i in range(num_blocks):
          x = self.cell(x)
          output[:,(i*self.block_size):((i+1)*self.block_size),:,:] = self.out_mapping(x)

        return output
    def cell(self, x):
        x_out = x.permute(0, 3, 1, 2)
        for i in range(self.depth):
          x1 = self.fourier_conv_layers[i](x_out)
          x2 = self.w[i](x_out)
          x_out = self.activation(x1) + x2
        return x_out.permute(0, 2, 3, 1)

class FNO_RNN_2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size_x, spatial_size_y, width, depth = 4, activation = torch.nn.ReLU()):
        super(FNO_RNN_2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.spatial_size_x = spatial_size_x
        self.spatial_size_y = spatial_size_y 

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv2d(self.width, self.width, spatial_size_x, spatial_size_y ))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv2d(self.width, self.width, (1, 1)))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x, num_time_steps):
        x = self.in_mapping(x)
        output = torch.zeros(x.shape[0], num_time_steps,self.spatial_size_x, self.spatial_size_y,self.out_channels).to(x.device)
        for i in range(num_time_steps):
          x = self.cell(x)
          output[:,i,:,:,:] = self.out_mapping(x)

        return output
    def cell(self, x):
        x_out = x.permute(0, 3, 1, 2)
        for i in range(self.depth):
          x1 = self.fourier_conv_layers[i](x_out)
          x2 = self.w[i](x_out)
          x_out = self.activation(x1) + x2
        return x_out.permute(0,2,3,1)