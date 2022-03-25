import torch
from .layers import FourierConv2d, FourierConv3d

class FNO_FF_1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size, num_time_steps, width, depth = 4, activation = torch.nn.ReLU()):
        super(FNO_FF_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth 
        self.spatial_size = spatial_size
        self.num_time_steps = num_time_steps

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv2d(self.width, self.width, self.num_time_steps, self.spatial_size))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv2d(self.width, self.width, (1,1)))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x):
        x = self.in_mapping(x)
        o = torch.zeros(x.shape[0], self.num_time_steps,self.spatial_size,self.width).to(x.device)
        o[:,0,:,:] = x.squeeze(1)
        o = o.permute(0, 3, 1, 2)
        for i in range(self.depth):
          o1 = self.fourier_conv_layers[i](o)
          o2 = self.w[i](o)
          o = self.activation(o1) + o2
        o = o.permute(0, 2, 3, 1)
        o = self.out_mapping(o)
        return o

class FNO_FF_2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size_x, spatial_size_y, num_time_steps, width, depth = 4, activation = torch.nn.ReLU()):
        super(FNO_FF_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth 
        self.spatial_size_x = spatial_size_x
        self.spatial_size_y = spatial_size_y
        self.num_time_steps = num_time_steps

        self.in_mapping = torch.nn.Linear(in_channels, self.width)

        self.fourier_conv_layers = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.fourier_conv_layers.append(FourierConv3d(self.width, self.width, self.num_time_steps, self.spatial_size_x, self.spatial_size_y))

        self.w = torch.nn.ModuleList()
        for _ in range(self.depth):
          self.w.append(torch.nn.Conv3d(self.width, self.width, (1,1,1)))

        self.activation = activation

        self.out_mapping = torch.nn.Linear(self.width, out_channels)

    def forward(self, x):
        x = self.in_mapping(x)
        o = torch.zeros(x.shape[0], self.num_time_steps,self.spatial_size_x,self.spatial_size_y, self.width).to(x.device)
        o[:,0,:,:,:] = x.squeeze(1)
        o = o.permute(0, 4, 1, 2, 3)
        for i in range(self.depth):
          o1 = self.fourier_conv_layers[i](o)
          o2 = self.w[i](o)
          o = self.activation(o1) + o2
        o = o.permute(0, 2, 3, 4, 1)
        o = self.out_mapping(o)
        return o