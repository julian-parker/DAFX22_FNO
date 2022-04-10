import torch

class FourierConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, size, bias = True, periodic = False):
        super(FourierConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if not periodic:
            self.size = size
        else:
            self.size = size // 2

        self.weights = torch.nn.Parameter(torch.view_as_real((1 / (in_channels * out_channels)) * torch.randn(in_channels, out_channels, self.size, dtype=torch.cfloat)))
        self.biases = torch.nn.Parameter(torch.view_as_real((1 / out_channels) * torch.randn(out_channels, self.size, dtype=torch.cfloat)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        if not self.periodic:
          padding = self.size
          x = torch.nn.functional.pad(x, [0,padding]) 

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size] = torch.einsum("bix,iox->box",x_ft[:, :, :self.size],  torch.view_as_complex(self.weights))
        if self.bias:
          out_ft[:, :, :self.size] +=  torch.view_as_complex(self.biases)
        out = torch.fft.irfft(out_ft, n=x.size(-1))
        if not self.periodic:
          out = out[..., :-padding]
        return out

class FourierConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, size_x, size_y, bias = True, periodic = False):
        super(FourierConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if not periodic:
          self.size_x = size_x
          self.size_y = size_y
        else:
          self.size_x = size_x // 2
          self.size_y = size_y // 2

        self.weights = torch.nn.Parameter(torch.view_as_real((1 / (in_channels * out_channels)) * torch.randn(in_channels, out_channels, self.size_x, self.size_y, dtype=torch.cfloat)))
        self.biases = torch.nn.Parameter(torch.view_as_real((1 / out_channels) * torch.randn(out_channels, self.size_x, self.size_y,dtype=torch.cfloat)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        if not self.periodic:
          x = torch.nn.functional.pad(x, [0,self.size_y, 0, self.size_x]) 

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)

        out_ft[:, :, :self.size_x, :self.size_y] = torch.einsum("bixy,ioxy->boxy",x_ft[:, :, :self.size_x, :self.size_y],  torch.view_as_complex(self.weights))
        if self.bias:
          out_ft[:, :, :self.size_x, :self.size_y] +=  torch.view_as_complex(self.biases)
        out = torch.fft.irfft2(out_ft)
        if not self.periodic:
          out = out[..., :self.size_x, :self.size_y]
        return out

class FourierConv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, size_x, size_y, size_z, bias = True, periodic = False):
        super(FourierConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if not periodic:
          self.size_x = size_x
          self.size_y = size_y
          self.size_z = size_z
        else:
          self.size_x = size_x // 2
          self.size_y = size_y // 2
          self.size_z = size_z // 2

        self.weights = torch.nn.Parameter(torch.view_as_real((1 / (in_channels * out_channels)) * torch.randn(in_channels, out_channels, self.size_x, self.size_y, self.size_z, dtype = torch.cfloat)))
        self.biases = torch.nn.Parameter(torch.view_as_real((1 / out_channels) * torch.randn(out_channels, self.size_x, self.size_y, self.size_z, dtype = torch.cfloat)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        if not self.periodic:
          x = torch.nn.functional.pad(x, [0,self.size_z, 0, self.size_y, 0, self.size_x]) 

        x_ft = torch.fft.rfft3(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size_x, :self.size_y] = torch.einsum("bixyz,ioxyz->boxyz",x_ft[:, :, :self.size_x, :self.size_y, :self.size_z],  torch.view_as_complex(self.weights))
        if self.bias:
          out_ft[:, :, :self.size_x, :self.size_y, self.size_z] +=  torch.view_as_complex(self.biases)
        out = torch.fft.irfft3(out_ft)
        if not self.periodic:
          out = out[..., :self.size_x, :self.size_y, :self.size_z]
        return out
