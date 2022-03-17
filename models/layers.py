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

        self.weights = torch.nn.Parameter((1 / (in_channels * out_channels)) * torch.complex(torch.rand(in_channels, out_channels, self.size),torch.rand(in_channels, out_channels, self.size)))
        self.biases = torch.nn.Parameter((1 / out_channels) * torch.complex(torch.rand(out_channels, self.size),torch.rand(out_channels, self.size)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        batchsize = x.shape[0]
        if not self.periodic:
          padding = x.shape[2]
          x = torch.nn.functional.pad(x, [0,padding]) 

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size] = torch.einsum("bix,iox->box",x_ft[:, :, :self.size], self.weights)
        if self.bias:
          out_ft[:, :, :self.size] += self.biases
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        if not self.periodic:
          x = x[..., :-padding]
        return x

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

        self.weights = torch.nn.Parameter((1 / (in_channels * out_channels)) * torch.complex(torch.rand(in_channels, out_channels, self.size_y, self.size_x),torch.rand(in_channels, out_channels, self.size_y, self.size_x)))
        self.biases = torch.nn.Parameter((1 / out_channels) * torch.complex(torch.rand(out_channels, self.size_y, self.size_x),torch.rand(out_channels, self.size_y, self.size_x)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        batchsize = x.shape[0]
        if not self.periodic:
          x = torch.nn.functional.pad(x, [0,self.size_x, 0, self.size_y]) 

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.size_y, :self.size_x] = torch.einsum("bixy,ioxy->boxy",x_ft[:, :, :self.size_y, :self.size_x], self.weights)
        if self.bias:
          out_ft[:, :, :self.size_y, :self.size_x] += self.biases
        x = torch.fft.irfft2(out_ft)
        if not self.periodic:
          x = x[..., :self.size_y, :self.size_x]
        return x

