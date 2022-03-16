import torch

class FourierConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes, bias = True, periodic = False):
        super(FourierConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weights = torch.nn.Parameter((1 / (in_channels * out_channels)) * torch.complex(torch.rand(in_channels, out_channels, self.modes),torch.rand(in_channels, out_channels, self.modes)))
        self.biases = torch.nn.Parameter((1 / out_channels) * torch.complex(torch.rand(out_channels, self.modes),torch.rand(out_channels, self.modes)))
        self.bias = bias
        self.periodic = periodic

    def forward(self, x):
        batchsize = x.shape[0]
        if not self.periodic:
          padding = x.shape[2]
          x = torch.nn.functional.pad(x, [0,padding]) 

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box",x_ft[:, :, :self.modes], self.weights)
        if self.bias:
          out_ft[:, :, :self.modes] += self.biases
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        if not self.periodic:
          x = x[..., :-padding]
        return x


