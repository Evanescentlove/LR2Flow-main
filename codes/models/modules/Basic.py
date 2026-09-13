import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import math
import random

from models.modules.ActNorms import ActNorm2d
from . import thops

from utils.util import opt_get
import models.modules.module_util as mutil


class Conv2d(nn.Conv2d):
    pad_dict = {
        "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
        "valid": lambda kernel, stride: [0 for _ in kernel]
    }

    @staticmethod
    def get_padding(padding, kernel_size, stride):
        # make paddding
        if isinstance(padding, str):
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            padding = padding.lower()
            try:
                padding = Conv2d.pad_dict[padding](kernel_size, stride)
            except KeyError:
                raise ValueError("{} is not supported".format(padding))
        return padding

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1], padding="same", groups=1,
                 do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, groups=groups, bias=(not do_actnorm))
        # init weight with std
        self.weight.data.normal_(mean=0.0, std=weight_std)
        if not do_actnorm:
            self.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)
        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x

# Zero initialization. We initialize the last convolution of each NN() with zeros, such that each affine
# coupling layer initially performs an identity function; we found that this helps training very deep networks.
class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=3):
        padding = Conv2d.get_padding(padding, kernel_size, stride)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * self.logscale_factor)


class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x): # logs: log(sigma)
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        if mean is None and logs is None:
            return -0.5 * (x ** 2 + GaussianDiag.Log2PI)
        else:
            return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])

    @staticmethod
    def sample(mean, logs, eps_std=None):
        """
        sample eps from a zero mean normal distribution, then apply scale and shift transform
        Args:
            mean: mean of return value
            logs: log scale of return value
            eps_std: standard variance of sampled normal distribution
        """
        # eps_std = eps_std or 1 # may cause problem when eps_std is 0
        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs) * eps_std)
        return mean + torch.exp(logs) * eps

    @staticmethod
    def sample_eps(shape, eps_std, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.normal(mean=torch.zeros(shape),
                           std=torch.ones(shape) * eps_std)
        return eps


class LaplaceDiag:
    Log2= float(np.log(2))

    @staticmethod
    def likelihood(mean, logs, x): # logs: log(sigma)
        if mean is None and logs is None:
            return  - (torch.abs(x) +  LaplaceDiag.Log2)
        else:
            return - (logs + (torch.abs(x - mean)) / torch.exp(logs) + LaplaceDiag.Log2)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = LaplaceDiag.likelihood(mean, logs, x)
        return thops.sum(likelihood, dim=[1, 2, 3])


def squeeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = unsqueeze2d(input, self.factor)
            return output, logdet

class UnSqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = unsqueeze2d(input, self.factor)
            return output, logdet
        else:
            output = squeeze2d(input, self.factor)
            return output, logdet

class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = torch.sigmoid(input)
            logdet += -thops.sum(F.softplus(input)+F.softplus(-input), dim=[1, 2, 3])
            return output, logdet
        else:
            output = -torch.log(torch.reciprocal(input) - 1.)
            logdet += -thops.sum(torch.log(input) + torch.log(1.-input), dim=[1, 2, 3])
            return output, logdet

# used in SRFlow
class Split2d_conditional(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = logs_eps
        self.position = position
        self.opt = opt

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, y_onehot=None):
        if not reverse:
            # self.input = input
            z1, z2 = self.split_ratio(input)
            mean, logs = self.split2d_prior(z1, ft)

            eps = (z2 - mean) / self.exp_eps(logs)

            logdet = logdet + self.get_logdet(logs, mean, z2)

            # print(logs.shape, mean.shape, z2.shape)
            # self.eps = eps
            # print('split, enc eps:', eps)
            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None:
                #print("WARNING: eps is None, generating eps untested functionality!")
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)

            eps = eps.to(mean.device)
            z2 = mean + self.exp_eps(logs) * eps

            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(logs, mean, z2)

            return z, logdet
            # return z, logdet, eps

    def get_logdet(self, logs, mean, z2):
        logdet_diff = GaussianDiag.logp(mean, logs, z2)
        # print("Split2D: logdet diff", logdet_diff.item())
        return logdet_diff

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2


''' Not used anymore '''
class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            z1, z2 = thops.split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = GaussianDiag.logp(mean, logs, z2) + logdet
            return z1, logdet
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = thops.cat_feature(z1, z2)
            return z, logdet

class Split2d_LR(nn.Module):
    def __init__(self, num_channels, num_channels_split):
        super().__init__()
        self.num_channels_split = num_channels_split
        self.conv = Conv2dZeros(num_channels_split, (num_channels-num_channels_split)*2)

    def split2d_prior(self, z):
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def forward(self, input, eps_std=None, logdet=0., reverse=False):
        if not reverse:
            z1, z2 = input[:, :self.num_channels_split, ...], input[:, self.num_channels_split:, ...]
            mean, logs = self.split2d_prior(z1)
            logdet += GaussianDiag.logp(mean, logs, z2)
            return z1, logdet
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet

# DenseBlock for affine coupling (flow)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gc=32, bias=True, init='xavier', for_flow=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channels + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_channels + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_channels + 4 * gc, out_channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # init as 'xavier', following the practice in https://github.com/VLL-HD/FrEIA/blob/c5fe1af0de8ce9122b5b61924ad75a19b9dc2473/README.rst#useful-tips--engineering-heuristics
        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

        # initialiize input to all zeros to have zero mean and unit variance
        if for_flow:
            mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


# ResidualDenseBlock for multi-layer feature extraction
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, init='xavier'):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # init as 'xavier', following the practice in https://github.com/VLL-HD/FrEIA/blob/c5fe1af0de8ce9122b5b61924ad75a19b9dc2473/README.rst#useful-tips--engineering-heuristics
        if init == 'xavier':
            # mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
            mutil.initialize_weights_xavier(self.conv5, 0)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # residual scaling are helpful

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x # residual scaling are helpful

class RDN(nn.Module):
    '''composed of rrdb blocks'''

    def __init__(self, in_channels, out_channels, nb=3, nf=64, gc=32, init='xavier', for_flow=True):
        super(RDN, self).__init__()

        RRDB_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv_first, self.trunk_conv, self.conv_last], 0.1)
        else:
            mutil.initialize_weights([self.conv_first, self.trunk_conv, self.conv_last], 0.1)

        if for_flow:
            mutil.initialize_weights(self.conv_last, 0)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.trunk_conv(self.RRDB_trunk(x)) + x
        return self.conv_last(x)


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, init='xavier', for_flow=True):
        super(FCN, self).__init__()
        self.conv1 = Conv2d(in_channels, hidden_channels, kernel_size=[3, 3], stride=[1, 1])
        self.conv2 = Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden])
        self.conv3 = Conv2dZeros(hidden_channels, out_channels, kernel_size=[3, 3], stride=[1, 1])
        self.relu = nn.ReLU(inplace=False)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

        if for_flow:
            mutil.initialize_weights(self.conv3, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class HaarDownsampling(nn.Module):
    def __init__(self, in_channels):
        super(HaarDownsampling, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out, logdet
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.in_channels), logdet


class WaveletTightFrame(nn.Module):
    def __init__(self, in_channels=3):
        """
        Args:
            in_channels: image (low-frequencies) channels
        """
        super(WaveletTightFrame, self).__init__()
        self.in_channels = in_channels
        self.kernel_1d = torch.tensor([
            [1 / 4, 1 / 2, 1 / 4],
            [np.sqrt(2) / 4, 0, -np.sqrt(2) / 4],
            [-1 / 4, 1 / 2, -1 / 4]
        ], dtype=torch.float32)
        self.kernel_2d = torch.zeros(9, 1, 3, 3)
        num = 0
        for x in self.kernel_1d:
            for y in self.kernel_1d:
                self.kernel_2d[num, 0, ...] = torch.mm(x.reshape((3, 1)), y.reshape((1, 3)))
                num += 1

        self.kernel_2d = torch.cat([self.kernel_2d] * self.in_channels, dim=0)
        self.rekernel_2d = self.kernel_2d.flip([2, 3])  # lr-flip and ud-flip

        self.kernel_2d = nn.Parameter(self.kernel_2d)  # add kernel_2d into module's parameter list
        self.kernel_2d.requires_grad = False  # un-trainable

        self.rekernel_2d = nn.Parameter(self.rekernel_2d)
        self.rekernel_2d.requires_grad = False

    def mirror_padding(self, x, d, mode=1):
        """
        Args:
            x: input that to be extended
            d: padding length in each dim (int)
        """
        assert isinstance(d, int)
        x_ind = [3, 4, 5] if mode == 2 else []  # channels for x-axis to be asymmetric
        y_ind = [1, 4, 7] if mode == 2 else []  # channels for y-axis to be asymmetric
        input_shape = x.shape
        x_size, y_size = input_shape[-2], input_shape[-1]
        out = F.pad(x, (d, d, d, d), 'reflect')  # symmetric extension
        if x_ind is not None or y_ind is not None:
            for i in range(input_shape[-3]):
                if i % 9 in x_ind:
                    out[..., i, :d, :] = -out[..., i, :d, :]
                    out[..., i, d + x_size:, :] = -out[..., i, d + x_size:, :]
                if i % 9 in y_ind:
                    out[..., i, :, :d] = -out[..., i, :, :d]
                    out[..., i, :, d + y_size:] = -out[..., i, :, d + y_size:]

        return out

    def squeeze(self, x):
        """
        Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return x

    def undo_squeeze(self, x):
        """
        unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, H * 2, W * 2)
        return x

    def upsample(self, raw_tensor):
        """
        pad 0 inside raw_tensor in 'cross' manner
        """
        b, c, h, w = raw_tensor.shape
        out = torch.zeros((b, c, h * 2, w * 2)).to(raw_tensor.device)
        out[..., 0::2, 0::2] = raw_tensor[..., 0::1, 0::1]
        return out

    def forward(self, x, reverse=False):
        """ One level wavelet decomposition or reconstruction. """
        assert len(x.shape) == 4
        if not reverse:
            # down-sampling
            # decompose low-frequencies and squeeze high-frequencies
            b, _, h, w = x.shape
            low, high = x[:, :self.in_channels, ...], x[:, self.in_channels:, ...]

            # squeeze high-frequencies
            high_decomp = self.squeeze(high)

            # wavelet decompose low-frequencies
            low_decomp = F.conv2d(input=F.pad(low, (1, 1, 1, 1), 'reflect'), weight=(2 * self.kernel_2d).to(x.device),
                                  stride=2, groups=self.in_channels)
            low_decomp = low_decomp.reshape([b, self.in_channels, 9, h // 2, w // 2])
            low_decomp = torch.transpose(low_decomp, 1, 2)
            low_decomp = low_decomp.reshape([b, 9 * self.in_channels, h // 2, w // 2])

            # concatenate decomposed low-frequencies and high-frequencies
            out = torch.cat((low_decomp, high_decomp), dim=1)
            return out, None

        else:
            x, rest = x[:, :9 * self.in_channels, ...], x[:, 9 * self.in_channels:, ...]
            recon_rest = self.undo_squeeze(rest)

            b, _, h, w = x.shape
            x = x.view(b, 9, self.in_channels, h, w)
            x = torch.transpose(x, 1, 2)
            x = x.reshape([b, 9 * self.in_channels, h, w])
            x = self.upsample(x)  # zero padding in 'cross' manner
            recon_low = F.conv2d(input=self.mirror_padding(x, 1, mode=2), weight=(2 * self.rekernel_2d).to(x.device), stride=1, groups=9 * self.in_channels)
            recon_low = recon_low.view(b, self.in_channels, 9, h * 2, w * 2).sum(dim=2)

            # concatenate reconstruct low-frequencies and rest components
            out = torch.cat((recon_low, recon_rest), dim=1)
            return out, None


class HardThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        output = input.clone()
        output[output.abs() < threshold] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        # only keep the gradients of the elements that were not set to zero in the forward pass
        grad_input = grad_output.clone()
        grad_input[input.abs() < threshold] = 0
        return grad_input, None


def hard_threshold(x, lam):
    return HardThreshold.apply(x, lam)


def soft_threshold(x, lam):
    return torch.sign(x) * torch.maximum(torch.abs(x) - lam, torch.tensor(0.0))


def thresholding(x, lam, option):
    if option == 's':
        return soft_threshold(x, lam)
    elif option == 'h':
        return hard_threshold(x, lam)
    else:
        raise NotImplementedError



