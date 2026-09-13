import torch
import torch.nn as nn

from models.modules.Basic import WaveletTightFrame, RDN
from models.modules.FlowStep import FlowStep
from models.modules.ConditionalFlow import ConditionalFlow


class DenoisingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt['n_channels']
        self.hf_channels = self.n_channels * 8

        # rescaling network
        self.wavelet = WaveletTightFrame(self.n_channels)

        self.uncond = nn.ModuleList([
            FlowStep(in_channels=self.n_channels * 5,
                     flow_permutation='none',
                     flow_coupling='Affine3shift',
                     LRvsothers=True if k % 2 == 0 else False,
                     opt=opt)
            for k in range(opt['flowstep'])
        ])
        self.cond = ConditionalFlow(in_channels=self.hf_channels,
                                    cond_channels=self.n_channels * 2,
                                    n_flow_step=opt['condFlow']['flowstep'],
                                    opt=opt['condFlow'])

        # denoising network
        self.denoiser_lf = RDN(self.n_channels * 2 + 1, self.n_channels * 2, nf=128, gc=64)
        self.denoiser_hf = RDN(self.hf_channels + 1, self.hf_channels, nf=128, gc=64)

    def forward(self, img_H, img_L, noise_level):
        lf_H, hf_H = self.downscale(img_H)
        lf_L, hf_L = self.downscale(img_L)

        fake_H, fake_lf, fake_hf = self.denoise_and_upscale(lf_L, hf_L, noise_level)

        loss_hr = (fake_H - img_H).abs().mean()
        loss_lf = ((fake_lf - lf_H) ** 2).mean()
        loss_hf = ((fake_hf - hf_H) ** 2).mean()
        return loss_hr, loss_lf, loss_hf

    def denoise(self, img_L, noise_level):
        lf_L, hf_L = self.downscale(img_L)
        fake_H, _, _ = self.denoise_and_upscale(lf_L, hf_L, noise_level)
        return fake_H

    def downscale(self, hr):
        coeff, _ = self.wavelet(hr, reverse=False)
        coeff = self.stack(coeff, num_in=1)
        for layer in self.uncond:
            coeff, _ = layer(coeff, reverse=False)
        lf, hf = self.unstack(coeff, num_out=2)
        return lf, hf

    def denoise_and_upscale(self, fake_lf, fake_hf, noise_level):
        # denoise lf
        lf = self.denoiser_lf(self.cat_noise_level(fake_lf, noise_level))

        # denoise hf
        fake_z, _ = self.cond(fake_hf, u=lf, reverse=False)
        z = self.denoiser_hf(self.cat_noise_level(fake_z, noise_level))
        hf, _ = self.cond(z, u=lf, reverse=True)

        coeff = self.stack([lf, hf], num_in=2)
        for layer in reversed(self.uncond):
            coeff, _ = layer(coeff, reverse=True)
        coeff = self.unstack(coeff, num_out=1)
        hr, _ = self.wavelet(coeff, reverse=True)
        return hr.clamp(0, 1), lf, hf

    def stack(self, x, num_in):
        """
        num_in = 1: split [l, h1, h2] into [l, h1] and [l, h2]
        num_in = 2: split [l, h] into [l1, h1] and [l2, h2]
        """
        if num_in == 1:
            l, h1, h2 = torch.split(x, [self.n_channels, 4 * self.n_channels, 4 * self.n_channels], dim=1)
            x1 = torch.cat([l, h1], dim=1)
            x2 = torch.cat([l, h2], dim=1)
            return torch.cat([x1, x2], dim=0)  # cat along batch dim
        elif num_in == 2:
            assert isinstance(x, list) and len(x) == 2
            l, h = x
            l1, l2 = torch.split(l, [self.n_channels, self.n_channels], dim=1)
            h1, h2 = torch.split(h, [4 * self.n_channels, 4 * self.n_channels], dim=1)
            x1 = torch.cat([l1, h1], dim=1)
            x2 = torch.cat([l2, h2], dim=1)
            return torch.cat([x1, x2], dim=0)
        else:
            raise ValueError

    def unstack(self, x, num_out):
        """
        num_out = 1: cat [l1, h1] and [l2, h2] into [(l1 + l2) / 2, h1, h2]
        num_out = 2: cat [l1, h1] and [l2, h2] into [l1, l2] and [h1, h2]
        """
        x1, x2 = torch.chunk(x, 2, dim=0)  # split along batch dim
        l1, h1 = torch.split(x1, [self.n_channels, 4 * self.n_channels], dim=1)
        l2, h2 = torch.split(x2, [self.n_channels, 4 * self.n_channels], dim=1)
        if num_out == 1:
            l = (l1 + l2) / 2
            return torch.cat([l, h1, h2], dim=1)
        elif num_out == 2:
            l = torch.cat([l1, l2], dim=1)
            h = torch.cat([h1, h2], dim=1)
            return l, h
        else:
            raise ValueError

    def cat_noise_level(self, x, noise_level):
        """
        input noise level range [0, 255]
        """
        b, _, h, w = x.shape
        noise_level = (noise_level / 255.) * torch.ones(b, 1, h, w).float().to(x.device)
        return torch.cat([x, noise_level], dim=1)
