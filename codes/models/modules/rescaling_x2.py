import torch
import torch.nn as nn

from models.modules.Basic import HaarDownsampling, WaveletTightFrame
from models.modules.FlowStep import FlowStep
from models.modules.ConditionalFlow import ConditionalFlow


class Rescaling_x2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt['n_channels']
        nf_lr = opt['lf_flow']['condFlow']['RRDB_nf']

        self.wavelet = WaveletTightFrame(in_channels=self.n_channels)
        self.uncond = nn.ModuleList(
            [FlowStep(in_channels=5 * self.n_channels,
                      flow_permutation='none',
                      flow_coupling='Affine3shift',
                      LRvsothers=True if k % 2 == 0 else False,
                      opt=opt) for k in range(opt['flowstep'])]
        )
        self.cond = ConditionalFlow(in_channels=8 * self.n_channels,
                                    cond_channels=2 * self.n_channels + nf_lr,
                                    n_flow_step=opt['condFlow']['flowstep'],
                                    opt=opt['condFlow'])

        self.uncond_lf = nn.ModuleList(
            [FlowStep(in_channels=2 * self.n_channels,
                      flow_permutation='none',
                      flow_coupling='Affine3shift',
                      LRvsothers=True if k % 2 == 0 else False,
                      opt=opt['lf_flow']) for k in range(opt['lf_flow']['flowstep'])]
        )
        self.cond_lf = ConditionalFlow(in_channels=self.n_channels,
                                       cond_channels=self.n_channels,
                                       n_flow_step=opt['lf_flow']['condFlow']['flowstep'],
                                       opt=opt['lf_flow']['condFlow'])

    def forward(self, *, x, heat=0., reverse):
        return self.normal_flow(hr=x) if not reverse else self.reverse_flow(lr=x, heat=heat)

    def normal_flow(self, hr):
        coef, _ = self.wavelet(hr, reverse=False)
        coef = self.split_coef(x=coef, num_in=1)
        for layer in self.uncond:
            coef, _ = layer(coef, reverse=False)
        lf, hf = self.cat_coef(x=coef, num_out=2)
        lf_ = lf.clone()
        for layer in self.uncond_lf:
            lf_, _ = layer(lf_, reverse=False)
        lr, res = torch.split(lf_, [self.n_channels, self.n_channels], dim=1)
        z_res, u_lr = self.cond_lf(res, u=lr, reverse=False)
        z_hf, _ = self.cond(hf, u=[lf, u_lr], reverse=False)
        z = torch.cat([z_res.flatten(), z_hf.flatten()], dim=0)
        return torch.clamp(lr, 0, 1), z, [lf]

    def reverse_flow(self, lr, heat):
        res, u_lr = self.cond_lf(z=None, u=lr, eps_std=heat, reverse=True)
        lf = torch.cat([lr, res], dim=1)
        for layer in reversed(self.uncond_lf):
            lf, _ = layer(lf, reverse=True)
        hf, _ = self.cond(z=None, u=[lf, u_lr], eps_std=heat, reverse=True)
        coef = self.split_coef(x=[lf, hf], num_in=2)
        for layer in reversed(self.uncond):
            coef, _ = layer(coef, reverse=True)
        coef = self.cat_coef(x=coef, num_out=1)
        hr, _ = self.wavelet(coef, reverse=True)
        return torch.clamp(hr, 0, 1), [lf]

    def downscale(self, hr):
        coef, _ = self.wavelet(hr, reverse=False)

        coef = self.split_coef(x=coef, num_in=1)

        for layer in self.uncond:
            coef, _ = layer(coef, reverse=False)

        lf, hf = self.cat_coef(x=coef, num_out=2)

        lf_ = lf.clone()

        for layer in self.uncond_lf:
            lf_, _ = layer(lf_, reverse=False)

        lr, res = torch.split(lf_, [self.n_channels, self.n_channels], dim=1)

        z_res, u_lr = self.cond_lf(res, u=lr, reverse=False)

        z_hf, _ = self.cond(hf, u=[lf, u_lr], reverse=False)

        return lr.clamp_(0, 1), [z_hf, z_res]

    def upscale(self, lr, zs):
        z_hf, z_res = zs

        res, u_lr = self.cond_lf(z=z_res, u=lr, reverse=True)

        lf = torch.cat([lr, res], dim=1)

        for layer in reversed(self.uncond_lf):
            lf, _ = layer(lf, reverse=True)

        hf, _ = self.cond(z=z_hf, u=[lf, u_lr], reverse=True)

        coef = self.split_coef(x=[lf, hf], num_in=2)

        for layer in reversed(self.uncond):
            coef, _ = layer(coef, reverse=True)

        coef = self.cat_coef(x=coef, num_out=1)

        hr, _ = self.wavelet(coef, reverse=True)

        return torch.clamp(hr, 0., 1.)

    def split_coef(self, x, num_in):
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

    def cat_coef(self, x, num_out):
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









    

