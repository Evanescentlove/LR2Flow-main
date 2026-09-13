import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.Basic import WaveletTightFrame
from models.modules.FlowStep import FlowStep
from models.modules.ConditionalFlow import ConditionalFlow


class Rescaling_x4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt['n_channels']
        nf_lr = opt['lf_flow']['condFlow']['RRDB_nf']
        nf_lf = opt['condFlow']['RRDB_nf']

        self.wavelet = WaveletTightFrame(in_channels=self.n_channels)

        self.uncond = nn.ModuleList(
            [
                nn.ModuleList([FlowStep(in_channels=9 * self.n_channels if i_level == 0 else 5 * self.n_channels,
                                        flow_permutation='none',
                                        flow_coupling='Affine3shift',
                                        LRvsothers=True if k % 2 == 0 else False,
                                        opt=opt) for k in range(opt['flowstep'][i_level])]) for i_level in range(2)
            ]
        )

        self.cond = nn.ModuleList(
            [ConditionalFlow(in_channels=8 * self.n_channels,
                             cond_channels=self.n_channels + nf_lf + nf_lr if i_level == 0 else 2 * self.n_channels + nf_lr,
                             n_flow_step=opt['condFlow']['flowstep'][i_level],
                             opt=opt['condFlow']) for i_level in range(2)]
        )

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

        for layer in self.uncond[0]:
            coef, _ = layer(coef, reverse=False)

        lf, hf = torch.split(coef, [self.n_channels, 8 * self.n_channels], dim=1)

        coef_, _ = self.wavelet(lf, reverse=False)
        coef_ = self.split_coef(x=coef_, num_in=1)

        for layer in self.uncond[1]:
            coef_, _ = layer(coef_, reverse=False)

        lf_, hf_ = self.cat_coef(x=coef_, num_out=2)
        lf__ = lf_.clone()

        for layer in self.uncond_lf:
            lf__, _ = layer(lf__, reverse=False)

        lr, res = torch.split(lf__, [self.n_channels, self.n_channels], dim=1)

        z_res, u_lr = self.cond_lf(res, u=lr, reverse=False)

        z_hf_, u_lf_ = self.cond[1](hf_, u=[lf_, u_lr], reverse=False)

        u = F.interpolate(torch.cat([u_lf_, u_lr], dim=1), scale_factor=2, mode='nearest')
        z_hf, _ = self.cond[0](hf, u=[lf, u], reverse=False)

        z = torch.cat([z_res.flatten(), z_hf_.flatten(), z_hf.flatten()], dim=0)

        return torch.clamp(lr, 0., 1.), z, [lf, lf_]

    def reverse_flow(self, lr, heat):
        res, u_lr = self.cond_lf(z=None, u=lr, eps_std=heat, reverse=True)

        lf_ = torch.cat([lr, res], dim=1)

        for layer in reversed(self.uncond_lf):
            lf_, _ = layer(lf_, reverse=True)

        hf_, u_lf_ = self.cond[1](z=None, u=[lf_, u_lr], eps_std=heat, reverse=True)

        coef_ = self.split_coef(x=[lf_, hf_], num_in=2)

        for layer in reversed(self.uncond[1]):
            coef_, _ = layer(coef_, reverse=True)

        coef_ = self.cat_coef(x=coef_, num_out=1)

        lf, _ = self.wavelet(coef_, reverse=True)

        u = F.interpolate(torch.cat([u_lf_, u_lr], dim=1), scale_factor=2, mode='nearest')
        hf, _ = self.cond[0](z=None, u=[lf, u], eps_std=heat, reverse=True)

        coef = torch.cat([lf, hf], dim=1)

        for layer in reversed(self.uncond[0]):
            coef, _ = layer(coef, reverse=True)

        hr, _ = self.wavelet(coef, reverse=True)

        return torch.clamp(hr, 0., 1.), [lf, lf_]

    def downscale(self, hr):
        coef, _ = self.wavelet(hr, reverse=False)

        for layer in self.uncond[0]:
            coef, _ = layer(coef, reverse=False)

        lf, hf = torch.split(coef, [self.n_channels, 8 * self.n_channels], dim=1)

        coef_, _ = self.wavelet(lf, reverse=False)

        coef_ = self.split_coef(x=coef_, num_in=1)

        for layer in self.uncond[1]:
            coef_, _ = layer(coef_, reverse=False)

        lf_, hf_ = self.cat_coef(x=coef_, num_out=2)
        lf__ = lf_.clone()

        for layer in self.uncond_lf:
            lf__, _ = layer(lf__, reverse=False)

        lr, res = torch.split(lf__, [self.n_channels, self.n_channels], dim=1)

        z_res, u_lr = self.cond_lf(res, u=lr, reverse=False)

        z_hf_, u_lf_ = self.cond[1](hf_, u=[lf_, u_lr], reverse=False)

        u = F.interpolate(torch.cat([u_lf_, u_lr], dim=1), scale_factor=2, mode='nearest')
        z_hf, _ = self.cond[0](hf, u=[lf, u], reverse=False)

        return lr.clamp_(0, 1), [z_hf, z_hf_, z_res]

    def upscale(self, lr, zs):
        z_hf, z_hf_, z_res = zs

        res, u_lr = self.cond_lf(z=z_res, u=lr, reverse=True)

        lf_ = torch.cat([lr, res], dim=1)

        for layer in reversed(self.uncond_lf):
            lf_, _ = layer(lf_, reverse=True)

        hf_, u_lf_ = self.cond[1](z=z_hf_, u=[lf_, u_lr], reverse=True)

        coef_ = self.split_coef(x=[lf_, hf_], num_in=2)

        for layer in reversed(self.uncond[1]):
            coef_, _ = layer(coef_, reverse=True)

        coef_ = self.cat_coef(x=coef_, num_out=1)

        lf, _ = self.wavelet(coef_, reverse=True)

        u = F.interpolate(torch.cat([u_lf_, u_lr], dim=1), scale_factor=2, mode='nearest')
        hf, _ = self.cond[0](z=z_hf, u=[lf, u], reverse=True)

        coef = torch.cat([lf, hf], dim=1)

        for layer in reversed(self.uncond[0]):
            coef, _ = layer(coef, reverse=True)

        hr, _ = self.wavelet(coef, reverse=True)

        return hr.clamp_(0., 1.)

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








