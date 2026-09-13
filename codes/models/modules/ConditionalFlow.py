import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import thops
from utils.util import opt_get
from models.modules.Basic import Conv2d, Conv2dZeros, GaussianDiag, DenseBlock, RRDB, FCN
from models.modules.FlowStep import FlowStep

import functools
import models.modules.module_util as mutil


class ConditionalFlow(nn.Module):
    def __init__(self, in_channels, cond_channels, n_flow_step=0, opt=None):
        super(ConditionalFlow, self).__init__()

        RRDB_nb = opt['RRDB_nb']
        RRDB_gc = opt['RRDB_gc']
        RRDB_nf = opt['RRDB_nf']

        RRDB_f = functools.partial(RRDB, nf=RRDB_nf, gc=RRDB_gc)
        # functions to exact features from u
        self.conv_first = nn.Conv2d(in_channels=cond_channels, out_channels=RRDB_nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.RRDB_trunk0 = mutil.make_layer(RRDB_f, RRDB_nb[0])
        self.RRDB_trunk1 = mutil.make_layer(RRDB_f, RRDB_nb[1])
        self.trunk_conv1 = nn.Conv2d(in_channels=RRDB_nf, out_channels=RRDB_nf, kernel_size=3, stride=1, padding=1, bias=True)

        # invertible flow condition on masked features
        self.additional_flow_steps = nn.ModuleList()
        for k in range(n_flow_step):
            self.additional_flow_steps.append(FlowStep(in_channels=in_channels,
                                                       cond_channels=RRDB_nf,
                                                       flow_permutation=opt['flow_permutation'],
                                                       flow_coupling=opt['flow_coupling'], opt=opt))

        # extract shift and scale from masked features
        self.f = Conv2dZeros(in_channels=RRDB_nf, out_channels=2 * in_channels)

    def forward(self, z, u, eps_std=None, logdet=None, reverse=False):
        if isinstance(u, list):
            u = torch.cat(u, dim=1)
        cond_feature = self.get_conditional_feature_Rescaling(u)

        if not reverse:
            for layer in self.additional_flow_steps:
                z, _ = layer(z, u=cond_feature, logdet=None, reverse=False)
            h = self.f(cond_feature)
            mean, scale = thops.split_feature(h, "cross")
            logscale = 0.318 * torch.atan(2 * scale)
            z = (z - mean) * torch.exp(-logscale)
            return z, cond_feature
        else:
            h = self.f(cond_feature)
            mean, scale = thops.split_feature(h, "cross")
            logscale = 0.318 * torch.atan(2 * scale)
            z = GaussianDiag.sample(mean, logscale, eps_std) if z is None else z * torch.exp(logscale) + mean
            for layer in reversed(self.additional_flow_steps):
                z, _ = layer(z, u=cond_feature, logdet=None, reverse=True)
            return z, cond_feature

    def get_conditional_feature_Rescaling(self, u):
        u_feature_first = self.conv_first(u)
        u_feature = self.trunk_conv1(self.RRDB_trunk1(self.RRDB_trunk0(u_feature_first))) + u_feature_first  # num channels = num_features_total

        return u_feature


