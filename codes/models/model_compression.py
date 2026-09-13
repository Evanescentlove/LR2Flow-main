import logging
import random
import math

import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
import utils.util as util
from models.modules.Basic import Quantization, thresholding
from models.modules.compressor import REALCOMP
from models.modules.jpeg import DiffJPEG

logger = logging.getLogger('base')


class FactorizedGaussianEntropyModel(nn.Module):
    def __init__(self, channels, init_scale=0.1, min_scale=1e-6, quant_step=1/255.):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Better softplus initialization
        init = math.log(math.exp(init_scale) - 1.0)
        self.log_scale = nn.Parameter(torch.full((1, channels, 1, 1), init))

        self.min_scale = min_scale
        self.quant_step = quant_step

    def _standardized_cumulative(self, inputs):
        return 0.5 * torch.erfc(-inputs / math.sqrt(2.0))

    def likelihood(self, y_hat):
        scale = F.softplus(self.log_scale) + self.min_scale

        upper = (y_hat - self.mu + 0.5 * self.quant_step) / scale
        lower = (y_hat - self.mu - 0.5 * self.quant_step) / scale

        likelihood = self._standardized_cumulative(upper) - self._standardized_cumulative(lower)
        return likelihood.clamp(min=1e-9)

    def rate_bpp(self, y_hat, num_pixels):
        likelihood = self.likelihood(y_hat)
        bits = -torch.log(likelihood) / math.log(2.0)
        return bits.sum() / num_pixels

    def forward(self, y_hat, num_pixels):
        return self.rate_bpp(y_hat, num_pixels)


class CompressionModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.train_opt = opt['train']
        # define network
        from models.networks.network_rescaling import RescalingNet
        self.netG = RescalingNet(opt['network_G'])
        self.netG = DataParallel(self.netG).to(self.device)
        self.Quantization = Quantization()

        if self.is_train:
            train_opt = opt['train']
            self.netG.train()
            self.heat = train_opt['heat']
            self.comp_qualitys = [i for i in range(train_opt['comp_quality'][0], train_opt['comp_quality'][1] + 1) if i % 5 == 0]

            # define entropy model
            self.netE = DataParallel(FactorizedGaussianEntropyModel(channels=opt["n_channels"]).to(self.device))

            # define loss
            self.cri_hr = nn.L1Loss().to(self.device)
            self.cri_lr = nn.MSELoss().to(self.device)
            self.cri_lf = nn.MSELoss().to(self.device)

            self.w_hr = train_opt['weight_hr']
            self.w_lr = train_opt['weight_lr']
            self.w_z = train_opt['weight_z']
            self.w_lf = train_opt['weight_lf']
            self.w_rate = train_opt["weight_rate"]

            # gradient clip & norm
            self.max_grad_clip = train_opt['max_grad_clip']
            self.max_grad_norm = train_opt['max_grad_norm']

            # define optimizer
            optim_params = [p for p in self.netG.parameters() if p.requires_grad] + [p for p in self.netE.parameters() if p.requires_grad]
            self.optimizer_G = AdamW(optim_params, train_opt['lr_G'], (train_opt['beta1'], train_opt['beta2']),
                                     weight_decay=0.0)
            self.optimizers.append(self.optimizer_G)

            # define scheduler
            self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer_G, train_opt['lr_steps'], train_opt['lr_gamma']))

            self.log_dict = OrderedDict()

        # val
        self.heat_ = opt['val']['heat']
        self.comp_quality_ = opt['val']['comp_quality']

        self.print_network()  # print network
        self.load()  # load G if needed

    def feed_data(self, data):
        self.var_L = data['LQ'].to(self.device)
        self.real_H = data['GT'].to(self.device)

    def loss_lf(self, lf_forw, lf_back):
        assert len(lf_forw) == len(lf_back)
        w_lf = self.w_lf if isinstance(self.w_lf, (list, tuple)) else [self.w_lf,] * len(lf_forw)
        loss_lf = 0.0
        for i in range(len(lf_forw)):
            loss_lf = loss_lf + w_lf[i] * self.cri_lf(lf_forw[i], lf_back[i])
        return loss_lf

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        # randomly select compression quality factor
        quality = random.choice(self.comp_qualitys)
        diffcomp = DiffJPEG(differentiable=True, quality=quality).to(self.device)
        # forward
        lr, z, lf_forw = self.netG(x=self.real_H, reverse=False)
        qlr = self.Quantization(lr)
        comp_lr = diffcomp(qlr)
        # backward
        fake_H, lf_back = self.netG(x=comp_lr, heat=self.heat, reverse=True)

        loss_hr = self.w_hr * self.cri_hr(fake_H, self.real_H)
        self.log_dict['loss_hr'] = loss_hr.item()

        loss_lr = self.w_lr * self.cri_lr(lr, self.var_L)
        self.log_dict['loss_lr'] = loss_lr.item()

        loss_z = self.w_z * (z**2).mean()
        self.log_dict['loss_z'] = loss_z.item()

        loss_lf = self.loss_lf(lf_forw, lf_back)
        self.log_dict['loss_lf'] = loss_lf.item()

        batch, _, H, W = self.real_H.size()
        rate_bpp = self.netE(qlr, batch * H * W)

        loss_rate = self.w_rate * rate_bpp
        self.log_dict["loss_rate"] = loss_rate.item()

        loss = loss_hr + loss_lr + loss_z + loss_lf + loss_rate
        loss.backward()
        self.gradient_clip()
        self.optimizer_G.step()

    def gradient_clip(self):
        if self.max_grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.max_grad_clip)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)

    def test(self):
        self.netG.eval()
        self.fake_L_from_H = {}
        self.fake_H = {}

        with torch.no_grad():
            for quality in self.comp_quality_:
                realcomp = REALCOMP(quality=quality)
                lr, *_ = self.netG(x=self.real_H, reverse=False)
                comp_lr = realcomp(self.Quantization(lr))
                self.fake_L_from_H[quality] = comp_lr
                self.fake_H[quality], *_ = self.netG(x=comp_lr, heat=self.heat_, reverse=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        for quality in self.comp_quality_:
            out_dict[('LQ_fromH', quality)] = self.fake_L_from_H[quality].detach()[0].float().cpu()
            out_dict[('SR', quality)] = self.fake_H[quality].detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        # resume training automatically if resume_state=='auto'
        _, get_resume_model_path = util.get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            logger.info('Automatically loading model for G [{:s}] ...'.format(get_resume_model_path))
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)

            resume_model_path_E = get_resume_model_path.replace("_G.pth", "_E.pth")
            logger.info(f"Automatically loading model for Entropy Model [{resume_model_path_E}].")
            self.load_network(resume_model_path_E, self.netE, strict=True, submodule=None)
            return

        # resume training according to given paths (pretrain path has been overrided by resume path)
        if self.opt.get('path') is not None:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True))

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netE, "E", iter_label)
