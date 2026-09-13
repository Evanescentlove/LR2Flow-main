import logging
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

logger = logging.getLogger('base')


class RescalingModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.train_opt = opt['train']

        # define network
        self.netG = networks.define_G(opt=opt['network_G'])
        self.netG = DataParallel(self.netG).to(self.device)

        self.Quantization = Quantization()

        if self.is_train:
            train_opt = self.train_opt
            self.netG.train()

            self.heat = train_opt['heat']

            # define loss
            self.cri_hr = nn.L1Loss().to(self.device)
            self.cri_lr = nn.MSELoss().to(self.device)
            self.cri_lf = nn.MSELoss().to(self.device)

            self.w_hr = train_opt['weight_hr']
            self.w_lr = train_opt['weight_lr']
            self.w_z = train_opt['weight_z']
            self.w_lf = train_opt['weight_lf']

            # gradient clip & norm
            self.max_grad_clip = train_opt['max_grad_clip']
            self.max_grad_norm = train_opt['max_grad_norm']

            # define optimizer
            optim_params = [p for p in self.netG.parameters() if p.requires_grad]
            self.optimizer_G = AdamW(optim_params,
                                     lr=train_opt['lr_G'],
                                     betas=(train_opt['beta1'], train_opt['beta2']),
                                     weight_decay=0)
            self.optimizers.append(self.optimizer_G)

            # define scheduler
            self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer_G,
                                                            milestones=train_opt['lr_steps'],
                                                            gamma=train_opt['lr_gamma']))

            self.log_dict = OrderedDict()

        # val
        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']

        self.print_network()  # print network
        self.load()  # load G if needed

    def feed_data(self, data):
        self.var_L = data['LQ'].to(self.device)
        self.real_H = data['GT'].to(self.device)

    def loss_lf(self, lf_forw, lf_back):
        assert len(lf_forw) == len(lf_back)
        w_lf = [self.w_lf,] * len(lf_forw) if not isinstance(self.w_lf, list) else self.w_lf

        loss_lf = 0.
        for i in range(len(lf_forw)):
            loss_lf = loss_lf + w_lf[i] * self.cri_lf(lf_forw[i], lf_back[i])

        return loss_lf

    def process_lr(self, lr):
        # quantization, add noise
        lr = self.Quantization(lr)
        prob = np.random.rand()
        if prob < self.train_opt['add_noise_prob']:
            lr = lr + self.train_opt['noise_scale'] * torch.randn_like(lr)
        return lr

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        fake_L, z, lf_forw = self.netG(x=self.real_H, reverse=False)  # downscaling
        fake_H, lf_back = self.netG(x=self.process_lr(fake_L), heat=self.heat, reverse=True)  # upscaling

        loss_hr = self.w_hr * self.cri_hr(fake_H, self.real_H)
        loss_lr = self.w_lr * self.cri_lr(fake_L, self.var_L)
        loss_z = self.w_z * (z ** 2).mean()
        loss_lf = self.loss_lf(lf_forw=lf_forw, lf_back=lf_back)

        self.log_dict['loss_hr'] = loss_hr.item()
        self.log_dict['loss_lr'] = loss_lr.item()
        self.log_dict['loss_z'] = loss_z.item()
        self.log_dict['loss_lf'] = loss_lf.item()

        loss_total = loss_hr + loss_lr + loss_z + loss_lf
        loss_total.backward()
        self.gradient_clip()
        self.optimizer_G.step()

    def gradient_clip(self):
        if self.max_grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.max_grad_clip)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)

    def test(self):
        self.netG.eval()
        self.fake_H = {}

        with torch.no_grad():
            fake_L, *_ = self.netG(x=self.real_H, reverse=False)
            self.fake_L_from_H = self.Quantization(fake_L)
            for heat in self.heats:
                for i in range(self.n_sample):
                    self.fake_H[(heat, i)], *_ = self.netG(x=self.fake_L_from_H, heat=heat, reverse=True)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['LQ_fromH'] = self.fake_L_from_H.detach()[0].float().cpu()
        for heat in self.heats:
            for i in range(self.n_sample):
                out_dict[('SR', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()

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

        # resume training according to given paths (pretrain path has been overrided by resume path)
        if self.opt.get('path') is not None:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True))

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
