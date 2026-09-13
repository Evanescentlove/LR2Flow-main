import logging
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.optim import Adam, AdamW
from torch.optim import lr_scheduler
import models.networks as networks

from models.networks.network_denoising import DenoisingNet
from models.base_model import BaseModel
import utils.util as util
from utils.model_util import test_one_split, test_split_x8

logger = logging.getLogger('base')


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

        # define network
        self.netG = DenoisingNet(opt['network_G'])
        self.netG = DataParallel(self.netG).to(self.device)

        if self.is_train:
            train_opt = opt['train']
            self.netG.train()
            self.noise_levels = [i for i in range(train_opt['noise_level'][0], train_opt['noise_level'][1] + 1)]

            # define loss
            self.w_lf = train_opt['weight_lf']
            self.w_hf = train_opt['weight_hf']

            # gradient clip & norm
            self.max_grad_clip = train_opt['max_grad_clip']
            self.max_grad_norm = train_opt['max_grad_norm']

            # define optimizer
            optim_params = [p for p in self.netG.parameters() if p.requires_grad]
            self.optimizer = AdamW(
                optim_params, train_opt['lr'], (train_opt['beta1'], train_opt['beta2']),
                weight_decay=0.0
            )
            self.optimizers.append(self.optimizer)

            # define scheduler
            self.schedulers.append(
                lr_scheduler.MultiStepLR(self.optimizer, train_opt['lr_steps'], train_opt['lr_gamma'])
            )

            self.log_dict = OrderedDict()

        # val
        self.nl = opt['val']['noise_level']
        self.ensemble = opt['val'].get("self_ensemble", True)

        self.print_network()
        self.load()

    def feed_data(self, data):
        self.img_L = data['LQ'].to(self.device)
        self.img_H = data['GT'].to(self.device)

    def generate_L(self, img_H):
        b = img_H.shape[0]
        rnd_noise_level = torch.tensor(random.choices(self.noise_levels, k=b)).reshape(b, 1, 1, 1).float().to(self.device)
        img_L = img_H + torch.randn_like(img_H) * (rnd_noise_level / 255.)
        return img_L, rnd_noise_level

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        img_L, noise_level = self.generate_L(self.img_H)
        loss_hr, loss_lf, loss_hf = self.netG(self.img_H, img_L, noise_level)

        loss_lf = self.w_lf * loss_lf
        loss_hf = self.w_hf * loss_hf
        self.log_dict['loss_hr'] = loss_hr.item()
        self.log_dict['loss_lf'] = loss_lf.item()
        self.log_dict['loss_hf'] = loss_hf.item()

        loss = loss_hr + loss_lf + loss_hf
        loss.backward()
        self.gradient_clip()
        self.optimizer.step()

    def gradient_clip(self):
        if self.max_grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.netG.parameters(), self.max_grad_clip)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.max_grad_norm)

    def test(self):
        self.netG.eval()
        model = self.netG.module.denoise
        with torch.no_grad():
            if not self.ensemble:
                self.img_E = test_one_split(model, self.img_L, self.nl, refield=64)
            else:
                self.img_E = test_split_x8(model, self.img_L, self.nl, refield=64)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_L'] = self.img_L.detach()[0].float().cpu()
        out_dict['img_H'] = self.img_H.detach()[0].float().cpu()
        out_dict['img_E'] = self.img_E.detach()[0].float().cpu()
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
