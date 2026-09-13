import logging
import torch


def define_G(opt):
    from models.networks.network_rescaling import RescalingNet
    return RescalingNet(opt=opt)


