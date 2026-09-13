import importlib

import torch
import torch.nn as nn


class RescalingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        scale = opt['scale']
        model_filename = 'models.modules.rescaling_x{}'.format(scale)
        modellib = importlib.import_module(model_filename)

        model = None
        target_model_name = 'Rescaling_x{}'.format(scale)
        for name, cls in modellib.__dict__.items():
            if name.lower() == target_model_name.lower():
                model = cls
        if model is None:
            print('In {:s}.py, there should be a subclass of torch.nn.Module with class name that matches {:s}.'.format(
                model_filename, target_model_name))
            exit(0)
        self.flow = model(opt)

    def forward(self, **kwargs):
        return self.flow(**kwargs)

    def rescale(self, x, reverse):
        return self.flow.downscale(hr=x) if not reverse else self.flow.upscale(*x)
