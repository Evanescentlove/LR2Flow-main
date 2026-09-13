import importlib
import logging
import os

logger = logging.getLogger('base')


def create_model(opt, step=0, **opt_kwargs):
    for k, v in opt_kwargs.items():
        opt[k] = v

    name = opt['name']
    if "rescaling" in name.lower():
        from models.model_rescaling import RescalingModel
        model = RescalingModel(opt=opt)
    elif "compression" in name.lower():
        from models.model_compression import CompressionModel
        model = CompressionModel(opt)
    elif 'denoising' in name.lower():
        from models.model_denoising import DenoisingModel
        model = DenoisingModel(opt)

    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    return model
