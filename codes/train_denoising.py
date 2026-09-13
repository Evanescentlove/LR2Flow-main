import os
import math
import argparse
import random
import logging
import numpy as np
import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import socket
import getpass
import lpips


def main():
    #### setup options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--job_id', type=str, default=0)
    parser.add_argument('--job_path', type=str, default='')
    args = parser.parse_args()
    opt = option.parse(args.opt, args.gpu_ids, is_train=True)
    device_id = torch.cuda.current_device()

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    print(torch.__version__)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = util.get_resume_paths(opt)
        if resume_state_path is None:
            resume_state = None
        else:
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # override model pretrain path with resume path
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train{}_'.format(args.job_id) + opt['name'], level=logging.INFO, screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val{}_'.format(args.job_id) + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(), opt['gpu_ids'], args.job_id, args.job_path))
    logger.info(option.dict2str(opt))

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)
    loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs+10):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate, schedulers
            model.update_learning_rate(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}:{:.4e} '.format(k, v)
                logger.info(message)

            #### save models and training states before validation
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

            # validation
            if current_step % opt['val']['val_freq'] == 0:
                idx = 0
                avg_psnr = 0.0
                avg_ssim = 0.0
                # avg_lpips = 0.0

                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()

                    img_H = visuals['img_H']
                    img_E = visuals['img_E']
                    # avg_lpips += float(loss_fn_alex(2 * img_H.to('cuda') - 1, 2 * img_E.to('cuda') - 1).cpu())

                    img_H = util.tensor2img(img_H)  # uint8
                    img_E = util.tensor2img(img_E)  # uint8
                    avg_psnr += util.calculate_psnr(img_E, img_H)
                    avg_ssim += util.calculate_ssim(img_E, img_H)

                # log
                logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(
                    getpass.getuser(), socket.gethostname(), opt['gpu_ids'], args.job_id, args.job_path))
                logger.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))

                # avg_lpips /= idx
                avg_psnr /= idx
                avg_ssim /= idx
                # logger.info('PSNR/SSIM/LPIPS: {:.2f}/{:.2f}/{:.4f} '.format(avg_psnr, avg_ssim, avg_lpips))
                # logger_val.info('PSNR/SSIM/LPIPS: {:.2f}/{:.2f}/{:.4f} '.format(avg_psnr, avg_ssim, avg_lpips))
                logger.info('PSNR/SSIM: {:.2f}/{:.2f} '.format(avg_psnr, avg_ssim))
                logger_val.info('PSNR/SSIM: {:.2f}/{:.2f} '.format(avg_psnr, avg_ssim))

                del visuals

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of model training.')


if __name__ == "__main__":
    main()
