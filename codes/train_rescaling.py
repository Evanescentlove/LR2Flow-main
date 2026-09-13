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
    for epoch in range(start_epoch, total_epochs + 1):
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
                psnr_dict = {}
                psnr_y_dict = {}
                avg_lr_psnr_y = 0.0

                for _, val_data in enumerate(val_loader):
                    idx += 1
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()

                    # calculate LR psnr
                    gt_img_lr = util.tensor2img(visuals['LQ'])
                    sr_img_lr = util.tensor2img(visuals['LQ_fromH'])
                    gt_img_lr = gt_img_lr / 255.
                    sr_img_lr = sr_img_lr / 255.

                    _, _, lr_psnr_y, _ = util.calculate_psnr_ssim(gt_img_lr, sr_img_lr, 0)
                    avg_lr_psnr_y += lr_psnr_y

                    # deal with sr images
                    for heat in opt['val']['heats']:
                        sr_img_list =[]
                        for sample in range(opt['val']['n_sample']):
                            gt_img = visuals['GT']
                            sr_img = visuals['SR', heat, sample]
                            sr_img_list.append(sr_img.unsqueeze(0)*255)

                            gt_img = util.tensor2img(gt_img)  # uint8
                            sr_img = util.tensor2img(sr_img)  # uint8

                            gt_img = gt_img / 255.
                            sr_img = sr_img / 255.

                            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
                            psnr_dict[(idx, heat, sample)], ssim, psnr_y_dict[(idx, heat, sample)], ssim_y = util.calculate_psnr_ssim(gt_img, sr_img, crop_border)

                # log
                logger.info('{}@{}, GPU {}, Job_id {}, Job path {}'.format(getpass.getuser(), socket.gethostname(), opt['gpu_ids'], args.job_id, args.job_path))
                logger.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('# {}, Validation (<epoch:{:3d}, iter:{:8d}>)'.format(opt['name'], epoch, current_step))

                avg_lr_psnr_y = avg_lr_psnr_y / idx
                for heat in opt['val']['heats']:
                    avg_psnr = 0.0
                    avg_psnr_y = 0.0

                    for iidx in range(1, idx+1):
                        for sample in range(opt['val']['n_sample']):
                            avg_psnr += psnr_dict[(iidx, heat, sample)]
                            avg_psnr_y += psnr_y_dict[(iidx, heat, sample)]

                    avg_psnr = avg_psnr / idx / opt['val']['n_sample']
                    avg_psnr_y = avg_psnr_y / idx / opt['val']['n_sample']

                    # log
                    logger.info('({}samples,heat:{:.1f}) PSNR/PSNR_Y: {:.2f}/{:.2f}, LR_PSNR_Y: {:.2f}'.format(opt['val']['n_sample'], heat, avg_psnr, avg_psnr_y, avg_lr_psnr_y))
                    logger_val.info('({}samples,heat:{:.1f}) PSNR/PSNR_Y: {:.2f}/{:.2f}, LR_PSNR_Y: {:.2f}'.format(opt['val']['n_sample'], heat, avg_psnr, avg_psnr_y, avg_lr_psnr_y))

                del visuals

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of model training.')


if __name__ == '__main__':
    main()
