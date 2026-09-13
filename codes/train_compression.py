import os
import math
import argparse
import random
import logging
import numpy as np
import torch
import cv2

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import socket
import getpass
import lpips
from tqdm import tqdm


def main():
    #### setup options
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--job_id", type=str, default=0)
    parser.add_argument("--job_path", type=str, default="")
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
    logger.info(f"Start training from epoch: {start_epoch:d}, iter: {current_step:d}")
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
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.get_current_learning_rate():.3e}> "
                for k, v in logs.items():
                    message += f"{k:s}:{v:.4e} "
                logger.info(message)

            #### save models and training states before validation
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(current_step)
                model.save_training_state(epoch, current_step)

            # validation
            if current_step % opt["val"]["val_freq"] == 0:
                idx = 0
                # psnr_dict = {}
                psnr_y_dict = {}
                ssim_y_dict = {}
                # lr_psnr_y_dict = {}
                bpp_dict = {quality: 0. for quality in opt["val"]["comp_quality"]}

                dataset_dir = opt["path"]["val_images"]  # save LR images
                util.mkdir(dataset_dir)

                for _, val_data in tqdm(enumerate(val_loader)):
                    idx += 1
                    img_path = val_data["GT_path"][0]
                    img_name = os.path.splitext(os.path.basename(img_path))[0]

                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    gt_img = util.tensor2img(visuals["GT"])
                    # gt_img_lr = util.tensor2img(visuals['LQ'])

                    for quality in opt["val"]["comp_quality"]:
                        sr_img = util.tensor2img(visuals["SR", quality])
                        lr_img = util.tensor2img(visuals["LQ_fromH", quality])

                        # calculate bpp
                        # _, encoded = cv2.imencode(".jpg", lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                        # lr_size = encoded.size  # bytes
                        save_lr_path = os.path.join(dataset_dir, f"{img_name}_LR.jpg")
                        cv2.imwrite(save_lr_path, lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        lr_size = os.path.getsize(save_lr_path)
                        H, W, _ = gt_img.shape
                        bpp = lr_size * 8 / (H * W)
                        bpp_dict[quality] += bpp

                        # _, _, lr_psnr_y_dict[(idx, quality)], _ = (
                        #     util.calculate_psnr_ssim(gt_img_lr/255., sr_img_lr/255., 0))
                        _, _, psnr_y_dict[(idx, quality)], ssim_y_dict[(idx, quality)] = util.calculate_psnr_ssim(gt_img/255., sr_img/255., opt["scale"])

                # log
                logger.info(f"{getpass.getuser()}@{socket.gethostname()}, GPU {opt['gpu_ids']}, Job_id {args.job_id}, Job path {args.job_path}")
                logger.info(f"# {opt['name']}, Validation (<epoch:{epoch:3d}, iter:{current_step:8d}>)")
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(f"# {opt['name']}, Validation (<epoch:{epoch:3d}, iter:{current_step:8d}>)")

                for quality in opt["val"]["comp_quality"]:
                    # avg_psnr = 0.0
                    avg_psnr_y = 0.0
                    # avg_lr_psnr_y = 0.0
                    avg_ssim_y = 0.

                    for iidx in range(1, idx + 1):
                        # avg_psnr += psnr_dict[(iidx, quality)]
                        avg_psnr_y += psnr_y_dict[(iidx, quality)]
                        # avg_lr_psnr_y += lr_psnr_y_dict[(iidx, quality)]
                        avg_ssim_y += ssim_y_dict[(iidx, quality)]

                    # avg_psnr /= idx
                    avg_psnr_y /= idx
                    # avg_lr_psnr_y /= idx
                    avg_ssim_y /= idx
                    avg_bpp = bpp_dict[quality] / idx

                    # log
                    log_str = f"(quality:{quality:d}) PSNR_Y/SSIM_Y/BPP: {avg_psnr_y:.2f}/{avg_ssim_y:.4f}/{avg_bpp:.4f}"
                    logger.info(log_str)
                    logger_val.info(log_str)

                del visuals

    logger.info("Saving the final model.")
    model.save("latest")
    logger.info("End of model training.")


if __name__ == '__main__':
    main()
