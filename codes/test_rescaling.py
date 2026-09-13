import os.path
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
import options.options as option
import utils.util as util
from utils.imresize import imresize
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import lpips



#### options
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, help='Path to options YMAL file.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)
device_id = torch.cuda.current_device()

#### mkdir and logger
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
             and 'pretrain_model' not in key and 'resume' not in key and 'load_submodule' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

# set random seed
util.set_random_seed(0)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\n\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    idx = 0
    psnr_dict={} # for HR image
    ssim_dict={}
    psnr_y_dict = {}
    ssim_y_dict = {}
    lpips_dict = {}
    avg_lr_psnr = 0.0 # for generated LR image
    avg_lr_ssim = 0.0
    avg_lr_psnr_y = 0.0
    avg_lr_ssim_y = 0.0

    for test_data in test_loader:
        idx += 1
        img_path = test_data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.feed_data(test_data)
        model.test()
        visuals = model.get_current_visuals()

        # calculate PSNR for LR
        gt_img_lr = util.tensor2img(visuals['LQ'])
        sr_img_lr = util.tensor2img(visuals['LQ_fromH'])
        save_img_path = os.path.join(dataset_dir, 'LR_{:s}.png'.format(img_name))
        util.save_img(sr_img_lr, save_img_path)
        gt_img_lr = gt_img_lr / 255.
        sr_img_lr = sr_img_lr / 255.

        lr_psnr, lr_ssim, lr_psnr_y, lr_ssim_y = util.calculate_psnr_ssim(gt_img_lr, sr_img_lr, 0)
        avg_lr_psnr += lr_psnr
        avg_lr_ssim += lr_ssim
        avg_lr_psnr_y += lr_psnr_y
        avg_lr_ssim_y += lr_ssim_y

        for heat in opt['val']['heats']:
            psnr = 0.0
            ssim = 0.0
            psnr_y = 0.0
            ssim_y = 0.0
            lpips_value = 0.0

            for sample in range(opt['val']['n_sample']):
                gt_img = visuals['GT']  # [0, 1]
                sr_img = visuals['SR', heat, sample]  # [0, 1]
                lpips_dict[(idx, heat, sample)] = float(loss_fn_alex(2 * gt_img.to('cuda') - 1, 2 * sr_img.to('cuda') - 1).cpu())
                lpips_value += lpips_dict[(idx, heat, sample)]

                gt_img = util.tensor2img(gt_img)  # uint8
                sr_img = util.tensor2img(sr_img)  # uint8
                save_img_path = os.path.join(dataset_dir, 'SR_{:s}_{:.1f}_{:d}.png'.format(img_name, heat, sample))
                util.save_img(sr_img, save_img_path)

                gt_img = gt_img / 255.  # [0, 1]
                sr_img = sr_img / 255.  # [0, 1]

                psnr_dict[(idx, heat, sample)], ssim_dict[(idx, heat, sample)], \
                psnr_y_dict[(idx, heat, sample)], ssim_y_dict[(idx, heat, sample)] = util.calculate_psnr_ssim(gt_img, sr_img, crop_border)
                psnr += psnr_dict[(idx, heat, sample)]
                ssim += ssim_dict[(idx, heat, sample)]
                psnr_y += psnr_y_dict[(idx, heat, sample)]
                ssim_y += ssim_y_dict[(idx, heat, sample)]

            # mean pixel-wise variance
            psnr /= opt['val']['n_sample']
            ssim /= opt['val']['n_sample']
            psnr_y /= opt['val']['n_sample']
            ssim_y /= opt['val']['n_sample']
            lpips_value /= opt['val']['n_sample']

            logger.info('{:20s} ({}samples),heat:{:.1f}) '
                        'HR:PSNR/SSIM/PSNR_Y/SSIM_Y/LPIPS: {:.2f}/{:.4f}/{:.2f}/{:.4f}/{:.4f}, '
                        'LR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}'.format(
                img_name, opt['val']['n_sample'], heat,
                psnr, ssim, psnr_y, ssim_y, lpips_value,
                lr_psnr, lr_ssim, lr_psnr_y, lr_ssim_y))

    # Average PSNR/SSIM results
    avg_lr_psnr /= idx
    avg_lr_ssim /= idx
    avg_lr_psnr_y /= idx
    avg_lr_ssim_y /= idx

    logger.info('-------------------------------------------------------------------------------------')
    for heat in opt['val']['heats']:
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_psnr_y = 0.0
        avg_ssim_y = 0.0
        avg_lpips = 0.0

        for iidx in range(1, idx+1):
            for sample in range(opt['val']['n_sample']):
                avg_psnr += psnr_dict[(iidx, heat, sample)]
                avg_ssim += ssim_dict[(iidx, heat, sample)]
                avg_psnr_y += psnr_y_dict[(iidx, heat, sample)]
                avg_ssim_y += ssim_y_dict[(iidx, heat, sample)]
                avg_lpips += lpips_dict[(iidx, heat, sample)]

        avg_psnr = avg_psnr / idx / opt['val']['n_sample']
        avg_ssim = avg_ssim / idx / opt['val']['n_sample']
        avg_psnr_y = avg_psnr_y / idx / opt['val']['n_sample']
        avg_ssim_y = avg_ssim_y / idx / opt['val']['n_sample']
        avg_lpips = avg_lpips / idx  / opt['val']['n_sample']

        # log
        logger.info(opt['path']['pretrain_model_G'])
        logger.info('----{} ({}images,{}samples,heat:{:.1f}) '
                    'average HR:PSNR/SSIM/PSNR_Y/SSIM_Y/LPIPS: {:.2f}/{:.4f}/{:.2f}/{:.4f}/{:.4f}, '
                    'LR:PSNR/SSIM/PSNR_Y/SSIM_Y: {:.2f}/{:.4f}/{:.2f}/{:.4f}'.format(
            test_set_name, idx, opt['val']['n_sample'], heat,
            avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y, avg_lpips,
            avg_lr_psnr, avg_lr_ssim, avg_lr_psnr_y, avg_lr_ssim_y))
