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
    psnr_y_dict = {}
    ssim_y_dict = {}
    lpips_dict = {}

    for test_data in test_loader:
        idx += 1
        img_path = test_data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.feed_data(test_data)
        model.test()
        visuals = model.get_current_visuals()

        for quality in opt['val']['comp_quality']:
            gt_img = visuals['GT']
            sr_img = visuals[('SR', quality)]
            lpips_dict[(idx, quality)] = float(loss_fn_alex(2 * gt_img.to('cuda') - 1, 2 * sr_img.to('cuda') - 1).cpu())

            gt_img = util.tensor2img(gt_img)
            sr_img = util.tensor2img(sr_img)
            save_img_path = os.path.join(dataset_dir, 'SR_{:s}_q{:d}.png'.format(img_name, quality))
            util.save_img(sr_img, save_img_path)

            _, _, psnr_y_dict[(idx, quality)], ssim_y_dict[(idx, quality)] = util.calculate_psnr_ssim(gt_img/255., sr_img/255., crop_border)

    logger.info('-------------------------------------------------------------------------------------')
    for quality in opt['val']['comp_quality']:
        avg_psnr_y = 0.0
        avg_ssim_y = 0.0
        avg_lpips = 0.0

        for iidx in range(1, idx + 1):
            avg_psnr_y += psnr_y_dict[(iidx, quality)]
            avg_ssim_y += ssim_y_dict[(iidx, quality)]
            avg_lpips += lpips_dict[(iidx, quality)]

        avg_psnr_y /= idx
        avg_ssim_y /= idx
        avg_lpips /= idx

        # log
        logger.info('----{} ({}images,quality{:d}) '
                    'average PSNR_Y/SSIM_Y/LPIPS: {:.2f}/{:.4f}/{:.4f}'.format(
            test_set_name, idx, quality,
            avg_psnr_y, avg_ssim_y, avg_lpips))
