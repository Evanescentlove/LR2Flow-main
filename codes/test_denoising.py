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
# loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\n\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    idx = 0
    avg_psnr = 0.0
    avg_ssim = 0.0
    # avg_lpips = 0.0

    for test_data in test_loader:
        idx += 1
        img_path = test_data['GT_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        model.feed_data(test_data)
        model.test()

        visuals = model.get_current_visuals()
        img_H = visuals['img_H']
        img_E = visuals['img_E']

        # lpips = float(loss_fn_alex(2 * img_H.to('cuda') - 1, 2 * img_E.to('cuda') - 1).cpu())
        # avg_lpips += lpips

        img_H = util.tensor2img(img_H)
        img_E = util.tensor2img(img_E)
        # save denoise results
        save_img_path = os.path.join(dataset_dir, '{:s}.png'.format(img_name))
        util.save_img(img_E, save_img_path)
        # calculate PSNR/SSIM
        psnr = util.calculate_psnr(img_E, img_H)
        ssim = util.calculate_ssim(img_E, img_H)
        avg_psnr += psnr
        avg_ssim += ssim
        logger.info('{:20s}: PSNR/SSIM/LPIPS: {:.2f}/{:.4f}/{:.4f}'.format(img_name, psnr, ssim, lpips))

    avg_psnr /= idx
    avg_ssim /= idx
    # avg_lpips /= idx
    logger.info('-------------------------------------------------------------------------------------')
    logger.info(opt['path']['pretrain_model_G'])
    # logger.info('----{} ({}images) average PSNR/SSIM/LPIPS: {:.2f}/{:.4f}/{:.4f}'.format(
    #     test_set_name, idx, avg_psnr, avg_ssim, avg_lpips))
    logger.info('----{} ({}images) average PSNR/SSIM: {:.2f}/{:.4f}'.format(
        test_set_name, idx, avg_psnr, avg_ssim))
