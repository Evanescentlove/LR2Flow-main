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
from tqdm import tqdm
import cv2


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to options YMAL file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    device_id = torch.cuda.current_device()

    # mkdir and logger
    util.mkdirs((path for key, path in opt["path"].items() if not key == "experiments_root"
                 and "pretrain_model" not in key and "resume" not in key and "load_submodule" not in key))
    util.setup_logger("base", opt["path"]["log"], "test_" + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # set random seed
    util.set_random_seed(0)

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info(f"Number of test images in [{dataset_opt['name']:s}]: {len(test_set):d}")
        test_loaders.append(test_loader)

    # load pretrained model
    model = create_model(opt)
    loss_fn_alex = lpips.LPIPS(net="alex").to("cuda")
    crop_boarder = opt["crop_boarder"] if opt["crop_boarder"] else opt["scale"]

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]
        num_imgs = len(test_loader.dataset)
        logger.info(f"\n\nTesting [{test_set_name:s}]...")
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
        util.mkdir(dataset_dir)

        psnr_y_dict = {quality: 0 for quality in opt["val"]["comp_quality"]}
        ssim_y_dict = {quality: 0 for quality in opt["val"]["comp_quality"]}
        lpips_dict = {quality: 0 for quality in opt["val"]["comp_quality"]}
        bpp_dict = {quality: 0 for quality in opt["val"]["comp_quality"]}

        for test_data in tqdm(test_loader):
            img_path = test_data["GT_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.feed_data(test_data)
            model.test()
            visuals = model.get_current_visuals()

            for quality in opt["val"]["comp_quality"]:
                gt_img = visuals["GT"]
                sr_img = visuals[("SR", quality)]
                lr_img = visuals[("LQ_fromH", quality)]

                # print(f"gt_img type:{type(gt_img)}")
                # print(f"sr_img type:{type(sr_img)}")
                lpips_dict[quality] += float(loss_fn_alex(2 * gt_img.to("cuda") - 1, 2 * sr_img.to("cuda") - 1).cpu())

                # calculate bpp
                save_lr_path = os.path.join(dataset_dir, img_name + "_LR.jpg")
                lr_img = util.tensor2img(lr_img)
                cv2.imwrite(save_lr_path, lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                lr_size = os.path.getsize(save_lr_path)  # get lr image size
                _, h, w = gt_img.shape
                bpp = lr_size * 8 / (h * w)
                bpp_dict[quality] += bpp

                # tensor to uint8
                gt_img = util.tensor2img(gt_img)
                sr_img = util.tensor2img(sr_img)

                _, _, psnr_y, ssim_y = util.calculate_psnr_ssim(gt_img/255, sr_img/255, crop_boarder)
                psnr_y_dict[quality] += psnr_y
                ssim_y_dict[quality] += ssim_y

        logger.info("-------------------------------------------------------------------------------------")
        for quality in opt["val"]["comp_quality"]:
            avg_psnr_y = psnr_y_dict[quality] / num_imgs
            avg_ssim_y = ssim_y_dict[quality] / num_imgs
            avg_lpips = lpips_dict[quality] / num_imgs
            avg_bpp = bpp_dict[quality] / num_imgs

            # log
            logger.info(f"----{test_set_name} ({num_imgs} images, quality {quality:d}) "
                        f"average PSNR_Y/SSIM_Y/LPIPS/BPP: {avg_psnr_y:.2f}/{avg_ssim_y:.4f}/{avg_lpips:.4f}/{avg_bpp:.4f}")


if __name__ == '__main__':
    main()
