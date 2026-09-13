import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    from utils import util as utils
except ImportError:
    pass


class GTLQxDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt['scale']
        if self.opt['phase'] == 'train':
            self.GT_size = opt['GT_size']
            self.LR_size = self.GT_size // self.scale
        self.LR_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])  # LR list
        self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        assert len(self.LR_paths) == len(self.GT_paths), 'GT and LR datasets have different number of images - {}, {}.'.format(len(self.LR_paths), len(self.GT_paths))

    def __getitem__(self, index):
        GT_path = self.GT_paths[index]
        LR_path = self.LR_paths[index]
        img_GT = util.read_img(env=None, path=GT_path)
        img_LR = util.read_img(env=None, path=LR_path)

        if self.opt['phase'] == 'train':
            # crop
            H, W, C = img_LR.shape
            rnd_top_LR = random.randint(0, max(0, H - self.LR_size))
            rnd_left_LR = random.randint(0, max(0, W - self.LR_size))
            rnd_top_GT = rnd_top_LR * self.scale
            rnd_left_GT = rnd_left_LR * self.scale

            img_GT = img_GT[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_LR = img_LR[rnd_top_LR:rnd_top_LR + self.LR_size, rnd_left_LR:rnd_left_LR + self.LR_size, :]

            # augmentation - flip, rotate
            img_GT, img_LR = util.augment([img_GT, img_LR], self.opt['use_flip'], self.opt['use_rot'])

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LR = util.channel_convert(img_LR.shape[2], self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        # modcrop
        _, H, W = img_LR.size()
        img_GT = img_GT[:, :H * self.scale, :W * self.scale]

        return {'LQ': img_LR, 'GT': img_GT, 'LQ_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)

