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


class GTLQDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt['phase'] == 'train':
            self.GT_size = opt['GT_size']
        self.LQ_paths = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])  # LQ list
        self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list
        assert len(self.LQ_paths) == len(self.GT_paths), 'GT and LQ datasets have different number of images - {}, {}.'.format(len(self.LQ_paths), len(self.GT_paths))

    def __getitem__(self, index):
        GT_path = self.GT_paths[index]
        LQ_path = self.LQ_paths[index]
        img_GT = util.read_img(env=None, path=GT_path)
        img_LQ = util.read_img(env=None, path=LQ_path)

        if self.opt['phase'] == 'train':
            # crop
            H, W, C = img_GT.shape
            rnd_top_GT = random.randint(0, max(0, H - self.GT_size))
            rnd_left_GT = random.randint(0, max(0, W - self.GT_size))
            rnd_top_LQ = rnd_top_GT
            rnd_left_LQ = rnd_left_GT

            img_GT = img_GT[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_LQ = img_LQ[rnd_top_LQ:rnd_top_LQ + self.GT_size, rnd_left_LQ:rnd_left_LQ + self.GT_size, :]

            # augmentation - flip, rotate
            img_GT, img_LQ = util.augment([img_GT, img_LQ], self.opt['use_flip'], self.opt['use_rot'])

        # change color space if necessary, deal with gray image
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        # modcrop
        _, H, W = img_GT.size()
        img_LQ = img_LQ[:, :H // 8 * 8, :W // 8 * 8]
        img_GT = img_GT[:, :H // 8 * 8, :W // 8 * 8]

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)

