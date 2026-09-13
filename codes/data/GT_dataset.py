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


class GTDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.noise_level = opt.get('noise_level', 0.0)
        if self.opt['phase'] == 'train':
            self.GT_size = opt['GT_size']
        if isinstance(opt['dataroot_GT'], list):
            GT_paths = [util.get_image_paths(opt['data_type'], dataroot) for dataroot in opt['dataroot_GT']]
            self.GT_paths = sum(GT_paths, [])
        else:
            self.GT_paths = util.get_image_paths(opt['data_type'], opt['dataroot_GT'])  # GT list

    def __getitem__(self, item):
        GT_path = self.GT_paths[item]
        img_GT = util.read_img(env=None, path=GT_path)
        if self.opt['phase'] == 'train':
            # crop
            H, W, C = img_GT.shape
            rnd_top_GT = random.randint(0, max(0, H - self.GT_size))
            rnd_left_GT = random.randint(0, max(0, W - self.GT_size))
            img_GT = img_GT[rnd_top_GT:rnd_top_GT + self.GT_size, rnd_left_GT:rnd_left_GT + self.GT_size, :]
            img_LQ = img_GT
            # augmentation - flip, rotate
            img_GT, img_LQ = util.augment([img_GT, img_LQ], self.opt['use_flip'], self.opt['use_rot'])
        else:
            # generate LQ on-the-fly
            np.random.seed(seed=0)
            img_LQ = img_GT + np.random.normal(0, self.noise_level/255., img_GT.shape)

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
        assert img_GT.size() == img_LQ.size()
        # _, H, W = img_GT.size()
        # img_GT = img_GT[:, :(H // self.scale * self.scale), :(W // self.scale * self.scale)]
        # img_LQ = img_LQ[:, :(H // self.scale * self.scale), :(W // self.scale * self.scale)]
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': GT_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.GT_paths)


# if __name__ == '__main__':
    # L = [[1, 2], [3, 4], [5]]
    # result = sum(L, [])
    # print(result)
