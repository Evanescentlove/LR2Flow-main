import torch
import utils.utils_image as util


def test_one_split(model, img_L, noise_level, refield=64):
    bs, ch, H, W = img_L.shape

    top = slice(0, (H // 2 // refield + 1) * refield)
    bottom = slice(H - (H // 2 // refield + 1) * refield, H)
    left = slice(0, (W // 2 // refield + 1) * refield)
    right = slice(W - (W // 2 // refield + 1) * refield, W)

    Ls = [img_L[..., top, left], img_L[..., top, right],
          img_L[..., bottom, left], img_L[..., bottom, right]]
    Es = [model(L, noise_level) for L in Ls]

    E = torch.zeros_like(img_L)
    E[..., :(H // 2), :(W // 2)] = Es[0][..., :(H // 2), :(W // 2)]
    E[..., :(H // 2), (W // 2):] = Es[1][..., :(H // 2), (-W + W // 2):]
    E[..., (H // 2):, :(W // 2)] = Es[2][..., (-H + H // 2):, :(W // 2)]
    E[..., (H // 2):, (W // 2):] = Es[3][..., (-H + H // 2):, (-W + W // 2):]
    return E


def test_split_x8(model, L, noise_level, refield=32):
    E_list = [test_one_split(model, util.augment_img_tensor(L, mode=i), noise_level=noise_level, refield=refield) for i in range(8)]
    for k, i in enumerate(range(len(E_list))):
        if i==3 or i==5:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=8-i)
        else:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E
