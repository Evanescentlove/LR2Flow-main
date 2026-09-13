# LR2Flow: Enhancing Low-resolution Image Representation Through Normalizing Flows

Official PyTorch implementation of **LR2Flow**, presented in:

> **Enhancing Low-resolution Image Representation Through Normalizing Flows**
> Chenglong Bao, Tongyao Pang, Zuowei Shen, Dihan Zheng, and Yihang Zou
> arXiv preprint arXiv:2601.06834, 2026.

[[Paper](https://arxiv.org/abs/2601.06834)]

## Pretrained Models

Pretrained models are available from [Google Drive](https://drive.google.com/drive/folders/1Bt3OrebG3D4JPB80j_p-PBVu26ma21Kr?usp=sharing).

Please download the corresponding `.pth` files and place them in:

```text
./experiments/pretrained_models/
```

## Configuration

Before training or testing, please configure the corresponding `.yml` files located in:

```text
./codes/options/train/
./codes/options/test/
```

The dataset paths, pretrained model paths, and other experimental settings can be specified in these configuration files.

## Testing

Run the following commands under the `./codes/` directory.

### Image Rescaling

**×2 image rescaling**

```bash
python3 test_rescaling.py --opt options/test/test_rescaling_x2.yml
```

**×4 image rescaling**

```bash
python3 test_rescaling.py --opt options/test/test_rescaling_x4.yml
```

### Image Compression

```bash
python3 test_compression.py --opt options/test/test_compression_x2.yml
```

Alternatively, to evaluate the compression performance in terms of bits per pixel (bpp), run:

```bash
python3 test_compression_bpp.py --opt options/test/test_compression_x2.yml
```

### Image Denoising

```bash
python3 test_denoising.py --opt options/test/test_denoising.yml
```

## Training

Run the following commands under the `./codes/` directory.

### Image Rescaling

**×2 image rescaling**

```bash
python3 train_rescaling.py --opt options/train/train_rescaling_x2.yml
```

**×4 image rescaling**

```bash
python3 train_rescaling.py --opt options/train/train_rescaling_x4.yml
```

### Image Compression

```bash
python3 train_compression.py --opt options/train/train_compression_x2.yml
```

### Image Denoising

```bash
python3 train_denoising.py --opt options/train/train_denoising.yml
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{bao2026enhancing,
  title   = {Enhancing Low-resolution Image Representation Through Normalizing Flows},
  author  = {Bao, Chenglong and Pang, Tongyao and Shen, Zuowei and Zheng, Dihan and Zou, Yihang},
  journal = {arXiv preprint arXiv:2601.06834},
  year    = {2026}
}
```

## Acknowledgements

This codebase is built upon [HCFlow](https://github.com/JingyunLiang/HCFlow) and is also inspired by [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling) and [BasicSR](https://github.com/XPixelGroup/BasicSR).
