import torch
import torch.nn as nn
from torchvision import transforms
import PIL
from kornia.color.yuv import rgb_to_yuv, yuv_to_rgb
import numpy as np


class RGBToYUV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        r, g, b = torch.chunk(x, chunks=3, dim=-3)

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        yuv_img = torch.cat((y, u, v), -3)

        return yuv_img


class YUVToRGB(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y, u, v = torch.chunk(x, chunks=3, dim=-3)
        r: torch.Tensor = y + 1.14 * v  # coefficient for g is 0
        g: torch.Tensor = y + -0.396 * u - 0.581 * v
        b: torch.Tensor = y + 2.029 * u  # coefficient for b is 0
        rgb_img: torch.Tensor = torch.cat((r, g, b), -3)
        return rgb_img


preprocess = transforms.Compose([
    transforms.CenterCrop(256),  # XXX: image size is fixed as 512x512
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    RGBToYUV(),
])

resize = transforms.Compose([
    transforms.Resize((64, 64), interpolation=PIL.Image.BICUBIC),
    transforms.Resize((256, 256), interpolation=PIL.Image.BICUBIC),
])

postprocess = transforms.Compose([
    YUVToRGB()
])
