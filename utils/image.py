import torch


def grayscale_to_3_channel(I):
    I = torch.stack((I[0][0],) * 3, dim=-1)
    return I

