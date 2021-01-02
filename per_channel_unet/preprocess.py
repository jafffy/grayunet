import torch
from torchvision import transforms
import PIL

preprocess = transforms.Compose([
    transforms.CenterCrop(512),  # XXX: image size is fixed as 512x512
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

resize = transforms.Compose([
    transforms.Resize((128, 128), interpolation=PIL.Image.BICUBIC),
    transforms.Resize((512, 512), interpolation=PIL.Image.BICUBIC),
])

