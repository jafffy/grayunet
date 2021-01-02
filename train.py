import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random

from per_channel_unet.unet import UNet
from per_channel_unet.preprocess import preprocess, resize
from utils.image import grayscale_to_3_channel

import sys


def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")

    writer = SummaryWriter()

    if len(sys.argv) == 2:
        model = torch.load(sys.argv[1]).to('cuda')
    else:
        model = UNet(n_channels=1, tile_size=512).to('cuda')

    lr = 1e-6
    idx = 0

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for t in range(10000):
        data = torchvision.datasets.ImageFolder(root='G:/workspace/cheap-dlss-torch/media/wow', transform=preprocess)
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=2)

        pool = []
        pool_count = 0
        pool_exponent = 0

        for (image_batch, _) in data_loader:
            loss_mean = 0
            loss_var = 0

            window_idx = 0

            if 2 ** pool_exponent > pool_count:
                pool.append(image_batch)
                pool_count += 1
                continue

            while True:
                image_batch = random.choice(pool)

                lowres_batch = resize(image_batch).to('cuda')
                answer_batch = image_batch.to('cuda')

                y_pred = model(lowres_batch, answer_batch)

                loss = criterion(y_pred, answer_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                writer.add_scalar('wow_batch_2_loss_overfit', loss.item(), idx)

                loss_val = loss.item()

                loss_mean += loss_val
                loss_var += loss_val * loss_val
                window_idx += 1

                if window_idx > 100:
                    loss_mean /= window_idx
                    loss_var /= window_idx
                    loss_var -= loss_mean * loss_mean
                    window_idx = 0

                    print(idx, len(pool), loss_mean, loss_var)

                    if loss_mean < 10 or loss_var < 10:
                        writer.add_image('wow_image_batch_pred', grayscale_to_3_channel(y_pred),
                                         global_step=idx,
                                         dataformats='HWC')
                        writer.add_image('wow_image_batch_answer', grayscale_to_3_channel(answer_batch),
                                         global_step=idx,
                                         dataformats='HWC')
                        break
                    else:
                        loss_mean = 0
                        loss_var = 0

                    torch.save(model.state_dict(), f'wow_batch_2_{idx}.state_dict')
                    torch.save(model, f'wow_batch_2_{idx}.model')

                idx += 1

            print(idx, loss.item())

            pool_exponent += 1
            pool_count = 0
            pool.clear()


if __name__ == '__main__':
    main()