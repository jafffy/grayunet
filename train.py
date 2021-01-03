import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import random

from per_channel_unet.unet import YUVUNet, UNet
from per_channel_unet.preprocess import preprocess, resize, postprocess

import sys


def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")

    writer = SummaryWriter()

    if False:
        y_model = UNet(n_channels=1, tile_size=256).to('cuda')
        u_model = UNet(n_channels=1, tile_size=256).to('cuda')
        v_model = UNet(n_channels=1, tile_size=256).to('cuda')
    else:
        y_model = torch.load('y_wow_batch_2_5400.model').to('cuda')
        u_model = torch.load('u_wow_batch_2_5400.model').to('cuda')
        v_model = torch.load('v_wow_batch_2_5400.model').to('cuda')

    lr = 1e-6
    idx = 0

    criterion = nn.MSELoss(reduction='sum')
    y_optimizer = optim.SGD(y_model.parameters(), lr=lr, momentum=0.9)
    u_optimizer = optim.SGD(u_model.parameters(), lr=lr, momentum=0.9)
    v_optimizer = optim.SGD(v_model.parameters(), lr=lr, momentum=0.9)

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

                y_lowres, u_lowres, v_lowres = torch.chunk(lowres_batch, chunks=3, dim=-3)
                y_answer, u_answer, v_answer = torch.chunk(answer_batch, chunks=3, dim=-3)

                y_pred = y_model(y_lowres, y_answer)
                u_pred = u_model(u_lowres, u_answer)
                v_pred = v_model(v_lowres, v_answer)

                # y_pred = torch.clamp(y_pred, min=0.0, max=1.0)
                # u_pred = torch.clamp(u_pred, min=0.0, max=1.0)
                # v_pred = torch.clamp(v_pred, min=0.0, max=1.0)

                y_loss = criterion(y_pred, y_answer)
                u_loss = criterion(u_pred, u_answer)
                v_loss = criterion(v_pred, v_answer)

                y_optimizer.zero_grad()
                u_optimizer.zero_grad()
                v_optimizer.zero_grad()

                y_loss.backward()
                u_loss.backward()
                v_loss.backward()

                y_optimizer.step()
                u_optimizer.step()
                v_optimizer.step()

                writer.add_scalar('y_wow_batch_2_loss_overfit', y_loss.item(), idx)
                writer.add_scalar('u_wow_batch_2_loss_overfit', u_loss.item(), idx)
                writer.add_scalar('v_wow_batch_2_loss_overfit', v_loss.item(), idx)

                loss_val = y_loss.item() + u_loss.item() + v_loss.item()
                writer.add_scalar('wow_batch_2_loss_overfit', loss_val, idx)

                loss_mean += loss_val
                loss_var += loss_val * loss_val
                window_idx += 1

                if window_idx > 100:
                    loss_mean /= window_idx
                    loss_var /= window_idx
                    loss_var -= loss_mean * loss_mean
                    window_idx = 0

                    print(idx, len(pool), loss_mean, loss_var)

                    torch.save(y_model, f'y_wow_batch_2_{idx}.model')
                    torch.save(y_model, f'u_wow_batch_2_{idx}.model')
                    torch.save(y_model, f'v_wow_batch_2_{idx}.model')

                    if loss_mean < 20 or loss_var < 10:
                        pred = torch.cat([y_pred, u_pred, v_pred], -3)
                        writer.add_image('wow_image_batch_pred',
                                         torch.clamp(postprocess(pred[0]), min=0.0, max=1.0),
                                         global_step=idx,
                                         dataformats='CHW')
                        writer.add_image('wow_image_batch_answer',
                                         torch.clamp(postprocess(answer_batch[0]), min=0.0, max=1.0),
                                         global_step=idx,
                                         dataformats='CHW')
                        break
                    else:
                        loss_mean = 0
                        loss_var = 0

                idx += 1

            print(idx, loss_val)

            if pool_exponent < 9:
                pool_exponent += 1
            else:
                pool_exponent = 0
            pool_count = 0
            pool.clear()


if __name__ == '__main__':
    main()
