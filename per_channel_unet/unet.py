import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bicubic')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Injection(nn.Module):
    def __init__(self, n_channels, tile_size):
        super().__init__()

        self.n_channels = n_channels

        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=(3, 3))
        self.sigmoid = nn.Sigmoid()
        self.fc_inj1 = nn.Linear(tile_size - 2, tile_size)
        self.fc_inj2 = nn.Linear(tile_size - 2, tile_size)

    def forward(self, x):
        x = torch.transpose(x, 2, 3)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = self.fc_inj1(x)
        x = torch.transpose(x, 2, 3)
        x = self.fc_inj2(x)

        return x


class UNet(nn.Module):
    def __init__(self, n_channels, tile_size):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.tile_size = tile_size

        self.fc = nn.Linear(self.tile_size, self.tile_size)

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 1024)

        self.up0 = Up(2048, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 32)
        self.outc = OutConv(32, self.n_channels)

        self.injection = Injection(n_channels=n_channels, tile_size=tile_size)

    def forward(self, x, inj):
        skip_connection = inj

        inj = self.injection(inj)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x = torch.add(x, inj)
        x = torch.add(x, skip_connection)

        return self.fc(x)


class YUVUNet(nn.Module):
    def __init__(self, tile_size):
        super().__init__()

        self.y_unet = UNet(1, tile_size)
        self.u_unet = UNet(1, tile_size)
        self.v_unet = UNet(1, tile_size)

    def forward(self, x, inj):
        x_y, x_u, x_v = torch.chunk(x, chunks=3, dim=-3)
        inj_y, inj_u, inj_v = torch.chunk(inj, chunks=3, dim=-3)

        x_y = self.y_unet(x_y, inj_y)
        x_u = self.u_unet(x_u, inj_u)
        x_v = self.v_unet(x_v, inj_v)

        return torch.cat([x_y, x_u, x_v], -3)
