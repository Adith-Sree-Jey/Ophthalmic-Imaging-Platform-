"""
Baseline models for retinal vessel segmentation comparison (research paper).
- UNet: plain U-Net (Ronneberger-style), no residuals.
- ResUNet: U-Net with residual blocks in encoder/decoder.
- R2UNet: Recurrent Residual U-Net (recurrent conv in each block).
All use same interface: in_channels=3, out_channels=1, arbitrary spatial size.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two conv layers (plain U-Net)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(ConvBlock(in_ch, out_ch), ConvBlock(out_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block: x + conv2(conv1(x))."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch)
        self.conv2 = ConvBlock(ch, ch)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class RecurrentBlock(nn.Module):
    """Recurrent block (R2U-Net): run same conv t times with residual."""
    def __init__(self, ch, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(ConvBlock(ch, ch), ConvBlock(ch, ch))

    def forward(self, x):
        for _ in range(self.t):
            x = x + self.conv(x)
        return x


class UNet(nn.Module):
    """Plain U-Net: encoder-decoder, no residuals."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class ResUNet(nn.Module):
    """ResUNet: U-Net with residual blocks (same topology as DeepVesselNet)."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(in_channels, 32), ResidualBlock(32))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvBlock(32, 64), ResidualBlock(64))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvBlock(64, 128), ResidualBlock(128))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(128, 256), ResidualBlock(256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(256, 128), ResidualBlock(128))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(128, 64), ResidualBlock(64))
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(64, 32), ResidualBlock(32))
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


class R2UNet(nn.Module):
    """R2U-Net: Recurrent Residual U-Net (t=2 recurrent steps per block)."""
    def __init__(self, in_channels=3, out_channels=1, t=2):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(in_channels, 32), RecurrentBlock(32, t=t))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvBlock(32, 64), RecurrentBlock(64, t=t))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvBlock(64, 128), RecurrentBlock(128, t=t))
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(ConvBlock(128, 256), RecurrentBlock(256, t=t))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(256, 128), RecurrentBlock(128, t=t))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(128, 64), RecurrentBlock(64, t=t))
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(64, 32), RecurrentBlock(32, t=t))
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


def get_baseline_model(name, in_channels=3, out_channels=1):
    """Return baseline by name: UNet, ResUNet, R2UNet."""
    n = name.strip().lower()
    if n == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    if n == "resunet":
        return ResUNet(in_channels=in_channels, out_channels=out_channels)
    if n == "r2unet":
        return R2UNet(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unknown baseline: {name}. Use UNet, ResUNet, or R2UNet.")


if __name__ == "__main__":
    for name in ["UNet", "ResUNet", "R2UNet"]:
        m = get_baseline_model(name)
        x = torch.randn(1, 3, 256, 256)
        y = m(x)
        assert y.shape == (1, 1, 256, 256), y.shape
        print(f"{name}: OK, shape={y.shape}")
