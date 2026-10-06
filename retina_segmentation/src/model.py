import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual Block with two conv layers"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DeepVesselNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(DeepVesselNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, 32),
            ResidualBlock(32)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(32, 64),
            ResidualBlock(64)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(64, 128),
            ResidualBlock(128)
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(128, 256),
            ResidualBlock(256)
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(256, 128),
            ResidualBlock(128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(128, 64),
            ResidualBlock(64)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(64, 32),
            ResidualBlock(32)
        )

        # Output layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        cat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(cat3)

        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)

        return torch.sigmoid(self.out_conv(d1))


if __name__ == "__main__":
    # Test the model
    model = DeepVesselNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Batch size 1, RGB, 256x256
    y = model(x)
    print("Output shape:", y.shape)  # Expect [1, 1, 256, 256]