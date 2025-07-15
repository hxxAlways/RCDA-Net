# Forest Fire Spread Prediction Using Deep Learning.
# Article from: https://doi.org/10.1117/12.2585997

import torch
import torch.nn as nn

class FUNetCast(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super(FUNetCast, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 128)  # 10 -> 128, 64x64
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        self.enc2 = self.conv_block(128, 256)  # 128 -> 256, 32x32
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.enc3 = self.conv_block(256, 512)  # 256 -> 512, 16x16
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # bottleneck
        self.bottleneck = self.conv_block(512, 1024)  # 512 -> 1024, 8x8

        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec3 = self.conv_block(1024, 512)  # 拼接 512+512 -> 512
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec2 = self.conv_block(512, 256)  # 拼接 256+256 -> 256
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32x32 -> 64x64
        self.dec1 = self.conv_block(256, 128)  # 拼接 128+128 -> 128

        # Last layer
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)  # 128 -> 1

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # encoding
        e1 = self.enc1(x)  # 128x64x64
        e2 = self.pool1(e1)  # 128x32x32
        e2 = self.enc2(e2)  # 256x32x32
        e3 = self.pool2(e2)  # 256x16x16
        e3 = self.enc3(e3)  # 512x16x16
        e4 = self.pool3(e3)  # 512x8x8

        # bottleneck
        b = self.bottleneck(e4)  # 1024x8x8

        # decoding
        d3 = self.upconv3(b)  # 512x16x16
        d3 = torch.cat((d3, e3), dim=1)  # 拼接：512+512 -> 1024x16x16
        d3 = self.dec3(d3)  # 512x16x16
        d2 = self.upconv2(d3)  # 256x32x32
        d2 = torch.cat((d2, e2), dim=1)  # 拼接：256+256 -> 512x32x32
        d2 = self.dec2(d2)  # 256x32x32
        d1 = self.upconv1(d2)  # 128x64x64
        d1 = torch.cat((d1, e1), dim=1)  # 拼接：128+128 -> 256x64x64
        d1 = self.dec1(d1)  # 128x64x64

        out = self.final(d1)  # 1x64x64
        return torch.sigmoid(out)

if __name__ == '__main__':
    model = FUNetCast(in_channels=12, out_channels=1)
    # print(model)
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)

    # Generate random inputs data (matching paper's specs)
    batch_size = 4  # Batch of 4 samples
    input_channels = 12  # 12 inputs channels (terrain, wind, moisture, etc.)
    height, width = 256, 256  # 256x256 pixels (as per paper)

    # Random inputs tensor (simulating 12-channel inputs image)
    x = torch.randn(batch_size, input_channels, height, width)
    # Forward pass
    output = model(x)

    # Check shapes
    print(f"Input shape: {x.shape}")  # Expected: [4, 12, 256, 256]
    print(f"Output shape: {output.shape}")  # Expected: [4, 1, 256, 256]