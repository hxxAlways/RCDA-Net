import torch
import torch.nn as nn

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # A0 is initialized to 1
        self.A0 = nn.Parameter(torch.FloatTensor(torch.eye(hide_channel)), requires_grad=True)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        nn.init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))
        return x * y

# RAGCA
class Attention(nn.Module):
    def __init__(self, ch, ratio=16):
        super(Attention, self).__init__()
        self.W_g = nn.Conv2d(ch, ch, kernel_size=1)
        self.agca = AGCA(ch, ratio)

    def forward(self, g, x):
        g = self.W_g(g)
        fused = g + x
        # Channel Attention（AGCA）
        attn_c = self.agca(fused)
        # weighting
        return attn_c + x # Residual fusion

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class RAGCA(nn.Module):
    def __init__(self, in_ch=12, num_classes=1, depth=4):
        super(RAGCA, self).__init__()
        assert depth >= 2 and depth <= 5, "Depth must be at between 2 and 5"

        self.depth = depth
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        base_channels = 64
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.attentions = nn.ModuleList()

        # Encoder: Building Depth Layer
        in_ch = in_ch
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(conv_block(in_ch, out_ch))
            in_ch = out_ch

        # Decoder: Depth-1 layer upsampling
        for i in range(depth - 1, 0, -1):
            ch_in = base_channels * (2 ** i)
            ch_out = base_channels * (2 ** (i - 1))

            self.upsamples.append(up_conv(ch_in, ch_out))
            self.attentions.append(Attention(ch_out))
            self.decoders.append(conv_block(ch_in, ch_out))  # 注意concat后的输入是2×ch_out

        self.Conv_1x1 = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc_feats = []

        # Encoding Path
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_feats.append(x)
            if i != self.depth - 1:
                x = self.Maxpool(x)

        # Decoding Path
        for i in range(self.depth - 1):
            x_up = self.upsamples[i](x)
            enc = enc_feats[self.depth - 2 - i]
            attn = self.attentions[i](x_up, enc)
            x = torch.cat((attn, x_up), dim=1)
            x = self.decoders[i](x)

        out = self.Conv_1x1(x)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    model = RAGCA()
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