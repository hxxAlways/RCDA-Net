import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, norm_cfg=None, act_cfg=None):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        if norm_cfg:
            if norm_cfg['type'] == 'BN':
                layers.append(nn.BatchNorm2d(out_channels, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5)))
        if act_cfg:
            if act_cfg['type'] == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif act_cfg['type'] == 'SiLU':
                layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CAA(nn.Module):
    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.max_pool = nn.MaxPool2d(7, 1, 3)

        self.conv1 = ConvModule(channels * 2, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        pooled = torch.cat([avg, max_], dim=1)  # B x 2C x 7 x 7

        attn = self.conv1(pooled)
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.conv2(attn)
        attn = self.act(attn)
        return attn

# CAA
class Attention(nn.Module):
    def __init__(self, ch):
        super(Attention, self).__init__()
        self.W_g = nn.Conv2d(ch, ch, kernel_size=1)
        self.caa = CAA(ch)

    def forward(self, g, x):
        g = self.W_g(g)
        fused = g + x
        # Spatial Attention（CAA）
        attn_s = self.caa(fused)
        # weighting
        return x * attn_s + x # Residual fusion

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

class RCA(nn.Module):
    def __init__(self, in_ch=12, num_classes=1, depth=4):
        super(RCA, self).__init__()
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
            self.decoders.append(conv_block(ch_in, ch_out))

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
    model = RCA()
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