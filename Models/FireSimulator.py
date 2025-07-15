# A Spatio-Temporal Neural Network Forecasting Approach for Emulation of Firefront Models.
# Article from: https://doi.org/10.23919/SPA53010.2022.9927888

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        # decoding
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class OuterNetwork(nn.Module):
    def __init__(self, autoencoder):
        super(OuterNetwork, self).__init__()
        self.autoencoder = autoencoder
        
        # Spatial feature extraction (5-channel input)
        self.spatial_feature_extractor = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Weather feature extraction (6-channel input)
        self.weather_feature_extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

    def forward(self, fire_state, spatial_data, weather_data):
        # Code fire status (1 channel)
        latent_fire = self.autoencoder.encoder(fire_state)

        # Processing spatial data (5 channels)
        spatial_features = self.spatial_feature_extractor(spatial_data)

        # Processing weather data (6 channels)
        batch_size, _, h, w = spatial_features.shape
        weather_features = self.weather_feature_extractor(weather_data)
        weather_features = weather_features.view(batch_size, -1, 1, 1).expand(-1, -1, h, w)

        return latent_fire, spatial_features, weather_features

class InnerNetwork(nn.Module):
    def __init__(self):
        super(InnerNetwork, self).__init__()
        # Adjust the number of input channels to 32 + 64 + 64 = 160
        self.down1 = nn.Conv2d(160, 128, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)

    def forward(self, latent_fire, spatial_features, weather_features):
        # Splicing all features together
        x = torch.cat([latent_fire, spatial_features, weather_features], dim=1)  # [b, 160, 64, 64]

        # U-Net structure
        x1 = F.relu(self.down1(x))  # [b, 128, 32, 32]
        x = F.relu(self.up1(x1))  # [b, 64, 64, 64]
        x = F.relu(self.conv(x))  # [b, 32, 64, 64]

        # Residual connection
        return latent_fire + x

class FireSpreadEmulator(nn.Module):
    def __init__(self):
        super(FireSpreadEmulator, self).__init__()
        self.autoencoder = Autoencoder()
        self.outer_net = OuterNetwork(self.autoencoder)
        self.inner_net = InnerNetwork()

        # Freeze the weight of the autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Split input tensor [b, 12, 256, 256]
        fire_state = x[:, 0:1, :, :]    # Channel 0: Fire status [b, 1, 256, 256]
        spatial_data = x[:, 1:6, :, :]  # 1-5 channels: spatial data [b, 5, 256, 256]
        weather_data = torch.mean(x[:, 6:], dim=[2,3])   # Channel 6-11: Weather data [b, 6] (assuming weather data is spatially uniform)

        # External network processing
        latent_fire, spatial_features, weather_features = self.outer_net(
            fire_state, spatial_data, weather_data)

        # Internal network prediction
        new_latent_fire = self.inner_net(latent_fire, spatial_features, weather_features)

        # Decoding the fire status
        predicted_fire = self.autoencoder.decoder(new_latent_fire)

        return predicted_fire

if __name__ == '__main__':
    model = FireSpreadEmulator()
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