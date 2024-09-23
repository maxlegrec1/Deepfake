import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, latent_dim, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.leaky_relu(self.conv4(x), 0.1)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            latent_dim, 256, kernel_size=4, stride=2, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(
            64, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        x = F.leaky_relu(self.deconv1(x), 0.1)
        x = F.leaky_relu(self.deconv2(x), 0.1)
        x = F.leaky_relu(self.deconv3(x), 0.1)
        x = torch.sigmoid(self.deconv4(x))
        return x


class Inter(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.dense1 = nn.Linear(latent_dim * 8 * 8, latent_dim)
        self.dense2 = nn.Linear(latent_dim, latent_dim * 8 * 8)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1)
        x = F.leaky_relu(self.dense1(x), 0.1)
        x = F.leaky_relu(self.dense2(x), 0.1)
        return x.view(b, c, h, w)


class LIAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.inter_AB = Inter(latent_dim)
        self.inter_B = Inter(latent_dim)
        self.decoder = Decoder(latent_dim * 2, in_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        inter_AB = self.inter_AB(encoded)
        inter_B = self.inter_B(encoded)
        latent = torch.cat([inter_AB, inter_B], dim=1)
        return self.decoder(latent)


# Usage
model = LIAE()
x = torch.randn(1, 3, 128, 128)
output = model(x)
print(output.shape)  # Should be torch.Size([1, 3, 128, 128])
