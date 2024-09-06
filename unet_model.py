from collections import OrderedDict
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, initial_features=32):
        super(UNet, self).__init__()

        features = initial_features
        self.encoder1 = self._create_block(in_channels, features, name="encoder1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._create_block(features, features * 2, name="encoder2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._create_block(features * 2, features * 4, name="encoder3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._create_block(features * 4, features * 8, name="encoder4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._create_block(features * 8, features * 16, name="bottleneck")

        self.upsampleconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._create_block((features * 8) * 2, features * 8, name="decoder4")
        self.upsampleconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._create_block((features * 4) * 2, features * 4, name="decoder3")
        self.upsampleconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._create_block((features * 2) * 2, features * 2, name="decoder2")
        self.upsampleconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._create_block(features * 2, features, name="decoder1")

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool1(encoder1))
        encoder3 = self.encoder3(self.pool2(encoder2))
        encoder4 = self.encoder4(self.pool3(encoder3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(encoder4))

        # Decoding
        decoder4 = self.upsampleconv4(bottleneck)
        decoder4 = torch.cat((decoder4, encoder4), dim=1)
        decoder4 = self.decoder4(decoder4)

        decoder3 = self.upsampleconv3(decoder4)
        decoder3 = torch.cat((decoder3, encoder3), dim=1)
        decoder3 = self.decoder3(decoder3)

        decoder2 = self.upsampleconv2(decoder3)
        decoder2 = torch.cat((decoder2, encoder2), dim=1)
        decoder2 = self.decoder2(decoder2)

        decoder1 = self.upsampleconv1(decoder2)
        decoder1 = torch.cat((decoder1, encoder1), dim=1)
        decoder1 = self.decoder1(decoder1)

        # Final Convolution and Sigmoid Activation
        return torch.sigmoid(self.final_conv(decoder1))

    @staticmethod
    def _create_block(in_channels, features, name):
        # A helper method to create blocks in the U-Net
        return nn.Sequential(
            OrderedDict([
                (f"{name}_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}_bn1", nn.BatchNorm2d(features)),
                (f"{name}_relu1", nn.ReLU(inplace=True)),
                (f"{name}_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                (f"{name}_bn2", nn.BatchNorm2d(features)),
                (f"{name}_relu2", nn.ReLU(inplace=True)),
            ])
        )


