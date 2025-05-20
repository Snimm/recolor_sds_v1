#@title Generator
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_channels):
        super().__init__()

        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_channels, 1)
        nn.BatchNorm2d(n_channels)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = torch.tanh(self.conv_last(x))
        # out = torch.sigmoid(self.conv_last(x))

        return out




# class ShadingUNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()

#         self.dconv_down1 = double_conv(3, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(256 + 512, 256)
#         self.dconv_up2 = double_conv(128 + 256, 128)
#         self.dconv_up1 = double_conv(128 + 64, 64)

#         self.conv_last = nn.Conv2d(64, n_channels, 1)


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x = self.dconv_up3(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv2], dim=1)

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv1], dim=1)

#         x = self.dconv_up1(x)

#         # out = torch.tanh(self.conv_last(x))
#         out = 0.5 + torch.sigmoid(self.conv_last(x))
#         # out = 1. - torch.tanh(self.conv_last(x))
#         # out = self.conv_last(x)
# #         return out.repeat([1, 3, 1, 1])


# class ShadingUNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()

#         self.dconv_down1 = double_conv(n_channels, 16)
#         self.dconv_down2 = double_conv(16, 32)
#         self.dconv_down3 = double_conv(32, 64)
#         self.dconv_down4 = double_conv(64, 128)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(64 + 128, 64)
#         self.dconv_up2 = double_conv(32 + 64, 32)
#         self.dconv_up1 = double_conv(32 + 16, 16)

#         self.conv_last = nn.Conv2d(16, 1, 1)
#         self.n_channels = n_channels


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x = self.dconv_up3(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv2], dim=1)

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv1], dim=1)

#         x = self.dconv_up1(x)
#         out = 2 * torch.sigmoid(self.conv_last(x))
#         return out.repeat([1, self.n_channels, 1, 1])


# class AlphaUNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()
#         self.dconv_down1 = double_conv(n_channels, 16)
#         self.dconv_down2 = double_conv(16, 32)
#         self.dconv_down3 = double_conv(32, 64)
#         self.dconv_down4 = double_conv(64, 128)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = double_conv(64 + 128, 64)
#         self.dconv_up2 = double_conv(32 + 64, 32)
#         self.dconv_up1 = double_conv(32 + 16, 16)

#         self.conv_last = nn.Conv2d(16, n_channels + 1, 1)
#         self.n_channels = n_channels


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x = self.dconv_up3(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv2], dim=1)

#         x = self.dconv_up2(x)
#         x = self.upsample(x)
#         x = torch.cat([x, conv1], dim=1)

#         x = self.dconv_up1(x)
#         out = self.conv_last(x)

#         out_layer = torch.sigmoid(out[:, :self.n_channels])
#         alpha_layer = torch.sigmoid(out[:, -1])
#         return out_layer, alpha_layer





# class TwoLayerUNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()

#         self.dconv_down1 = double_conv(n_channels, 16)
#         self.dconv_down2 = double_conv(16, 32)
#         self.dconv_down3 = double_conv(32, 64)
#         self.dconv_down4 = double_conv(64, 128)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3_mult = double_conv(64 + 128, 64)
#         self.dconv_up2_mult = double_conv(32 + 64, 32)
#         self.dconv_up1_mult = double_conv(32 + 16, 16)

#         self.dconv_up3_div = double_conv(64 + 128, 64)
#         self.dconv_up2_div = double_conv(32 + 64, 32)
#         self.dconv_up1_div = double_conv(32 + 16, 16)

#         self.conv_last_mult = nn.Conv2d(16, 1, 1)
#         self.conv_last_div = nn.Conv2d(16, 1, 1)
#         self.n_channels = n_channels


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x_mult = self.dconv_up3_mult(x)
#         x_mult = self.upsample(x_mult)
#         x_mult = torch.cat([x_mult, conv2], dim=1)

#         x_div = self.dconv_up3_div(x)
#         x_div = self.upsample(x_div)
#         x_div = torch.cat([x_div, conv2], dim=1)

#         x_mult = self.dconv_up2_mult(x_mult)
#         x_mult = self.upsample(x_mult)
#         x_mult = torch.cat([x_mult, conv1], dim=1)

#         x_div = self.dconv_up2_div(x_div)
#         x_div = self.upsample(x_div)
#         x_div = torch.cat([x_div, conv1], dim=1)

#         x_mult = self.dconv_up1_mult(x_mult)
#         x_div = self.dconv_up1_div(x_div)

#         # out = torch.sigmoid(self.conv_last(x))
#         out_mult = torch.sigmoid(self.conv_last_mult(x_mult))
#         out_div = torch.sigmoid(self.conv_last_div(x_div))
#         out_div = torch.clip(out_div, 0.1, 1.)
#         out_mult = torch.clip(out_mult, 0.1, 1.)
#         return out_mult.repeat([1, self.n_channels, 1, 1]), out_div.repeat([1, self.n_channels, 1, 1])
    
    
    
    
    
# class Colorization_layer_UNet(nn.Module):

#     def __init__(self, n_channels):
#         super().__init__()

#         self.dconv_down1 = double_conv(n_channels, 16)
#         self.dconv_down2 = double_conv(16, 32)
#         self.dconv_down3 = double_conv(32, 64)
#         self.dconv_down4 = double_conv(64, 128)

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3_A = double_conv(64 + 128, 64)
#         self.dconv_up2_A = double_conv(32 + 64, 32)
#         self.dconv_up1_A = double_conv(32 + 16, 16)

#         self.dconv_up3_B = double_conv(64 + 128, 64)
#         self.dconv_up2_B = double_conv(32 + 64, 32)
#         self.dconv_up1_B = double_conv(32 + 16, 16)

#         self.conv_last_A = nn.Conv2d(16, 1, 1)
#         self.conv_last_B = nn.Conv2d(16, 1, 1)
#         self.n_channels = n_channels


#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)

#         x_A = self.dconv_up3_A(x)
#         x_A = self.upsample(x_A)
#         x_A = torch.cat([x_A, conv2], dim=1)

#         x_B = self.dconv_up3_B(x)
#         x_B = self.upsample(x_B)
#         x_B = torch.cat([x_B, conv2], dim=1)

#         x_A = self.dconv_up2_A(x_A)
#         x_A = self.upsample(x_A)
#         x_A = torch.cat([x_A, conv1], dim=1)

#         x_B = self.dconv_up2_B(x_B)
#         x_B = self.upsample(x_B)
#         x_B = torch.cat([x_B, conv1], dim=1)

#         x_A = self.dconv_up1_A(x_A)
#         x_B = self.dconv_up1_B(x_B)

#         # out = torch.sigmoid(self.conv_last(x))
#         out_A = torch.sigmoid(self.conv_last_A(x_A))
#         out_B = torch.sigmoid(self.conv_last_B(x_B))
#         out_B = torch.clip(out_B, 0.1, 1.)
#         out_A = torch.clip(out_A, 0.1, 1.)
#         return out_A, out_B
    
class ColorizationUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2): # Input L (1 ch), output A, B (2 ch)
        super().__init__()
        self.n_channels = in_channels

        # Encoder
        self.dconv_down1 = double_conv(self.n_channels, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128) # Bottleneck

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.dconv_up3 = double_conv(64 + 128, 64) # Cat bottleneck + conv3_out
        self.dconv_up2 = double_conv(32 + 64, 32)  # Cat conv3_up_out + conv2_out
        self.dconv_up1 = double_conv(16 + 32, 16)  # Cat conv2_up_out + conv1_out

        self.conv_last = nn.Conv2d(16, out_channels, 1) # Output 2 channels for A and B

    def forward(self, x):
        print("Input shape to unet generator:", x.shape)
        # x is expected to be the L channel, shape (B, 1, H, W)
        conv1 = self.dconv_down1(x)
        x_pool1 = self.maxpool(conv1)

        conv2 = self.dconv_down2(x_pool1)
        x_pool2 = self.maxpool(conv2)

        conv3 = self.dconv_down3(x_pool2)
        x_pool3 = self.maxpool(conv3)

        x_bottom = self.dconv_down4(x_pool3) # Bottleneck

        x_up3 = self.upsample(x_bottom)
        # Ensure consistent channel numbers for concatenation if original image size is not a power of 2
        # For now, assuming standard U-Net skip connections
        x_up3 = torch.cat([x_up3, conv3], dim=1) 
        x_up3 = self.dconv_up3(x_up3)

        x_up2 = self.upsample(x_up3)
        x_up2 = torch.cat([x_up2, conv2], dim=1)
        x_up2 = self.dconv_up2(x_up2)

        x_up1 = self.upsample(x_up2)
        x_up1 = torch.cat([x_up1, conv1], dim=1)
        x_up1 = self.dconv_up1(x_up1)

        out_ab = self.conv_last(x_up1)
        # Output A and B channels. Use tanh to get values in [-1, 1].
        # These will be scaled later to the actual Lab A,B range (e.g., approx -110 to 110).
        return torch.tanh(out_ab)