"""
Date: April-1-2025
Author: Yingqi

This code is to construct the base U-Net in UltRAP-Net

"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    basic double convolution layer: conv+batch_norm+activ+conv+batch_norm+activ
    """

    def __init__(self, in_channel, out_channel, kernel=3):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class InConv(nn.Module):
    """
    1st layer for input: only double conv
    """

    def __init__(self, in_channel, out_channel, kernel=3):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel, kernel)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    """
    basic module in encoder: MaxPool2d + DoubleConv
    """

    def __init__(self, in_channel, out_channel, kernel=3):
        super(DownSample, self).__init__()
        self.down_sample = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channel, out_channel, kernel)

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """
    basic module in decoder: Upsample or ConvTranspose2d + DoubleConv
    """

    def __init__(self, in_channel, out_channel, kernel_conv=3, scale_up=2):
        super(UpSample, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=scale_up, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channel, out_channel, kernel_conv)

    @staticmethod
    def cropping(x_small, x_large):
        length_large_x = x_large.size()[-2]
        length_small_x = x_small.size()[-2]
        start_x = int((length_large_x - length_small_x) / 2)

        length_large_y = x_large.size()[-1]
        length_small_y = x_small.size()[-1]
        start_y = int((length_large_y - length_small_y) / 2)

        x_cropped = x_large.narrow(-2, start_x, length_small_x).narrow(-1, start_y, length_small_y)
        return x_cropped

    def forward(self, x_in, x_encoder):
        """
        1. upsample 2. catenate 3. conv
        :param x: output from the last layer
        :param x_encoder: output from the corresponding layer in encoder
        :return:
        """
        x = self.up_sample(x_in)

        # the size of output from corresponding layer in encoder is not equal to the size of upsampled one
        # need cropping
        x_cropped = self.cropping(x, x_encoder)
        x_cat = torch.cat((x_cropped, x), dim=1)
        x_conv = self.conv(x_cat)
        return x_conv


class OutConv(nn.Module):
    def __init__(self, in_channel, final_out_channel, out_channel=None):
        super(OutConv, self).__init__()
        if out_channel is None:
            self.out_channel = [32, 16, 8, final_out_channel]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, self.out_channel[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel[0], self.out_channel[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel[1], self.out_channel[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel[2], self.out_channel[3], kernel_size=3, padding=1),
        )

    def forward(self, x):
        """

        :param x: [batch, 64, H, W]
        :return: [batch, 5, H, W]
        """
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=8, final_out_channel=5, conv_kernel=3, hidden_channel=None):
        super(UNet, self).__init__()
        if hidden_channel is None:
            hidden_channel = [64, 128, 256, 512, 1024]
        """
        input: in_channel: set number, default: 8
        in-down1-down2-down3-down4-up1-up2-up3-up4-out
        """
        self.in_conv = InConv(in_channel, hidden_channel[0], conv_kernel)
        self.down_1 = DownSample(hidden_channel[0], hidden_channel[1], conv_kernel)
        self.down_2 = DownSample(hidden_channel[1], hidden_channel[2], conv_kernel)
        self.down_3 = DownSample(hidden_channel[2], hidden_channel[3], conv_kernel)
        self.down_4 = DownSample(hidden_channel[3], hidden_channel[4], conv_kernel)
        self.up_1 = UpSample(hidden_channel[4] + hidden_channel[3], hidden_channel[3], conv_kernel)
        self.up_2 = UpSample(hidden_channel[3] + hidden_channel[2], hidden_channel[2], conv_kernel)
        self.up_3 = UpSample(hidden_channel[2] + hidden_channel[1], hidden_channel[1], conv_kernel)
        self.up_4 = UpSample(hidden_channel[1] + hidden_channel[0], hidden_channel[0], conv_kernel)
        self.out_conv = OutConv(hidden_channel[0], final_out_channel=final_out_channel)
        self.tgc_conv = TGCNet(final_out_channel, final_out_channel, 1)

    def forward(self, x):
        """

        :param x: x: [bs, set_num, H_in, W_in]
        :return: x_out: [bs, 5, H, W]
        suppose: H_in = W_in = 700, H = W = 516, the label is the center [516, 516] of [700, 700]
        """
        x_down_1 = self.in_conv(x)
        x_down_2 = self.down_1(x_down_1)
        x_down_3 = self.down_2(x_down_2)
        x_down_4 = self.down_3(x_down_3)
        x_down_5 = self.down_4(x_down_4)
        x_up_1 = self.up_1(x_down_5, x_down_4)
        x_up_2 = self.up_2(x_up_1, x_down_3)
        x_up_3 = self.up_3(x_up_2, x_down_2)
        x_up_4 = self.up_4(x_up_3, x_down_1)
        x_out = self.out_conv(x_up_4)
        return x_out


class TGCNet(nn.Module):
    """This code is to add a time gain compensation function on the raw parameters"""

    def __init__(self, in_channel, out_channel, kernel=1):
        super(TGCNet, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel, kernel)

    def forward(self, raw_x):
        out = self.conv(raw_x)
        return out


class ImgReconstructor(nn.Module):
    """
    This net is to reconstruct the image itself using predicted constant and inconstant parameters
    """

    def __init__(self, scale_factor=1.4, set_size=8, in_channels=6, kernel=None):
        super(ImgReconstructor, self).__init__()
        if kernel is None:
            kernel = [7, 5, 3]

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernel[0], stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel[0], stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel[0], stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel[0], stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=kernel[2], stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, constant_map, inconstant_map):
        all_maps = torch.cat((constant_map, inconstant_map), dim=1)
        x_conv_1 = self.upsample(all_maps)
        x_conv_2 = self.conv_1(x_conv_1)
        x_conv_3 = self.conv_2(x_conv_2)
        return x_conv_3


