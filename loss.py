# IMPLEMENTATION BASED ON CODE:
# https://github.com/martkartasev/sepconv
#
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# The VGG loss in this file is copied from
# https://github.com/ekgibbons/pytorch-sepconv/blob/master/python/_support/VggLoss.py
# The SsimLoss loss in this file is copied (with minor modifications) from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

from math import exp

import torch
import torch.nn.functional as F
import torchvision
import src.config as config
from torch import nn
from torch.autograd import Variable

minRGB = [65.0/255, 154.0/255, 74.0/255]
maxRGB = [91.0/255, 190.0/255, 101.0/255]

class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)

        loss = torch.norm(outputFeatures - targetFeatures, 2)

        return config.VGG_FACTOR * loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.l1(output, target)


class SsimLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SsimLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return -_ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim_significant(img1, img2, window_size=11, size_average=True):
    return ssim(img1, img2, window_size, size_average, True)

def ssim(img1, img2, window_size=11, size_average=True, significant_only=False):

    if len(img1.size()) == 3:
        img1 = torch.stack([img1], dim=0)
        img2 = torch.stack([img2], dim=0)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, significant_only)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, significant_only = False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # exclude totally green pixels from mean
    if significant_only:

        tensor_size = ssim_map.size()  # assime b is the same size
        zero_array = torch.zeros(tensor_size)

        ssim_map_green = torch.where((minRGB[0] <= img1[:, 0, :, :]), ssim_map, zero_array)
        ssim_map_green = torch.where((img1[:, 0, :, :] <= maxRGB[0]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((minRGB[1] <= img1[:, 1, :, :]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((img1[:, 1, :, :] <= maxRGB[1]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((minRGB[2] <= img1[:, 2, :, :]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((img1[:, 2, :, :] <= maxRGB[2]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((minRGB[0] <= img2[:, 0, :, :]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((img2[:, 0, :, :] <= maxRGB[0]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((minRGB[1] <= img2[:, 1, :, :]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((img2[:, 1, :, :] <= maxRGB[1]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((minRGB[2] <= img2[:, 2, :, :]), ssim_map_green, zero_array)
        ssim_map_green = torch.where((img2[:, 2, :, :] <= maxRGB[2]), ssim_map_green, zero_array)

        ssim_map_green_sum = torch.sum(ssim_map_green)  ## GREEN PIXEL SSIM
        ssim_map_sum = torch.sum(ssim_map)  ## TOTAL SSIM
        ssim_map_corrected_sum = ssim_map_sum - ssim_map_green_sum  ## NON-GREEN PIXEL SSIM

        ssim_map_green_sum_RGB = torch.sum(ssim_map_green, 1)
        sq_error_green_count = torch.nonzero(ssim_map_green_sum_RGB).size()[0]
        num_signif = tensor_size[2] * tensor_size[3] - sq_error_green_count

        # ssim_map_sum_check = 0.0
        #
        # countGreen_check = 0;
        # for i in range(tensor_size[2]):
        #     for j in range(tensor_size[3]):
        #         # if pixel is green in both increase counter
        #         if (minRGB[0] <= img1[0, 0, i, j] <= maxRGB[0] and minRGB[1] <= img1[0, 1, i, j] <= maxRGB[1] and minRGB[2] <= img1[
        #             0, 2, i, j] <= maxRGB[2]) and \
        #                 (minRGB[0] <= img2[0, 0, i, j] <= maxRGB[0] and minRGB[1] <= img2[0, 1, i, j] <= maxRGB[1] and minRGB[2] <= img2[
        #                     0, 2, i, j] <= maxRGB[2]):  # inrange
        #             countGreen_check += 1
        #         else:  # add to error
        #             ssim_map_sum_check += torch.sum(ssim_map[:, :, i, j])
        #
        # num_signif_check = tensor_size[2] * tensor_size[3] - countGreen_check

        if (num_signif == 0):
            # image is all green - print warning
            print("Warning: image is entirely green - please delete")
            return 0
        else:
            return ssim_map_corrected_sum / (3 * num_signif)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
