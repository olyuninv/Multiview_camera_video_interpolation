#
# KTH Royal Institute of Technology
#

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


import torch
import numpy as np
import cv2 as cv


def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = frames[0].size
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_cv(frame))

    writer.release()


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()

def mse_significant(a: torch.Tensor, b: torch.Tensor, minRGB, maxRGB) -> torch.Tensor:
    sq_error = (a - b) ** 2

    tensor_size = a.size()  # assime b is the same size

    #exclude totally green pixels from mean
    countGreen = 0;
    for i in range(tensor_size[1]):
        for j in range(tensor_size[2]):
            # if pixel is green in both increase counter
            if minRGB[0] <= a[0,i,j]  <= maxRGB[0] and minRGB[1] <= a[1,i,j]  <= maxRGB[2] and minRGB[0] <= a[2,i,j]  <= maxRGB[2]:  # inrange
                countGreen+=1

    num_signif = tensor_size[1] * tensor_size[2] - countGreen

    return sum(sq_error)/ num_signif

def psnr(approx: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise ratio between two images
    :param approx: Approximated image as a tensor
    :param target: Target image as a tensor
    :return: PSNR as a tensor
    """
    _mse = mse(approx, target)
    _max = target.max()
    return 20 * _max.log10() - 10 * _mse.log10()

def psnr_significant(approx: torch.Tensor, target: torch.Tensor, minRGB, maxRGB) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise ratio between two images
    :param approx: Approximated image as a tensor
    :param target: Target image as a tensor
    :return: PSNR as a tensor
    """
    _mse = mse_significant(approx, target, minRGB, maxRGB)
    _max = target.max()
    return 20 * _max.log10() - 10 * _mse.log10()
