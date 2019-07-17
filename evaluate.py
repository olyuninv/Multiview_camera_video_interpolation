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
from docutils.nodes import line
from statsmodels.tsa.statespace._simulation_smoother import sSimulationSmoother
from torchvision.transforms import CenterCrop, RandomCrop
import os
from os.path import join

import argparse
import sys
sys.path.append("../")
from src.model import Net
from src.interpolate import interpolate
from src.extract_frames import extract_frames
from src.data_manager import load_img
from src.dataset import pil_to_tensor, tensor_to_pil, get_validation_set, get_test_set, get_test_set_offset, get_test_set_distance
from src.utilities import psnr, psnr_significant
from src.loss import ssim, ssim_significant
import src.config as config
from random import choice, randint, uniform, normalvariate
import matplotlib.pyplot as plt
#plt.rcParams['backend'] = 'Qt4Agg'
#plt.rcParams['backend.qt5'] = 'PyQt4'
import cv2 as cv

minRGB = [65.0/255, 154.0/255, 74.0/255]
maxRGB = [91.0/255, 190.0/255, 101.0/255]

## Unused
def test_metrics(model, video_path=None, frames=None, output_folder=None):

    if video_path is not None and frames is None:
        frames, _ = extract_frames(video_path)

    total_ssim = 0
    total_psnr = 0
    stride = 30
    iters = 1 + (len(frames) - 3) // stride

    triplets = []
    for i in range(iters):
        tup = (frames[i*stride], frames[i*stride + 1], frames[i*stride + 2])
        triplets.append(tup)

    iters = len(triplets)

    for i in range(iters):
        x1, gt, x2 = triplets[i]
        pred = interpolate(model, x1, x2)
        if output_folder is not None:
            frame_path = join(output_folder, f'wiz_{i}.jpg')
            pred.save(frame_path)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()
        print(f'#{i+1}/{iters} done')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')

## Unused
def test_wiz(model, output_folder=None):
    video_path = '/project/videos/see_you_again_540.mp4'
    test_metrics(model, video_path=video_path, output_folder=output_folder)

## Unused
def test_patches(model, validation_set=None):
    if validation_set is None:
        validation_set = get_test_set()

    # crop = RandomCrop(config.CROP_SIZE) #CenterCrop(config.CROP_SIZE)

    # load green image for testing jumpcuts
    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]
    greenImage = load_img('/media/lera/ADATA HV320/mv_output/entireGreen_0.jpg')
    greenImage = pil_to_numpy(greenImage)
    num_trials = 50
    patch_h, patch_w = config.PATCH_SIZE

    jumpcut_threshold = 15e-3

    for i, tup in enumerate(validation_set.tuples):
        x1, gt, x2 = [load_img(p) for p in tup]  # [crop(load_img(p)) for p in tup]

        for count in range(num_trials):

            img_w, img_h = x1.size

            i = randint(0, img_h - patch_h)
            j = randint(0, img_w - patch_w)

            left_patch = x1[i:i + patch_h, j:j + patch_w, :]
            right_patch = x2[i:i + patch_h, j:j + patch_w, :]
            middle_patch = gt[i:i + patch_h, j:j + patch_w, :]

            left_patch = pil_to_numpy(left_patch)
            middle_patch = pil_to_numpy(middle_patch)
            right_patch = pil_to_numpy(right_patch)

            if (same_image(left_patch, greenImage, jumpcut_threshold) and \
                same_image(middle_patch, greenImage, jumpcut_threshold)) or \
                    (same_image(middle_patch, greenImage, jumpcut_threshold) and \
                     same_image(right_patch, greenImage, jumpcut_threshold)):
                continue
            else:
                break

def simple_flow(frame1, frame2):
    """
    Runs SimpleFlow given two consecutive frames.
    :param frame1: Numpy array of the frame at time t
    :param frame2: Numpy array of the frame at time t+1
    :return: Numpy array with the flow for each pixel. Shape is same as input
    """
    flow = cv.optflow.calcOpticalFlowSF(frame1, frame2, layers=3, averaging_block_size=2, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=(0, 1))
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n)

def same_image(frame1, frame2, threshold=np.inf):
    pixels_per_channel = frame1.size / 3
    hist = lambda x: np.histogram(x.reshape(-1), 8, (0, 255))[0] / pixels_per_channel
    err = lambda a, b: ((hist(a) - hist(b)) ** 2).mean()

    return err(frame1[:, :, 0], frame2[:, :, 0]) < threshold and \
           err(frame1[:, :, 1], frame2[:, :, 1]) < threshold and \
           err(frame1[:, :, 2], frame2[:, :, 2]) < threshold

def visual_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) + 0.5

def evaluate_random_sample(model, results_folder, save=False, linearInterpolation=False, vis_difference=False, significantOnly = False):

    if significantOnly:
        print ('Evaluating non-green pixels only.')

    validation_set = get_test_set(100)  # 100 random tuples

    total_ssim = 0
    total_psnr = 0

    total_ssim_mean=0
    total_psnr_mean=0
    iters=len(validation_set.tuples)

    for i, tup in enumerate(validation_set.tuples):

        x1, gt, x2 = [load_img(p) for p in tup]

        pred = interpolate(model, x1, x2)

        leftFrame = tup[0].split("/")
        leftFrame_name = leftFrame[len(leftFrame) - 1]
        leftFrame_name = leftFrame_name[8: len(leftFrame_name) - 4]

        # SAVE prediction & patches
        if (save):
            x1.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, 1)))
            x2.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, 3)))
            gt.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, 2)))
            pred.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, '2_pred')))

        x1 = pil_to_tensor(x1)
        x2 = pil_to_tensor(x2)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)

        psnr_item = 0.0
        ssim_item = 0.0

        if significantOnly:
            psnr_item = psnr_significant(pred, gt).item()
            ssim_item = ssim_significant(pred, gt).item()
        else:
            psnr_item = psnr(pred, gt).item()
            ssim_item = ssim(pred, gt).item()

        total_psnr += psnr_item
        total_ssim += ssim_item

        if vis_difference:
            vis_diff = visual_difference(gt, pred)
            vis_diff = tensor_to_pil(vis_diff)
            vis_diff.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, 'vd')))

        if linearInterpolation:
            pred_mean= torch.mean(torch.stack((x1, x2), dim=0), dim=0)
            total_ssim_mean += ssim(pred_mean, gt).item()
            total_psnr_mean += psnr(pred_mean, gt).item()
            if save:
                pred_mean=tensor_to_pil(pred_mean)
                pred_mean.save(join(results_folder, '{}_{}.jpg'.format(leftFrame_name, 'mean')))

        print(f'#{i + 1} done, name: {leftFrame_name}, ssim: {ssim_item}, psnr: {psnr_item}')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    if significantOnly:
        print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for significant only')
    else:
        print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')

    if linearInterpolation:
        avg_ssim_mean = total_ssim_mean / iters
        avg_psnr_mean = total_psnr_mean / iters
        print(f'avg_ssim for linear interpolation: {avg_ssim_mean}, avg_psnr for linear interpolation: {avg_psnr_mean}')

def test_linear_interp(validation_set=None):

    if validation_set is None:
        validation_set = get_test_set(100)

    total_ssim = 0
    total_psnr = 0
    iters = len(validation_set.tuples)

    #crop = CenterCrop(config.CROP_SIZE)

    for tup in validation_set.tuples:
        x1, gt, x2, = [pil_to_tensor(load_img(p)) for p in tup] #[pil_to_tensor(crop(load_img(p))) for p in tup]
        pred = torch.mean(torch.stack((x1, x2), dim=0), dim=0)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')

def plot_distance_to_subject(model, results = None):
    ## This just plots results based on provided npy file - if this is significant or not - results correspond

    (steps_dict, mapSteps) = get_test_set_distance(0, 7, 0.5)

    if results is None:
        print('This code is unfinished. please provide results file containing ssims and psnrs')
    else:
        list_ssim = []
        list_psnr = []

        #for step in range(len(mapSteps)):
        for step in steps_dict:

            total_ssim = 0
            total_psnr = 0

            run_frames = steps_dict.get(step)
            count = len(run_frames)

            for run_frame in run_frames:
                run = int(run_frame[0])
                frame = run_frame[1]
                selected_rows = results[results[:, 0] == run]
                selected_rows = selected_rows[selected_rows[:, 1] == frame]
                selected_rows = selected_rows[:, np.array([False, False, True, True])]
                total_ssim += sum(selected_rows[:, 0])
                total_psnr += sum(selected_rows[:, 1])
                #count += len(selected_rows)

            avg_ssim = 0
            avg_psnr = 0

            if count != 0:
                avg_ssim = total_ssim / count
                avg_psnr = total_psnr / count
            print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for step {step}')

            list_ssim.append(avg_ssim)
            list_psnr.append(avg_psnr)

    # Create labels
    index = np.arange(len(mapSteps))
    labels = []
    min_column_index = 1
    max_column_index = 2

    for j, step in enumerate(mapSteps):
        labels.append(str(step[min_column_index]) + ' - ' + str(step[max_column_index]))

    # plot ssim
    plt.figure(1)
    plt.bar(index, list_ssim)
    plt.xlabel('Distance to subject', fontsize=7)
    plt.ylabel('SSIM', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)

    for i, v in zip(index, list_ssim):
        plt.text(i, v, "{:0.4f}".format(v), fontsize=7)

    plt.title('SSIM depending on distance')
    plt.show()

    # plot psnr
    plt.figure(2)
    plt.bar(index, list_psnr)
    plt.xlabel('Distance to subject', fontsize=7)
    plt.ylabel('PSNR', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)
    for i, v in zip(index, list_psnr):
        plt.text(i, v, "{:0.2f}".format(v), fontsize=7)
    plt.title('PSNR depending on distance')
    plt.show()

def plot_offset(model, results = None):
    ## This just plots results based on provided npy file - if this is significant or not - results correspond

    (run_to_step, mapSteps) = get_test_set_offset(0.05, 0.50, 0.05)

    if results is None:
        print ('Need to get tuples and run the test')
        print('This code is unfinished. Please provide results containing ssims and psnrs')

        #validation_set = get_test_set(False)

        ###
        #
        # for step in range(len(tuples_dict)):
        #     tuples_list = tuples_dict.get(step)
        #
        #     total_ssim = 0
        #     total_psnr = 0
        #     iters = len(tuples_list)
        #
        #     for i, tup in enumerate(tuples_list):
        #         x1, gt, x2 = [load_img(p) for p in tup]
        #         pred = interpolate(model, x1, x2)
        #         gt = pil_to_tensor(gt)
        #         pred = pil_to_tensor(pred)
        #         total_ssim += ssim(pred, gt).item()
        #         total_psnr += psnr(pred, gt).item()
        #         print(f'#{i + 1} in step {step} done')
        #
        #     avg_ssim = total_ssim / iters
        #     avg_psnr = total_psnr / iters
        #
        #     print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for step {step}')
        #
        #     list_ssim.append(avg_ssim)
        #     list_psnr.append(avg_psnr)
    else:
        (run_to_step, mapSteps) = get_test_set_offset(0.05, 0.50, 0.05)

        list_ssim = []
        list_psnr = []

        for step in range(len(mapSteps)):

            total_ssim = 0
            total_psnr = 0
            count = 0

            for runNumber in run_to_step:
                if run_to_step.get(runNumber) == step:
                    selected_rows = results[results[:,0] == runNumber ][:,np.array([False, False, True, True])]
                    total_ssim += sum (selected_rows[:, 0])
                    total_psnr += sum (selected_rows[:, 1])
                    count += len(selected_rows)

            avg_ssim = total_ssim / count
            avg_psnr = total_psnr / count
            print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for step {step}')

            list_ssim.append(avg_ssim)
            list_psnr.append(avg_psnr)

    #Create labels
    index = np.arange(len(mapSteps))
    labels = []
    min_column_index = 1
    max_column_index = 2

    for j, step in enumerate(mapSteps):
        labels.append("{:0.2f} - {:0.2f}".format(step[min_column_index],step[max_column_index]))

    # plot ssim
    plt.figure(1)
    plt.bar(index, list_ssim)
    plt.xlabel('Offset', fontsize=7)
    plt.ylabel('SSIM', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)
    for i, v in zip(index, list_ssim):
        plt.text(i, v, "{:0.4f}".format(v), fontsize=7)
    plt.title('SSIM depending on offset')
    plt.show()

    # plot psnr
    plt.figure(2)
    plt.bar(index, list_psnr)
    plt.xlabel('Offset', fontsize=7)
    plt.ylabel('PSNR', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)
    for i, v in zip(index,list_psnr):
        plt.text(i, v, "{:0.2f}".format(v), fontsize=7)
    plt.title('PSNR depending on offset')
    plt.show()

def record_all(model, results_folder, significant_only = False):

    validation_set = get_test_set(0, False)    # GET ALL TUPLES

    results = np.empty((len(validation_set.tuples), 4))

    #j = 0
    for i, tup in enumerate(validation_set.tuples):
        tuple_name = tup[0].split("/")
        run_number = int(tuple_name[len(tuple_name) - 1][8:12])
        frame_number = int(tuple_name[len(tuple_name) - 1][13:17])

        x1, gt, x2 = [load_img(p) for p in tup]
        pred = interpolate(model, x1, x2)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)

        this_psnr = 0.0
        this_ssim = 0.0

        if significant_only:
            this_psnr = psnr_significant(pred, gt).item()
            this_ssim = ssim_significant(pred, gt).item()
        else:
            this_psnr = psnr(pred, gt).item()
            this_ssim = ssim(pred, gt).item()

        results[i][0] = run_number
        results[i][1] = frame_number
        results[i][2] = this_ssim
        results[i][3] = this_psnr
        print(f'#{i + 1} done, runnumber: {run_number}, framenumber: {frame_number}, ssim: {this_ssim}, psnr: {this_psnr}')

        #j += 1
        #if j > 2:
        #    break

    results_file = ""

    if significant_only:
        results_file = join(results_folder, "ssim_psnr_all_8_significant.npy")
    else:
        results_file = join(results_folder, "ssim_psnr_all_8.npy")

    np.save(results_file, results)

    return results

def map_steps(min=5, max=50, stepSize=5):
    nSteps = int((max - min) / stepSize )

    mapSteps = None

    for step in range(0, nSteps):
        step_min = min + stepSize * step
        step_max = step_min + stepSize
        newStep = np.array((step, step_min, step_max))

        if mapSteps is None:
            mapSteps = newStep
        else:
            mapSteps = np.vstack((mapSteps, newStep))

    return nSteps, mapSteps

def plot_optic_flow_category(results_folder, results_file_name):
    ## This just plots results based on provided npy file - if this is significant or not - results correspond

    results_file = join(results_folder, results_file_name)
    np_array = np.load(results_file)

    dump_file = False
    if dump_file:
        np.savetxt(join(results_folder, "results.csv"), np_array, delimiter=",")

    min_opt_flow = np.amin(np_array[:,4], axis=0)
    max_opt_flow = np.amax(np_array[:,4], axis=0)

    (nSteps, mapSteps) = map_steps(0.0, 5.0, 0.2)

    # find rows in npy_file that correspond to the step
    step_column_index = 0
    min_column_index = 1
    max_column_index = 2
    run_column_index = 0
    frame_column_index = 1
    opt_flow_column_index = 4

    steps_dict = dict()

    # create dictionary for runs and frames in each step
    for step in range(nSteps):
        steps_dict[step] = []

    for step in mapSteps:
        for i in range(len(np_array)):
            if np_array[i][opt_flow_column_index] >= step[min_column_index] and np_array[i][opt_flow_column_index] <= \
                    step[max_column_index]:
                steps_dict.get(step[step_column_index]).append(
                    (np_array[i][run_column_index], np_array[i][frame_column_index]))

    list_ssim = []
    list_psnr = []

    # for step in range(len(mapSteps)):
    for step in steps_dict:

        total_ssim = 0
        total_psnr = 0

        run_frames = steps_dict.get(step)
        count = len(run_frames)

        for run_frame in run_frames:
            run = int(run_frame[0])
            frame = run_frame[1]
            selected_rows = np_array[np_array[:, 0] == run]
            selected_rows = selected_rows[selected_rows[:, 1] == frame]
            selected_rows = selected_rows[:, np.array([False, False, True, True, False])]
            total_ssim += sum(selected_rows[:, 0])
            total_psnr += sum(selected_rows[:, 1])
            # count += len(selected_rows)

        avg_ssim = 0
        avg_psnr = 0

        if count != 0:
            avg_ssim = total_ssim / count
            avg_psnr = total_psnr / count
        print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr} for step {step}')

        list_ssim.append(avg_ssim)
        list_psnr.append(avg_psnr)

    # Create labels
    index = np.arange(len(mapSteps))
    labels = []
    min_column_index = 1
    max_column_index = 2

    for j, step in enumerate(mapSteps):
        labels.append("{:0.2f} - {:0.2f}".format(step[min_column_index],step[max_column_index]))

    # plot ssim
    plt.figure(1)
    plt.bar(index, list_ssim)
    plt.xlabel('Optical Flow', fontsize=7)
    plt.ylabel('SSIM', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)

    for i, v in zip(index, list_ssim):
        plt.text(i, v, "{:0.4f}".format(v), fontsize=7)

    plt.title('SSIM depending on optical flow')
    plt.show()

    # plot psnr
    plt.figure(2)
    plt.bar(index, list_psnr)
    plt.xlabel('Optical Flow', fontsize=7)
    plt.ylabel('PSNR', fontsize=7)
    plt.xticks(index, labels, fontsize=7, rotation=30)
    for i, v in zip(index, list_psnr):
        plt.text(i, v, "{:0.2f}".format(v), fontsize=7)
    plt.title('PSNR depending on optical flow')
    plt.show()

    j = 0

def calculate_pixel_distance(x1: torch.Tensor, x2: torch.Tensor):
    # find the bounding boxes
    tensor_size = x1.size()  # assime b is the same size
    zero_array = torch.zeros(tensor_size)
    zero_array_RGB = torch.zeros((tensor_size[1], tensor_size[2]))
    ones_array_RGB = torch.ones((tensor_size[1], tensor_size[2]))

    # ones_array = torch.ones(tensor_size, dtype=torch.float32)

    # x1 bounding box
    x1_green = torch.where((minRGB[0] <= x1[0, :, :]), x1, zero_array)
    x1_green = torch.where((x1[0, :, :] <= maxRGB[0]), x1_green, zero_array)
    x1_green = torch.where((minRGB[1] <=x1[1, :, :]), x1_green, zero_array)
    x1_green = torch.where((x1[1, :, :] <= maxRGB[1]), x1_green, zero_array)
    x1_green = torch.where((minRGB[2] <= x1[2, :, :]), x1_green, zero_array)
    x1_green = torch.where((x1[2, :, :] <= maxRGB[2]), x1_green, zero_array)

    # save the image
    #x1_green_pil = tensor_to_pil(x1_green)
    #x1_green_pil.save(join('/home/lera/Documents/Mart_Kartasev_sepconv/', 'test_image.png'))

    x1_green_sum_RGB = torch.sum(x1_green, 0)

    x1_opposite = torch.where(x1_green_sum_RGB > 0, zero_array_RGB, ones_array_RGB)

    index_non_zero_x1 = torch.nonzero(x1_opposite)
    min_index_rows_x1 = torch.min(index_non_zero_x1, 0)[0][1]
    max_index_rows_x1 = torch.max(index_non_zero_x1, 0)[0][1]
    median_x1 = (max_index_rows_x1 - min_index_rows_x1) / 2
    
    # x2 bounding box 
    x2_green = torch.where((minRGB[0] <= x2[0, :, :]), x2, zero_array)
    x2_green = torch.where((x2[0, :, :] <= maxRGB[0]), x2_green, zero_array)
    x2_green = torch.where((minRGB[1] <= x2[1, :, :]), x2_green, zero_array)
    x2_green = torch.where((x2[1, :, :] <= maxRGB[1]), x2_green, zero_array)
    x2_green = torch.where((minRGB[2] <= x2[2, :, :]), x2_green, zero_array)
    x2_green = torch.where((x2[2, :, :] <= maxRGB[2]), x2_green, zero_array)

    x2_green_sum_RGB = torch.sum(x2_green, 0)

    x2_opposite = torch.where(x2_green_sum_RGB > 0, zero_array_RGB, ones_array_RGB)

    index_non_zero_x2 = torch.nonzero(x2_opposite)
    min_index_rows_x2 = torch.min(index_non_zero_x2, 0)[0][0]
    max_index_rows_x2 = torch.max(index_non_zero_x2, 0)[0][0]
    median_x2 = (max_index_rows_x2 - min_index_rows_x2) / 2
    
    return torch.abs(median_x2 - median_x1)

def add_optic_flow_column(model, results_folder, results_file_name):
    #print('===> Loading pure L1...')  ##model is not used
    #pure_l1 = Net.from_file(model)

    results_file = join(results_folder, results_file_name)
    np_array = np.load(results_file)

    n_rows = np_array.shape[0]
    zero_column = np.zeros(n_rows)

    #Add column of zeros to numpy array
    update_array = np.c_[np_array, zero_column]

    validation_set = get_test_set(0, False)  # GET ALL TUPLES

    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

    for i, tup in enumerate(validation_set.tuples):
        tuple_name = tup[0].split("/")
        run_number = int(tuple_name[len(tuple_name) - 1][8:12])
        frame_number = int(tuple_name[len(tuple_name) - 1][13:17])

        #x1, gt, x2 = [pil_to_numpy(load_img(p)) for p in tup]
        x1, gt, x2 = [pil_to_tensor(load_img(p)) for p in tup]

        #need pil to numpy
        #avg_flow = simple_flow(x1, x2)
        avg_flow = calculate_pixel_distance(x1, x2)

        update_row_indexs = np.where(np.all([update_array[:,0] == run_number, update_array[:, 1] == frame_number], axis=0))
        if len(update_row_indexs[0]) == 1:
            update_row_index = update_row_indexs[0][0]
            update_array[update_row_index, 4] = avg_flow

    results_file_flow = join(results_folder,  "ssim_psnr_all_8_significant_with_pixel_distance.npy") # "ssim_psnr_all_8_significant_with_flow.npy")
    np.save(results_file_flow, update_array)

    return update_array

def test_distance_category(model, results_folder, results_file_name):
    print('===> Loading pure L1...')
    pure_l1 = Net.from_file(model)

    results_file = join(results_folder, results_file_name)
    np_array = np.load(results_file)

    print('===> Testing distance to subject...')
    plot_distance_to_subject(pure_l1, np_array)

def test_offset_category(model, results_folder, results_file_name):
    print('===> Loading pure L1...')
    pure_l1 = Net.from_file(model)

    results_file = join(results_folder, results_file_name)
    np_array = np.load(results_file)

    print('===> Testing camera offset...')
    plot_offset(pure_l1, np_array)

def interpolate_all(model, results_folder, significant_only = False):
    print('===> Loading pure L1...')
    pure_l1 = Net.from_file(model)

    # record all ssim_psnr
    record_all(pure_l1, results_folder, significant_only)

def evaluate_all(model, results_folder, save=False, linearInterpolation=False, vis_difference=False, significantOnly = False):

    print('===> Loading pure L1...')
    pure_l1 = Net.from_file(model)

    print('===> Testing patches...')
    evaluate_random_sample(pure_l1, results_folder, save, linearInterpolation, vis_difference, significantOnly)

    #print('===> Testing linear interp...')
    #test_linear_interp()
    #print('avg_ssim: 0.6868560968339443, avg_psnr: 26.697076902389526')

    #print('===> Loading best models...')
    # best_model_qualitative = Net.from_file('./trained_models/best_model_qualitative.pth')
    # best_model_quantitative = Net.from_file('./trained_models/best_model_quantitative.pth')

    #print('===> Testing Wiz (qualitative)...')
    # test_wiz(best_model_qualitative, output_folder='/project/exp/wiz_qual/')
    #print('avg_ssim: 0.9658980375842044, avg_psnr: 37.27564642554835')

    #print('===> Testing Wiz (quantitative)...')
    # test_wiz(best_model_quantitative, output_folder='/project/exp/wiz_quant/')
    #print('avg_ssim: 0.9638479389642415, avg_psnr: 36.52394056822124')

def clean_data(test_folder):
    validation_set = get_test_set(0, False)  # GET ALL TUPLES

    jumpcut_threshold = 60e-6 #Good - 15e-6
    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

    bk_image = '/home/lera/Documents/surreal-master/datageneration/misc/background/just_green_test.png'
    greenImage = load_img(bk_image)
    greenImage = pil_to_numpy(greenImage)

    for i, tup in enumerate(validation_set.tuples):
        x1, gt, x2 = [pil_to_numpy(load_img(p)) for p in tup]

        # if 2 entirely green images - delete
        if (same_image(x1, greenImage, jumpcut_threshold) and \
            same_image(gt, greenImage, jumpcut_threshold)) or \
                (same_image(gt, greenImage, jumpcut_threshold) and \
                 same_image(x2, greenImage, jumpcut_threshold)):

            print('Deleting tuple %s' % tup[0])
            os.remove(tup[0])
            os.remove(tup[1])
            os.remove(tup[2])
            print('Tuple removed')


import time

if __name__ == '__main__':
    global start_time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Frame Interpolation')

    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    parser.add_argument('--test', type=str, required=True, help='evaluate_all,interpolate_all,offset_cat,distance_cat,add_oflow,oflow_cat')
    parser.add_argument('--results_folder', type=str, required=True, help='folder to output results if applicable')
    params = parser.parse_args()

    # Record all ssim/ psnr
    results_folder = params.results_folder  # '/home/lera/Documents/Mart_Kartasev_sepconv/results_71'

    if params.test == "clean_data":
        clean_data(results_folder)   #'/media/lera/ADATA HV320/mv_output/TEST'

    # test random sample of 100
    if params.test == "evaluate_all":
        evaluate_all(params.model, results_folder, False, False, False, True)

    if params.test == "interpolate_all":
        interpolate_all(params.model, results_folder, True)

    results_file_name = "ssim_psnr_all_8_significant.npy" #"ssim_psnr_all_7.npy"

    if params.test == "offset_cat":
        test_offset_category(params.model, results_folder, results_file_name)

    if params.test == "distance_cat":
        test_distance_category(params.model, results_folder, results_file_name)

    if params.test == "add_oflow":
        add_optic_flow_column(params.model, results_folder, results_file_name)

    if params.test == "oflow_cat":
        plot_optic_flow_category(results_folder, "ssim_psnr_all_8_significant_with_flow.npy")

    elapsed_time = time.time() - start_time
    print("Elapsed time: [%.2f s]" % elapsed_time)