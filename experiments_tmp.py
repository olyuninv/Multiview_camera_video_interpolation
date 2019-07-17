#
# KTH Royal Institute of Technology
#

import torch
import numpy as np
from docutils.nodes import line
from torchvision.transforms import CenterCrop, RandomCrop
from os.path import join

import sys
sys.path.append("../")
from src.model import Net
from src.interpolate import interpolate
from src.extract_frames import extract_frames
from src.data_manager import load_img
from src.dataset import pil_to_tensor, tensor_to_pil, get_validation_set, get_test_set, get_test_set_offset, get_test_set_distance
from src.utilities import psnr, psnr_significant
from src.loss import ssim
import src.config as config
from random import choice, randint, uniform, normalvariate
import matplotlib.pyplot as plt
import csv

minRGB = [65, 154, 74]
maxRGB = [90, 190, 100]

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


def test_wiz(model, output_folder=None):
    video_path = '/project/videos/see_you_again_540.mp4'
    test_metrics(model, video_path=video_path, output_folder=output_folder)

def same_image(frame1, frame2, threshold=np.inf):
    pixels_per_channel = frame1.size / 3
    hist = lambda x: np.histogram(x.reshape(-1), 8, (0, 255))[0] / pixels_per_channel
    err = lambda a, b: ((hist(a) - hist(b)) ** 2).mean()

    return err(frame1[:, :, 0], frame2[:, :, 0]) < threshold and \
           err(frame1[:, :, 1], frame2[:, :, 1]) < threshold and \
           err(frame1[:, :, 2], frame2[:, :, 2]) < threshold

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

def visual_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) + 0.5

def test_random_sample(model, validation_set=None, save=False, linearInterpolation=False, vis_difference=False):

    if validation_set is None:
        validation_set = get_test_set()

    total_ssim = 0
    total_significant_ssim = 0

    total_psnr = 0
    total_significant_psnr = 0

    total_ssim_mean=0
    total_psnr_mean=0
    iters=len(validation_set.tuples)

    for i, tup in enumerate(validation_set.tuples):

        x1, gt, x2 = [load_img(p) for p in tup]

        pred = interpolate(model, x1, x2)

        test_patches_dir = '/home/lera/Documents/Mart_Kartasev_sepconv/test_output'
        leftFrame = tup[0].split("/")
        leftFrame_name = leftFrame[len(leftFrame) - 1]
        leftFrame_name = leftFrame_name[8: len(leftFrame_name) - 4]

        # SAVE prediction & patches
        if (save):
            x1.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, 1)))
            x2.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, 3)))
            gt.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, 2)))
            pred.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, '2_pred')))

        x1 = pil_to_tensor(x1)
        x2 = pil_to_tensor(x2)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()
        total_significant_psnr += psnr_significant(pred, gt, minBGR, maxBGR).item()

        if vis_difference:
            vis_diff = visual_difference(gt, pred)
            vis_diff = tensor_to_pil(vis_diff)
            vis_diff.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, 'vd')))

        if linearInterpolation:
            pred_mean= torch.mean(torch.stack((x1, x2), dim=0), dim=0)
            total_ssim_mean += ssim(pred_mean, gt).item()
            total_psnr_mean += psnr(pred_mean, gt).item()
            if save:
                pred_mean=tensor_to_pil(pred_mean)
                pred_mean.save(join(test_patches_dir, '{}_{}.jpg'.format(leftFrame_name, 'mean')))

        print(f'#{i + 1} done')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')

    if linearInterpolation:
        avg_ssim_mean = total_ssim_mean / iters
        avg_psnr_mean = total_psnr_mean / iters
        print(f'avg_ssim for linear interpolation: {avg_ssim_mean}, avg_psnr for linear interpolation: {avg_psnr_mean}')

def test_linear_interp(validation_set=None):

    if validation_set is None:
        validation_set = get_test_set()

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
    (steps_dict, mapSteps) = get_test_set_distance(0, 7, 0.5)

    if results is None:
        print('This code is unfinished. please provide results file containing ssims and psnrs')
    else:
        list_ssim = []
        list_psnr = []

        #for step in range(len(mapSteps)):
        for step in steps_dict:

            total_ssim = 0
            total_significant_ssim = 0

            total_psnr = 0
            total_significant_psnr = 0

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
                total_significant_psnr += psnr_significant(pred, gt, minBGR, maxBGR).item()

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
            total_significant_ssim = 0

            total_psnr = 0
            total_significant_psnr = 0

            count = 0

            for runNumber in run_to_step:
                if run_to_step.get(runNumber) == step:
                    selected_rows = results[results[:,0] == runNumber ][:,np.array([False, False, True, True])]
                    total_ssim += sum (selected_rows[:, 0])
                    total_psnr += sum (selected_rows[:, 1])
                    total_significant_psnr += psnr_significant(pred, gt, minBGR, maxBGR).item()

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

def record_all(model, results_folder, exclude_green_pixels = False):

    validation_set = get_test_set(False)    # GET ALL TUPLES

    results = np.empty((len(validation_set.tuples), 4))

    #j=0

    for i, tup in enumerate(validation_set.tuples):
        tuple_name = tup[0].split("/")
        run_number = int(tuple_name[len(tuple_name) - 1][8:12])
        frame_number = int(tuple_name[len(tuple_name) - 1][13:17])

        x1, gt, x2 = [load_img(p) for p in tup]
        pred = interpolate(model, x1, x2)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)

        this_ssim = ssim(pred, gt).item()
        this_psnr = psnr(pred, gt).item()

        results[i][0] = run_number
        results[i][1] = frame_number
        results[i][2] = this_ssim
        results[i][3] = this_psnr
        print(f'#{i + 1} done')

        # j += 1
        # if j > 2:
        #     break

    results_file = join(results_folder, "ssim_psnr_all_7.npy")
    np.save(results_file, results)

    return results

def test_all():

    print('===> Loading pure L1...')
    pure_l1 = Net.from_file('../trained_models_71/model_epoch_14.pth')

    print('===> Testing patches...')
    test_random_sample(pure_l1, None, True, True, True)

    # Record all ssim/ psnr
    results_folder = '/home/lera/Documents/Mart_Kartasev_sepconv/results_71'

    #record all ssim_psnr
    record_all(pure_l1, results_folder)

    results_file = join(results_folder, "ssim_psnr_all_7.npy")
    np_array = np.load(results_file)

    print('===> Testing camera offset...')
    plot_offset(pure_l1, np_array)

    print('===> Testing distance to subject...')
    plot_distance_to_subject(pure_l1, np_array)

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

if __name__ == '__main__':
    test_all()
