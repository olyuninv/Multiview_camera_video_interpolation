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


import argparse
from timeit import default_timer as timer
from os.path import join
from os import listdir
from src.utilities import write_video
from src.data_manager import is_image, load_img


def stitch_frames(src_path, dest_path, output_fps=None, drop_frames=False):

    tick_t = timer()

    frames = [join(src_path, x) for x in listdir(src_path)]
    frames = [x for x in frames if is_image(x)]
    frames.sort()

    print('===> Loading frames...')
    if drop_frames:
        frames = [load_img(x) for i, x in enumerate(frames) if i % 2 == 0]
    else:
        frames = [load_img(x) for i, x in enumerate(frames)]

    print('===> Writing results...')
    write_video(dest_path, frames, output_fps)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Frames to video file')
    parser.add_argument('--src', dest='src_path', type=str, required=True, help='path to the directory containing the frames')
    parser.add_argument('--dest', dest='dest_path', type=str, required=True, help='path to the output file')
    parser.add_argument('--outputfps', dest='output_fps', type=int, required=False, default=None, help='frame-rate of the output')
    parser.add_argument('--dropframes', dest='drop_frames', type=bool, required=False, default=False, help='whether or not every other frame should be dropped')
    stitch_frames(**vars(parser.parse_args()))
