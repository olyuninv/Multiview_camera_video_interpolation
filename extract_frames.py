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


import sys
sys.path.append("../")

import imageio
import argparse

from os.path import join
from timeit import default_timer as timer
from PIL import Image


def extract_frames(video_path):

    def convert_frame(arg):
        return Image.fromarray(arg[:, :, :3], mode='RGB')

    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data().get('fps', None)
    frames = [convert_frame(x) for x in video_reader]

    return frames, fps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Frame Extraction')
    parser.add_argument('--src', type=str, required=True, help='path to the video')
    parser.add_argument('--dest', type=str, required=True, help='path to the output directory')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Extracting frames...')
    extracted_frames, _ = extract_frames(params.src)

    print('===> Writing results...')
    for i, frame in enumerate(extracted_frames):
        file_name = '{:05d}.jpg'.format(i)
        file_path = join(params.dest, file_name)
        frame.save(file_path)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
