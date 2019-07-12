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

import numpy as np
from tqdm import tqdm

from src import data_manager, dataset
from src.dataset import get_validation_set


def main():
    validation_set = get_validation_set()
    l1_loss = 0
    for p1, p2, p3 in tqdm(validation_set.tuples):
        p1, p2, p3, = [dataset.pil_to_numpy(data_manager.load_img(p)) for p in (p1, p2, p3)]
        p2_predicted = np.mean([p1, p3], axis=0)
        l1_loss += np.mean(np.abs(p2 - p2_predicted))
    print(f"Linear interpolation: {l1_loss / len(validation_set.tuples)}")


if __name__ == "__main__":
    main()
