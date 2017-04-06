#!/usr/bin/env python

import numpy as np
import os
import random
import sys

from scipy.ndimage import imread

dataset = sys.argv[1]
output = sys.argv[2]

cam_a = os.path.join(dataset, 'cam_a')
cam_b = os.path.join(dataset, 'cam_b')
dump_image = lambda p, fh: imread(p).tofile(fh)

a_path = lambda p: os.path.join(dataset, 'cam_a', p)
b_path = lambda p: os.path.join(dataset, 'cam_b', p)

a_files = os.listdir(cam_a)
b_files = os.listdir(cam_b)

a_files.sort()
b_files.sort()

with open(output, 'wb') as fh:
    for index, a_file in enumerate(a_files):
        b_file = b_files[index]

        dump_image(a_path(a_file), fh)
        dump_image(b_path(b_file), fh)
        np.uint8(1).tofile(fh)          # label

        b2_file = random.choice([f for i, f in enumerate(b_files) if i != index])
        dump_image(a_path(a_file), fh)
        dump_image(b_path(b2_file), fh)
        np.uint8(0).tofile(fh)          # label
