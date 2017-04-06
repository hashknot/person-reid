#!/usr/bin/env python

import numpy as np
import os
import random
import sys

from scipy.ndimage import imread

dataset = sys.argv[1]
outputdir = sys.argv[2]
minibatches = 4
minibatch_size = 500

cam_a = os.path.join(dataset, 'cam_a')
cam_b = os.path.join(dataset, 'cam_b')
dump_image = lambda p, fh: imread(p).tofile(fh)

a_path = lambda p: os.path.join(dataset, 'cam_a', p)
b_path = lambda p: os.path.join(dataset, 'cam_b', p)

a_files = os.listdir(cam_a)
b_files = os.listdir(cam_b)

a_files.sort()
b_files.sort()

records = []
random.seed()
for index, a_file in enumerate(a_files):
    b_file = b_files[index]

    a_array = imread(a_path(a_file))
    b_array = imread(b_path(b_file))
    records.append((a_array, b_array, np.uint8(1)))

    a2_index = b2_index = index
    while (a2_index == b2_index == index):
        a2_index = random.randint(0, len(a_files)-1)
        b2_index = random.randint(0, len(a_files)-1)

    records.append((a_array, imread(a_path(a_files[a2_index])), np.uint8(0)))
    records.append((a_array, imread(b_path(b_files[b2_index])), np.uint8(0)))
    records.append((b_array, imread(b_path(b_files[b2_index])), np.uint8(0)))

random.shuffle(records)

for i in xrange(minibatches):
    outfile_path = os.path.join(outputdir, 'data_batch_{}.bin'.format(i+1))
    with open(outfile_path, 'wb') as fh:
        for j in xrange(i*minibatch_size, (i+1)*minibatch_size):
            for r in records[j]:
                r.tofile(fh)

outfile_path = os.path.join(outputdir, 'data_test.bin')
with open(outfile_path, 'wb') as fh:
    for j in xrange((i+1)*minibatch_size, len(records)):
        for r in records[j]:
            r.tofile(fh)
