#!/usr/bin/env python2.7

import numpy as np
import os
import random
import sys

from scipy.ndimage import imread

dataset = sys.argv[1]
outputdir = sys.argv[2] if len(sys.argv) >= 3 else 'data'
minibatches = 10
minibatch_size = 500

split_channels = lambda x: np.array((x[:,:,0], x[:,:,1], x[:,:,2]))

images = os.listdir(dataset)

images.sort()

image_groups = [images[i:i+4] for i in xrange(0, len(images), 4)]
path = lambda p: os.path.join(dataset, p)

records = []
random.seed()
for image_group in image_groups:
    random.shuffle(image_group)
    a_file, b_file = image_group[:2]

    a_array = split_channels(imread(path(a_file)))
    b_array = split_channels(imread(path(b_file)))
    records.append((a_array, b_array, np.uint8(1)))

    a_file, b_file = image_group[2:]

    a_array = split_channels(imread(path(a_file)))
    b_array = split_channels(imread(path(b_file)))
    records.append((a_array, b_array, np.uint8(1)))

images = os.listdir(dataset)
random.shuffle(images)

images_2 = os.listdir(dataset)
random.shuffle(images_2)

for a_file, b_file in zip(images, images_2):
    a_array = split_channels(imread(path(a_file)))
    b_array = split_channels(imread(path(b_file)))
    label = 1 if a_file[:4] == b_file[:4] else 0
    records.append((a_array, b_array, np.uint8(0)))

random.shuffle(records)

for i in xrange(minibatches):
    outfile_path = os.path.join(outputdir, 'data_batch_{}.bin'.format(i+1))
    with open(outfile_path, 'wb') as fh:
        for j in xrange(i*minibatch_size, (i+1)*minibatch_size):
            for r in records[j]:
                r.tofile(fh)

outfile_path = os.path.join(outputdir, 'data_test.bin')
with open(outfile_path, 'wb') as fh:
    for j in xrange(minibatches*minibatch_size, len(records)):
        for r in records[j]:
            r.tofile(fh)
