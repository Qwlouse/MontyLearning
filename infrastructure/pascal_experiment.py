#!/usr/bin/python
# coding=utf-8
"""
# I use this demo experiment to find out if experiments are usable

# Padding
x_pad = 50
y_pad = 50
z_pad = 0

# Subset
subset = "val"
#allowed_classes = ['background', 'person', 'void']
allowed_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'tv', 'void']
#prefix = "seg_person_only"
prefix = "seg_167x167"
#prefix = "seg_250x250"

# Image manipulation
resize_factor = 0.3333333333333333

# samples_stuff
all_samples_filename = "{prefix}_{subset}-p{x_pad}x{y_pad}x{z_pad}-1M_background_samples.idx"

[color]
    pad_mode = "reflect"
    resize_mode = "bicubic"
    filename = "{prefix}_color_{subset}-p{x_pad}x{y_pad}x{z_pad}-imagedata.idx"
    grayscale = False
    use_channel_dim = True

[gray]
    pad_mode = "reflect"
    resize_mode = "bicubic"
    filename = "{prefix}_grayscale_{subset}-p{x_pad}x{y_pad}x{z_pad}-imagedata.idx"
    grayscale = True
    use_channel_dim = True

[label]
    pad_mode = "constant"
    pad_fill = 255
    resize_mode = "nearest"
    grayscale = False
    filename = "{prefix}_{subset}-p{x_pad}x{y_pad}x{z_pad}-labeldata.idx"
    use_channel_dim = False
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from scipy.misc import imresize
import datasets.pascal as pc
from mlizard import createExperiment
from infrastructure.idxconverter import write_idx_file
ex = createExperiment(config_string=__doc__)

@ex.stage
def get_image_path_list(subset):
    return list({'train' : pc.get_seg_train_image_files,
                 'val'   : pc.get_seg_val_image_files,
                 'all'   : pc.get_seg_trainval_image_files}[subset]())

@ex.stage
def get_label_path_list(subset):
    return list({'train' : pc.get_seg_train_class_label_files,
                 'val'   : pc.get_seg_val_class_label_files,
                 'all'   : pc.get_seg_trainval_class_label_files}[subset]())

get_classes_matrix = ex.stage(pc.get_classes_matrix)

@ex.stage
def get_data_subset(image_paths, classes, allowed_classes, subset, logger):
    logger.info("Loading {} subset of PascalVOC containing only {}.".format(subset, allowed_classes))
    allowed_class_ints = [pc.CLASS_LIST.index(c.lower()) for c in allowed_classes]
    valid_idx = pc.get_idx_containing_only(classes,  allowed_class_ints)
    image_paths = pc.get_image_subset(image_paths, valid_idx)
    logger.info("Got {} Images.".format(len(image_paths)))
    return image_paths

@ex.stage
def load_and_resize_images(img_paths, resize_factor, resize_mode, grayscale, logger):
    logger.info("Resizing the images by {}".format(resize_factor))
    images= pc.load_images_as_ndarrays(img_paths, grayscale=grayscale)
    small_images = map(lambda x : imresize(x, resize_factor, interp=resize_mode), images)
    # ensure 3d with (height, width, channels)
    for i in range(len(small_images)):
        if len(small_images[i].shape) < 3:
            height, width = small_images[i].shape
            small_images[i] = small_images[i].reshape(height, width, 1)
    # swap axis
    for i in range(len(small_images)):
        small_images[i] = np.swapaxes(small_images[i], 1, 2)
        small_images[i] = np.swapaxes(small_images[i], 0, 1)
    return small_images

@ex.stage
def pad_images_and_equalize_sizes(images, x_pad, y_pad, pad_mode, pad_fill=None):
    pad_width = (y_pad, x_pad)
    kwargs = {'constant_values' : pad_fill} if pad_fill is not None else {}
    return pc.pad_images_and_equalize_sizes_swapped(images, pad_width, pad_mode, **kwargs)


@ex.stage
def reshape_images(images, use_channel_dim):
    i1 = images.shape[0]    # nr images
    i2 = 1                  # stack size
    i3 = images.shape[1]   # channels
    i4 = images.shape[2]    # y
    i5 = images.shape[3]    # x
    if use_channel_dim:
        out_array = images.reshape(i1, i2, i3, i4, i5)
    else :
        out_array = images.reshape(i1, i2, i4, i5)
    return out_array

@ex.stage
def create_samples(labels_array, logger, void_class=255):
    classes = np.unique(labels_array)
    classes = classes[classes != void_class] # remove void-class
    class_samples = []
    for c in classes:
        samples = np.array(np.nonzero(labels_array == c), dtype=np.int32).T
        N = samples.shape[0]
        # add class label
        samples = np.hstack((samples, np.ones((N,1), dtype=np.int32)*c))
        # reorder to meet idx specs (x, y, z, img_nr, label)
        samples = samples[:,[3, 2, 1, 0, 4]]
        class_samples.append(samples)
        logger.info("Class %d : %d samples", c, N)
    return class_samples

@ex.stage
def write_samples_to_idx(samples, filename):
    np.random.shuffle(samples)
    write_idx_file(filename, samples, byteswap=False)


@ex.main
def main():
    image_paths = get_image_path_list()
    label_paths = get_label_path_list()
    classes = get_classes_matrix(label_paths)
    image_paths = get_data_subset(image_paths, classes)
    label_paths = get_data_subset(label_paths, classes)
    paths = {'color':image_paths,
             'gray' :image_paths,
             'label':label_paths}
    # prepare images
    image_dict = {}
    for t, p in paths.items():
        with ex.optionset(t) as o:
            images= o.load_and_resize_images(p)
            images= o.pad_images_and_equalize_sizes(images)
            images=o.reshape_images(images)
            write_idx_file(o.options['filename'].format(**o.options), images)
            image_dict[t] = images


    # create samples
    class_samples = create_samples(image_dict['label'])
    # reduce number of background samples
    np.random.shuffle(class_samples[0])
    class_samples[0] = class_samples[0][:1000000]
    write_samples_to_idx(np.vstack(tuple(class_samples)), ex.options['all_samples_filename'].format(**ex.options))


