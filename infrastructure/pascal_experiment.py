#!/usr/bin/python
# coding=utf-8
"""
# I use this demo experiment to find out if experiments are usable

# Padding
x_pad = 100
y_pad = 100
z_pad = 0

# Image manipulation
resize_factor = 0.5
image_resize_mode = "bilinear"
label_resize_mode = "nearest"

# Subset
subset = "train"
allowed_classes = ['background', 'person', 'void']

# Padding
x_pad = 100
y_pad = 100
z_pad = 0

# Image manipulation
resize_factor = 0.5

# Subset
subset = "train"
allowed_classes = ['background', 'person', 'void']

[color]
    pad_mode = "reflect"
    resize_mode = "bilinear"
    filename = "seg_person_only-color-{set}-p{x_pad}x{y_pad}x{z_pad}-imageset.idx"
    grayscale = True

[gray]
    pad_mode = "reflect"
    resize_mode = "bilinear"
    filename = "seg_person_only-gray-{set}-p{x_pad}x{y_pad}x{z_pad}-imageset.idx"
    grayscale = False

[label]
    pad_mode = "constant"
    pad_fill = 255
    resize_mode = "nearest"
    grayscale = True
    filename = "seg_person_only-{set}-p{x_pad}x{y_pad}x{z_pad}-labels.idx"

"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.misc import imresize
import datasets.pascal as pc
from mlizard.experiment import createExperiment
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

get_classes_matrix = ex.full_stage(pc.get_classes_matrix)

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
    # TODO swap axis for color?
    return small_images

@ex.stage
def pad_images_and_equalize_sizes(images, x_pad, y_pad, z_pad, pad_mode, pad_fill=0):
    if len(images[0].shape) == 3:
        pad_width = (y_pad, x_pad, z_pad)
    else:
        pad_width = (y_pad, x_pad)
    return pc.pad_images_and_equalize_sizes(images, pad_width, pad_mode, constant_value=pad_fill)

@ex.stage
def reshape_images(images):
    i1 = images.shape[0]    # nr images
    i2 = 1                  # stack size
    #i3 = 3                  # channels
    i4 = images.shape[1]    # y
    i5 = images.shape[2]    # x
    out_array = images.reshape(i1, i2, i4, i5, -1)
    out_array = np.swapaxes(out_array, 3, 4)
    out_array = np.swapaxes(out_array, 2, 3)
    return out_array



def main(img_pad_mode, label_pad_mode):
    image_paths = get_image_path_list()
    label_paths = get_label_path_list()
    classes = get_classes_matrix(label_paths)
    image_paths = get_data_subset(image_paths, classes)
    label_paths = get_data_subset(label_paths, classes)
    paths = {'color':image_paths,
             'gray':image_paths,
             'label':label_paths}
    images = {}
    for t, p in paths.items():
        with ex.optionset(t) as o:
            images= load_and_resize_images(p)
            images= pad_images_and_equalize_sizes(images)
            images[t]=reshape_images(images)
            write_idx_file(o.options['filename'].format(**o.options), images[t])
