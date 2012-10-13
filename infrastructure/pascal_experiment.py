#!/usr/bin/python
# coding=utf-8
"""
# I use this demo experiment to find out if experiments are usable

# Padding
x_pad = 100
y_pad = 100
z_pad = 0

img_pad_mode = "reflect"
label_pad_mode = "constant"
pad_fill = 255

# Image manipulation
resize_factor = 0.5
image_resize_mode = "bilinear"
label_resize_mode = "nearest"

# Subset
subset = "train"
allowed_classes = ['background', 'person', 'void']

# filenames
prefix_scheme = "seg_person_only-{color}-{set}-p{x_pad}x{y_pad}x{z_pad}-"


######################
# i think this should in fact look something like this, and then have the ability to repeat
# stages for different sections
# maybe like this:
# with ex.optionset("color") as o:
#     o.just_call_stage() # and get all the options from the color section
# for o in ex.optionsets(["color", "gray", "label"]):
#     o.just_call_stage()
#
# We could even implement optionsweep like this
# for o in ex.optionsweep(["gamma", "C", "lambda"], kernel=["linear", "RBF"])
#     o.call_stage()

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

[gray]
    pad_mode = "reflect"
    resize_mode = "bilinear"
    filename = "seg_person_only-gray-{set}-p{x_pad}x{y_pad}x{z_pad}-imageset.idx"

[label]
    pad_mode = "constant"
    pad_fill = 255
    resize_mode = "nearest"
    filename = "seg_person_only-{set}-p{x_pad}x{y_pad}x{z_pad}-labels.idx"
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.misc import imresize
import datasets.pascal as pc
from experiment import Experiment

ex = Experiment(__doc__)

@ex.full_stage
def get_image_path_list(subset):
    return list({'train' : pc.get_seg_train_image_files,
                 'val'   : pc.get_seg_val_image_files,
                 'all'   : pc.get_seg_trainval_image_files}[subset]())

@ex.full_stage
def get_label_path_list(subset):
    return list({'train' : pc.get_seg_train_class_label_files,
                 'val'   : pc.get_seg_val_class_label_files,
                 'all'   : pc.get_seg_trainval_class_label_files}[subset]())

get_classes_matrix = ex.full_stage(pc.get_classes_matrix)

@ex.full_stage
def get_data_subset(image_paths, label_paths, classes, allowed_classes, subset, logger):
    logger.info("Loading {} subset of PascalVOC containing only {}.".format(subset, allowed_classes))
    allowed_class_ints = [pc.CLASS_LIST.index(c.lower()) for c in allowed_classes]
    valid_idx = pc.get_idx_containing_only(classes,  allowed_class_ints)
    image_paths = pc.get_image_subset(image_paths, valid_idx)
    label_paths = pc.get_image_subset(label_paths, valid_idx)
    logger.info("Got {} Images.".format(len(image_paths)))
    return image_paths, label_paths

@ex.full_stage
def load_and_resize_images(img_paths, label_paths, resize_factor, image_resize_mode, label_resize_mode, logger):
    logger.info("Resizing the images by {}".format(resize_factor))
    images_gray = pc.load_images_as_ndarrays(img_paths, grayscale=True)
    images_color = pc.load_images_as_ndarrays(img_paths, grayscale=False)
    labels = pc.load_images_as_ndarrays(label_paths)
    small_images_color = map(lambda x : imresize(x, resize_factor, interp=image_resize_mode), images_color)
    small_images_gray = map(lambda x : imresize(x, resize_factor, interp=image_resize_mode), images_gray)
    small_labels = map(lambda x : imresize(x, resize_factor, interp=label_resize_mode), labels)
    return small_images_color, small_images_gray, small_labels

@ex.full_stage
def pad_images_and_equalize_sizes(images, x_pad, y_pad, z_pad, pad_mode, pad_fill):
    if len(images[0].shape) == 3:
        pad_width = (y_pad, x_pad, z_pad)
    else:
        pad_width = (y_pad, x_pad)
    return pc.pad_images_and_equalize_sizes(images, pad_width, pad_mode, constant_value=pad_fill)

def main(img_pad_mode, label_pad_mode):
    image_paths = get_image_path_list()
    label_paths = get_label_path_list()
    classes = get_classes_matrix(label_paths)
    image_paths, label_paths = get_data_subset(image_paths, label_paths, classes)
    colors, grays, labels = load_and_resize_images(image_paths, label_paths)
    colors = pc.pad_images_and_equalize_sizes(colors, pad_mode=img_pad_mode)
    grays = pc.pad_images_and_equalize_sizes(grays, pad_mode=img_pad_mode)
    labels = pc.pad_images_and_equalize_sizes(labels, pad_mode=label_pad_mode)




