#!/usr/bin/python
# coding=utf-8
"""
Methods to load and edit the Pascal VOC dataset.
"""
from __future__ import division, print_function, unicode_literals
from config import PASCAL_PATH
import os.path as path

IMAGE_SET_DIR          = path.join(PASCAL_PATH, "ImageSets")
SEGMENTATION_SET_DIR   = path.join(PASCAL_PATH, "ImageSets/Segmentation")
IMAGES_DIR             = path.join(PASCAL_PATH, "JPEGImages")
SEGMENTATION_CLASS_DIR = path.join(PASCAL_PATH, "SegmentationClass")
SEGMENTATION_OBJ_DIR   = path.join(PASCAL_PATH, "SegmentationObject")


seg_train_set_file    = path.join(SEGMENTATION_SET_DIR, "train.txt")
seg_val_set_file      = path.join(SEGMENTATION_SET_DIR, "val.txt")
seg_trainval_set_file = path.join(SEGMENTATION_SET_DIR, "trainval.txt")

# classes
BACKGROUND, AEROPLANE, BICYCLE, BIRD, BOAT, BOTTLE, BUS, CAR, CAT, CHAIR, \
COW, TABLE, DOG, HORSE, MOTORBIKE, PERSON, PLANT, SHEEP, SOFA, TRAIN, TV = range(21)


def get_image_paths_for(img_path, set_filename, ending):
    with open(set_filename, "r") as f:
        img_names = f.readlines()
    for n in img_names:
        yield path.join(img_path, n.strip() + ending)

get_seg_train_image_files    = lambda : get_image_paths_for(IMAGES_DIR, seg_train_set_file, ".jpg")
get_seg_val_image_files      = lambda : get_image_paths_for(IMAGES_DIR, seg_val_set_file, ".jpg")
get_seg_trainval_image_files = lambda : get_image_paths_for(IMAGES_DIR, seg_trainval_set_file, ".jpg")

get_seg_train_class_label_files    = lambda : get_image_paths_for(SEGMENTATION_CLASS_DIR, seg_train_set_file, ".png")
get_seg_val_class_label_files      = lambda : get_image_paths_for(SEGMENTATION_CLASS_DIR, seg_val_set_file, ".png")
get_seg_trainval_class_label_files = lambda : get_image_paths_for(SEGMENTATION_CLASS_DIR, seg_trainval_set_file, ".png")

get_seg_train_obj_label_files    = lambda : get_image_paths_for(SEGMENTATION_OBJ_DIR, seg_train_set_file, ".png")
get_seg_val_obj_label_files      = lambda : get_image_paths_for(SEGMENTATION_OBJ_DIR, seg_val_set_file, ".png")
get_seg_trainval_obj_label_files = lambda : get_image_paths_for(SEGMENTATION_OBJ_DIR, seg_trainval_set_file, ".png")
