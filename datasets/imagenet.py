#!/usr/bin/python
# coding=utf-8
# This file is part of the MLizard library published under the GPL3 license.
# Copyright (C) 2012  Klaus Greff
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
seed = 380434
ds_path = "/home/greff/Datasets/ImageNet/"
used_classes = [1, 2, 3, 4]
samples_per_class = 10
shuffle = True

"""
from __future__ import division, print_function, unicode_literals
from os.path import join
import numpy as np
from mlizard import createExperiment

ex = createExperiment("ImageNet", config_string=__doc__)

@ex.stage
def get_image_paths_and_classes(ds_path, logger):
    path = join(ds_path, "trainfiles.txt" )
    f = open(path, 'r')
    lines = f.readlines()
    lines = map(lambda s : s.split(None, 1), lines)
    paths =  np.array([join(ds_path, p) for p, c in lines])
    targets = np.array([[int(c)] for p, c in lines])
    logger.info("Opened path '%s' and got %d samples.", path, len(lines))
    return paths, targets

@ex.stage
def get_subset_indices(targets, used_classes, samples_per_class, shuffle, rnd):
    indices_for_class = []
    for c in used_classes:
        t = np.nonzero(targets.flatten() == c)[0]
        if shuffle:
            np.random.shuffle(t)
        indices_for_class.append(t[:samples_per_class])
    return indices_for_class

@ex.main
def main():
    paths, targets = get_image_paths_and_classes()
    cidx = get_subset_indices(targets)

