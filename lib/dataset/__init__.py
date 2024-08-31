'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.fdor import Fdor as fdor
from dataset.fdor_ssv import fdorSSV as fdor_ssv
from dataset.fdor_without_annotations import FdorWithoutAnnotations as fdor_without_annotations
from dataset.fdor_with_predictions import Fdor_with_predictions as fdor_with_predictions
from dataset.fdor_ssv_with_predictions import FdorSSV_with_predictions as fdor_ssv_with_predictions

from dataset.voxelpose_fdor import Voxelpose_fdor as voxelpose_fdor
from dataset.voxelpose_fdor_ssv import Voxelpose_fdor_ssv as voxelpose_fdor_ssv


from dataset.panoptic import Panoptic as panoptic
from dataset.panoptic_ssv import panopticSSV as panoptic_ssv
from dataset.shelf_synthetic import ShelfSynthetic as shelf_synthetic
from dataset.campus_synthetic import CampusSynthetic as campus_synthetic
from dataset.shelf import Shelf as shelf
from dataset.shelf_ssv import shelf_ssv as shelf_ssv
from dataset.campus import Campus as campus
from dataset.campus_ssv import campus_ssv as campus_ssv
