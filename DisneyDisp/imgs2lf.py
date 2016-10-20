#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import os

import numpy as np
import h5py
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from skimage.color import rgb2gray, gray2rgb
from miscpy import multiLoading, prepareSaving




def imgs2lf(input_dir, lf_file, lf_dataset='lightfield', img_extension = '.png', dtype=np.uint8, RGB=True):
    """
    Convert several images to a lightfield.

    Parameters
    ----------
    input_dir : string
        The directory where the ligthfield images are located.
    lf_file: string
        The filename  (including the directory), of the output file.
    lf_dataset : string, optional
        The new container name of the hdf5 file.
    img_extension : string, optional
        The file extension of the images to look for.
    dtype : numpy.dtype, optional
        The new data type for the downscaled lightfield. Must be either
        np.float64, np.uint8 or np.uint16.
    RGB : bool, optional
        If True, the output lightfield will be converted to RGB (default).
        Otherwise gray type images are stored.
    """

    # look for images
    files = multiLoading(identifier="*.{e}".format(e=img_extension), path=input_dir)

    # prepare saving
    lf_file = prepareSaving(lf_file, extension=".hdf5")

    # Which dtype should be used?
    if dtype == np.float64:
        img_0 = img_as_float(imread(files[0]))
    elif dtype == np.uint8:
        img_0 = img_as_ubyte(imread(files[0]))
    elif dtype == np.uint16:
        img_0 = img_as_uint(imread(files[0]))
    else:
        raise TypeError('The given data type is not supported!')


    # Do we shall take RGB or gray images?
    if (len(img_0.shape) == 3 and img_0.shape[2] == 3):
        rows, cols, orig_channels = img_0.shape  # automatically determine the images'shapes from the first image.
    elif len(img_0.shape) == 2:
        orig_channels = 1
        rows, cols = img_0.shape # automatically determine the images'shapes from the first image.
    else:
        raise TypeError('The given images are neither gray nor RGB images!')

    f_out = h5py.File(lf_file, 'w')
    if RGB:
        dataset = f_out.create_dataset(lf_dataset,
                                       shape=(len(files), rows, cols, 3),
                                       dtype=dtype)
    else:
        dataset = f_out.create_dataset(lf_dataset,
                                       shape=(len(files), rows, cols),
                                       dtype=dtype)
    for k in range(len(files)):

        if dtype == np.float64:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_float(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_float(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_float(imread(files[k]))
        elif dtype == np.uint8:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_ubyte(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_ubyte(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_ubyte(imread(files[k]))
        elif dtype == np.uint16:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_uint(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_uint(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_uint(imread(files[k]))
        else:
            raise TypeError('Given dtype not supported.')
    f_out.close()
