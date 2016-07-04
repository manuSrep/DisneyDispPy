#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Disney Disparity.
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import os
import argparse

import numpy as np
import h5py
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from skimage.color import rgb2gray, gray2rgb
from easyScripting import multiLoading, prepareSaving




def imgs2lf(input_dir, output_file, output_dataset='lightfield', img_extension = '.png', dtype=np.uint8, RGB=True):
    """
    Convert several images to a lightfield.

    Parameters
    ----------
    input_dir : string
        The directory where the ligthfield images are located.
    output_file: string
        The filename  (including the directory), of the output file.
    output_dataset : string, optional
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
    files = multiLoading(directory=input_dir, extension=img_extension)

    # prepare saving
    output_name = os.path.basename(output_file)
    output_dir = os.path.dirname(output_file)
    output_file = prepareSaving(output_name, path=output_dir, extension=".hdf5")


    # Which dtye should be used?
    if dtype is np.float64:
        img_0 = img_as_float(imread(files[0]))
    elif dtype is np.uint8:
        img_0 = img_as_ubyte(imread(files[0]))
    elif dtype is np.uint16:
        img_0 = img_as_uint(imread(files[0]))
    else:
        raise TypeError('Given dtype not supported.')


    rows, cols, orig_channels = img_0.shape # automatically determine the images'shapes from the first image.


    # Do we shall take RGB or gray images?
    if orig_channels != 3 or orig_channels != 1:
        raise TypeError('The given images are neither gray nor RGB images')
    if RGB:
        channels = 3
    else:
        channels = 1


    f_out = h5py.File(output_file, 'w')
    dataset = f_out.create_dataset(output_dataset, shape=(len(files), rows, cols, channels), dtype=dtype)
    for k in range(len(files)):

        if dtype is np.float64:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_float(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_float(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_float(imread(files[k]))
        elif dtype is np.uint8:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_ubyte(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_ubyte(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_ubyte(imread(files[k]))
        elif dtype is np.uint16:
            if orig_channels==1 and RGB:
                dataset[k, ...] = img_as_uint(gray2rgb(imread(files[k])))
            elif orig_channels==3 and not RGB:
                dataset[k, ...] = img_as_uint(rgb2gray(imread(files[k])))
            else:
                dataset[k, ...] = img_as_uint(imread(files[k]))
        else:
            raise TypeError('Given dtye not supported.')
    f_out.close()




################################################################################
#                                                                              #
#                       Can be used as a command line tool                     #
#                                                                              #
################################################################################



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Store image files into hdf5 container.')

    parser.add_argument('input_dir', help='The directory where the lightfield images are located.')
    parser.add_argument('--output_file', help='The filename (including the directory) of the output .hdf5 file.', default='lightfield')
    parser.add_argument('--output_dataset', help='The container name in the hdf5 file.', default='lightfield')
    parser.add_argument('--img_extension', help='The file extension of the images to look for.', default='.png')
    parser.add_argument('--dtype', help='The dtype used to store the lightfild data.', choices=['float', 'uint', 'ubyte'], default='uint')
    parser.add_argument('-RGB', help='Flag to determine if resulting lightfield should consists of RGB values.', action='store_true')

    args = parser.parse_args()

    type = np.float64
    if args.dtype == 'ubyte':
        type = np.uint8
    if args.dtype == 'uint':
        type = np.uint16

    imgs2lf(args.input_dir, output_file=args.output_file, output_dataset=args.output_dataset, img_extension=args.img_extension, dtype=type, RGB=args.RGB)