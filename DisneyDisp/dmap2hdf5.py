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
import struct
import argparse

import numpy as np
import h5py
from easyScripting import multiLoading, prepareSaving


def dmap2hdf5(input_dir, output_file, output_dataset='disparities'):
    """
    Convert results from .dmap files into .hdf5 file.

    Parameters
    ----------
    input_dir : string
        The directory where the .dmap files are located.
    output_file: string
        The filename  (including the directory), of the output file.
    output_dataset : string, optional
        The new container name of the hdf5 file.
    """

    files = multiLoading(directory=input_dir, extension=".dmap")

    # prepare saving
    output_name = os.path.basename(output_file)
    output_dir = os.path.dirname(output_file)
    output_file = prepareSaving(output_name, path=output_dir, extension=".hdf5")

    f_out = h5py.File(output_file, 'w')


    # we need to know the dimensions to store
    with open(files[0], 'rb') as f:  # open file
        b = f.read()  # read bytes
        # print("#bytes: {b}".format(b=len(b)))
        w, h = struct.unpack('<2i', b[0:8])  # width and hight
        # print("width:  {w}, heigth: {h}".format(w=w, h=h))

    f_out.create_dataset(output_dataset, shape=(len(files), h, w),
                    dtype=np.float64)

    # load all data from file
    for f, file in enumerate(files):
        with open(file, 'rb') as fl:  # open file
            b = fl.read()  # read bytes
            # print("#bytes: {b}".format(b=len(b)))
            w, h = struct.unpack('<2i', b[0:8])  # width and hight
            # print("width:  {w}, heigth: {h}".format(w=w, h=h))
            data = struct.unpack('<' + str(w * h) + 'f', b[0:w * h * 4])  # data
            data = np.array(data, np.float32).reshape((w, h))
            # print("dimensions: {w}x{h}".format(w=data.shape[0], h=data.shape[1]))

        f_out[f] = data


    f_out.close()




################################################################################
#                                                                              #
#                       Can be used as a command line tool                     #
#                                                                              #
################################################################################



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert results from .dmap files into .hdf5 file.')

    parser.add_argument('input_dir', help='The directory where the .dmap files are located.')
    parser.add_argument('output_file', help='TThe filename  (including the directory), of the output file.')
    parser.add_argument('--output_dataset', help='The container name in the hdf5 file.', default='disparities')

    args = parser.parse_args()

    dmap2hdf5(args.input_dir, args.output_file, output_dataset=args.output_dataset)