#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import os
import struct

import numpy as np
import h5py
from miscpy import multiLoading, prepareSaving


def dmap2hdf5(input_dir, disp_file, output_dataset='disparities'):
    """
    Convert results from .dmap files into .hdf5 file.

    Parameters
    ----------
    input_dir : string
        The directory where the .dmap files are located.
    disp_file: string
        The filename  (including the directory), of the output file.
    output_dataset : string, optional
        The new container name of the hdf5 file.
    """

    files = multiLoading(identifier="*.dmap", path=input_dir)

    # prepare saving
    output_file = prepareSaving(disp_file, extension=".hdf5")
    f_out = h5py.File(output_file, 'w')

    # we need to know the dimensions to store
    with open(files[0], 'rb') as f:  # open file
        b = f.read()  # read bytes
        #print("#bytes: {b}".format(b=len(b)))
        w, h = struct.unpack('<2i', b[0:8])  # width and hight
        #print("width:  {w}, heigth: {h}".format(w=w, h=h))

    f_out.create_dataset(output_dataset, shape=(len(files), h, w), dtype=np.float32)

    # load all data from file
    for f, file in enumerate(files):
        with open(file, 'rb') as fl:  # open file
            b = fl.read()  # read bytes
            #print("#bytes: {b}".format(b=len(b)))
            w, h = struct.unpack('<2i', b[0:8])  # width and hight
            #print("width:  {w}, heigth: {h}".format(w=w, h=h))
            data = struct.unpack('<' + str(w * h) + 'f', b[0:w * h * 4])  # data
            data = np.array(data, dtype=np.float32).reshape((h, w))
            #print("dimensions: {w}x{h}".format(w=data.shape[1], h=data.shape[0]))

        f_out[output_dataset][f] = data

    f_out.close()

