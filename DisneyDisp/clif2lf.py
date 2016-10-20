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
from miscpy import prepareLoading, prepareSaving




def clif2lf(clif_file, lf_file, clif_group, lf_dataset="lightfield" ):
    """
    Convert a standard .clif file to an .hdf5 lightfield file.

    Parameters
    ----------
    clif_file : string
        The .clif filename including the directory.
    lf_file : string
        The filename (including the directory) of the output .hdf5 lightfield.
    clif_group : string
        The container name inside the .clif file.
    lf_dataset : string, optional
        The dataset name inside the .hdf5 file for the lightfield.
    """

    # Initialze the hdf5 file objects
    fname_in = os.path.basename(clif_file)
    dir_in = os.path.dirname(clif_file)
    clif_file = prepareLoading(fname_in, path=dir_in)

    fname_out = os.path.basename(lf_file)
    dir_out = os.path.dirname(lf_file)
    lf_file = prepareSaving(fname_out, path=dir_out, extension=".hdf5")

    data = h5py.File(clif_file, 'r')[clif_group]
    data = np.swapaxes(data, 1, 3 )
    data = np.swapaxes(data, 1, 2)

    f_out = h5py.File(lf_file, 'w')
    f_out.create_dataset(lf_dataset, data=data)
    f_out.close()