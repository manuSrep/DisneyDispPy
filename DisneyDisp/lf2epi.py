#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import os
import math
import argparse

import numpy as np
import h5py
from progressbar import Bar, ETA, Percentage, ProgressBar

from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
from easyScripting import prepareLoading, prepareSaving




def calculate_resolutions(r0_v, r0_u, red_fac=2, min_res=11):
    """
    Create a list of downsampled resolutions (v,u) by reducing the initial
    resolution by a constant factor for each dimension up to a minimal
    resolution.

    Parameters
    ----------
    r0_v : uint16
        The initial resolution in v-dimension.
    r0_u : uint16
        The initial resolution in u-dimension.
    red_fac : uint , optional
        The reduction factor used for down sampling. Default is to halve the
        resolution each time.
    min_res: uint16.
        The minimal resolution to sample to. The program will stop when either u
        or v reach min_res.

    Returns
    -------
    array_like
        Each entry is a tuple (u,v) of resolutions.
    """

    r_all = [[r0_v, r0_u]] # initialize the output list

    v = math.ceil(float(r0_v) / float(red_fac))
    u = math.ceil(float(r0_u) / float(red_fac))

    while v > min_res and u > min_res:
        r_all.append([v,u])
        u = math.ceil(float(u) / float(red_fac))
        v = math.ceil(float(v) / float(red_fac))

    return np.array(r_all, dtype=np.uint16)




def downsample_lightfield(lf_in, lf_out, hdf5_group, r_all):
    """
    Reduces the dimension of the input lightfield to the values given.
    Results are stored in a new hdf5 file.

    Parameters
    ----------
    lf_in : string
        The input hdf5 filename (including the directory) of the lightfield.
    lf_out : string
        The output hdf5 filename (including the directory) of the lightfield
        in all resolutions.
    hdf5_group: string
        The container name inside the hdf5 file for the lightfield. The same
        name will be used for the new file.
    r_all: array_like
        All resolutions to create. Each entry is a tuple (u,v) of resolutions.
    """

    # Initialze the hdf5 file objects
    fname_in = os.path.basename(lf_in)
    dir_in = os.path.dirname(lf_in)
    lf_in = prepareLoading(fname_in, path=dir_in)

    fname_out = os.path.basename(lf_out)
    dir_out = os.path.dirname(lf_out)
    lf_out = prepareSaving(fname_out, path=dir_out, extension=".hdf5")

    lf_in = h5py.File(lf_in, 'r')
    lf_out = h5py.File(lf_out, 'w')

    data_in = lf_in[hdf5_group]
    
    
    # We need to store all resolutions
    grp_out = lf_out.create_group(hdf5_group)
    grp_out.attrs.create('resolutions', r_all)
    
    
    # Initialize a progress bar to follow the downsampling
    widgets = ['Downsaple lightfields: ', Percentage(), ' ', Bar(),' ', ETA(), ' ']
    progress = ProgressBar(widgets=widgets, maxval=r_all.shape[0]).start()

    for r,res in enumerate(r_all):
        
        data_out = grp_out.create_dataset(str(res[0]) + 'x' + str(res[1]), shape=(data_in.shape[0], res[0], res[1], data_in.shape[3]), dtype=np.float64)
            
        if r == 0: # at lowest resolution we take the original image
            for s in range(data_in.shape[0]):
                data_out[s] = img_as_float(data_in[s])
        else: # we smooth the imput data
            data_prior = grp_out[str(r_all[r-1][0]) + 'x' + str(r_all[r-1][1])]
            for s in range(data_in.shape[0]):
                data_smoothed = img_as_float(gaussian(data_prior[s], sigma=math.sqrt(0.5), multichannel=True))
                data_out[s] = resize(data_smoothed, (res[0], res[1]))
        
        progress.update(r)
    progress.finish()

    # Cleanup
    lf_in.close()
    lf_out.close()




def create_epis(lf_in, epi_out, hdf5_group_in, hdf5_group_out="epis",  dtype=np.float64, RGB=True):
    """
    Create epis for all resolutions given by the input lightfield.


    Parameters
    ----------
    lf_in : string
        The input hdf5 filename (including the directory) of the lightfield.
    epi_out : string
        The output hdf5 filename (including the directory) of the lightfield
        in all resolutions.
    hdf5_group_in: string
        The container name inside the hdf5 file for the lightfield. The same
        name will be used for the new file.
    hdf5_group_out: string, optional
        The container name inside the hdf5 file for the epis.
    dtype : numpy.dtype, optional
        The new data type for the epis. Must be either
        np.float64, np.uint8 or np.uint16.
    RGB : bool, optional
        If True, the output epis will be converted to RGB (default).
        Otherwise gray type images are stored.
    """

    # Initialze the hdf5 file objects
    fname_in = os.path.basename(lf_in)
    dir_in = os.path.dirname(lf_in)
    lf_in = prepareLoading(fname_in, path=dir_in)

    fname_out = os.path.basename(epi_out)
    dir_out = os.path.dirname(epi_out)
    epi_out = prepareSaving(fname_out, path=dir_out, extension=".hdf5")

    # Initialze the hdf5 file objects
    lf_in = h5py.File(lf_in, 'r')
    epi_out = h5py.File(epi_out, 'w')

    r_all = lf_in[hdf5_group_in].attrs.get('resolutions')[...]

    epi_grp = epi_out.create_group(hdf5_group_out)
    epi_grp.attrs.create('resolutions', r_all)



    # Initialize a progress bar to follow the conversion
    widgets = ['Create EPIs: ', Percentage(), ' ', Bar(),' ', ETA(), ' ']
    progress = ProgressBar(widgets=widgets, maxval=r_all.shape[0]).start()
    for r,res in enumerate(r_all):
        progress.update(r)

        set_name = str(res[0]) + 'x' + str(res[1])
        lf_data = lf_in[hdf5_group_in + '/' + set_name]

        # Do we shall take RGB or gray images?
        channels = lf_data.shape[-1]
        if channels != 3 and channels != 1:
            raise TypeError('The lightfield is neither gray nor RGB!')
        if RGB:
            epi_data = epi_grp.create_dataset(set_name, shape=(res[0],lf_data.shape[0], res[1],3), dtype=dtype)
        else:
            epi_data = epi_grp.create_dataset(set_name, shape=(res[0],lf_data.shape[0], res[1],), dtype=dtype)


        for v in range(res[0]):

            if dtype is np.float64:
                if RGB and channels == 1:
                    epi_data[v] = img_as_float(gray2rgb(lf_data[:,v,])).reshape(epi_data[v].shape)
                elif not RGB and channels == 3:
                    epi_data[v] = img_as_float(rgb2gray(lf_data[:,v,])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_float(lf_data[:,v,...]).reshape(epi_data[v].shape)
            elif dtype is np.uint16:
                if RGB and channels == 1:
                    epi_data[v] = img_as_uint(gray2rgb(lf_data[:,v,])).reshape(epi_data[v].shape)
                elif not RGB and channels == 3:
                    epi_data[v] = img_as_uint(rgb2gray(lf_data[:,v,])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_uint(lf_data[:,v,...]).reshape(epi_data[v].shape)
            elif dtype is np.uint8:
                if RGB and channels == 1:
                    epi_data[v] = img_as_ubyte(gray2rgb(lf_data[:,v,...])).reshape(epi_data[v].shape)
                elif not RGB and channels == 3:
                    epi_data[v] = img_as_ubyte(rgb2gray(lf_data[:,v,...])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_ubyte(lf_data[:,v,...]).reshape(epi_data[v].shape)
            else:
                raise TypeError('Given dtype not supported.')

    progress.finish()


    # Cleanup
    lf_in.close()
    epi_out.close()




################################################################################
#                                                                              #
#                       Can be used as a command line tool                     #
#                                                                              #
################################################################################


parser = argparse.ArgumentParser(description='Extract EPIs from a given lightfield and store them in a .hdf5 file.')

parser.add_argument('lightfiled', help='The filename including the directory of the lightfield.')
parser.add_argument('epis', help='The filename including the directory to save the EPI .hdf5 file.')
parser.add_argument('--hdf5_group_lf', help='The group name inside hdf5 File.', default='lightfield')
parser.add_argument('--hdf5_group_epi', help='The group name inside hdf5 File.', default='lightfield')
parser.add_argument('--dtype', help='The dtype used to store the lightfild data.', choices=[np.float64, np.uint8, np.uint16],default=np.float64)
parser.add_argument('-RGB',help='Flag to determine if resulting lightfield should consists of RGB values.',action='store_true')

if __name__ == "__main__":

    args = parser.parse_args()

    lf = h5py.File(args.lightfiled, 'r')
    r_all = calculate_resolutions(lf[args.h5path].shape[1], lf[args.h5path].shape[2])

    downsample_lightfield(args.lightfiled, 'tmp.hdf5', args.hdf5_group_lf, r_all)
    create_epis('tmp.hdf5', args.epi, args.hdf5_group_lf, hdf5_group_out=args.hdf5_group_epi, dtype=args.dtype, RGB=args.RGB)
    os.remove('tmp.hdf5')










