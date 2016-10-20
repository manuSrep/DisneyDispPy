#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
import h5py
from progressbar import Bar, ETA, Percentage, ProgressBar

from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from skimage.filters import gaussian
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
from miscpy import prepareLoading, prepareSaving




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

    v = np.ceil(float(r0_v) / float(red_fac))
    u = np.ceil(float(r0_u) / float(red_fac))

    while v > min_res and u > min_res:
        r_all.append([v,u])
        u = np.ceil(float(u) / float(red_fac))
        v = np.ceil(float(v) / float(red_fac))

    return np.array(r_all, dtype=np.uint16)




def downsample_lightfield(lf_in, lf_out, hdf5_dataset, r_all):
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
    hdf5_dataset: string
        The container name inside the hdf5 file for the lightfield. The same
        name will be used for the new file.
    r_all: array_like
        All resolutions to create. Each entry is a tuple (u,v) of resolutions.
    """

    # Initialize the hdf5 file objects
    lf_in = prepareLoading(lf_in)
    lf_out = prepareSaving(lf_out, extension=".hdf5")

    lf_in = h5py.File(lf_in, 'r')
    lf_out = h5py.File(lf_out, 'w')

    data_in = lf_in[hdf5_dataset]
    # Find out what data we have
    if len(data_in.shape) == 4 and data_in.shape[-1] == 3:
        RGB = True
    elif len(data_in.shape) == 3:
        RGB = False
    else:
        raise TypeError('The given lightfield contains neither gray nor RGB images!')

    # Which dtype should be used?
    if data_in.dtype == np.float64:
        DTYPE = np.float64
    elif data_in.dtype == np.uint8:
        DTYPE = np.uint8
    elif data_in. dtype == np.uint16:
        DTYPE = np.uint16
    else:
        raise TypeError('The given data type is not supported!')


    # We need to store all resolutions
    grp_out = lf_out.create_group(hdf5_dataset)
    grp_out.attrs.create('resolutions', r_all)

    # Initialize a progress bar to follow the downsampling
    widgets = ['Downscale lightfield: ', Percentage(), ' ', Bar(),' ', ETA(), ' ']
    progress = ProgressBar(widgets=widgets, max_val=r_all.shape[0]).start()

    for r,res in enumerate(r_all):

        if RGB:
            data_out = grp_out.create_dataset(str(res[0]) + 'x' + str(res[1]), shape=(data_in.shape[0], res[0], res[1], data_in.shape[3]), dtype=data_in.dtype)
        else:
            data_out = grp_out.create_dataset(str(res[0]) + 'x' + str(res[1]), shape=(data_in.shape[0], res[0], res[1]), dtype=data_in.dtype)

        if r == 0: # at lowest resolution we take the original image
            for s in range(data_in.shape[0]):
                data_out[s] = img_as_float(data_in[s])
        else: # we smooth the imput data
            data_prior = grp_out[str(r_all[r-1][0]) + 'x' + str(r_all[r-1][1])]
            for s in range(data_in.shape[0]):
                data_smoothed = img_as_float(gaussian(data_prior[s], sigma=np.sqrt(0.5), multichannel=True))
                if DTYPE is np.float64:
                    data_out[s] = img_as_float(resize(data_smoothed, (res[0], res[1])))
                elif DTYPE is np.uint16:
                    data_out[s] = img_as_uint(resize(data_smoothed, (res[0], res[1])))
                else:
                    data_out[s] = img_as_ubyte(resize(data_smoothed, (res[0], res[1])))
        
        progress.update(r)
    progress.finish()

    # Cleanup
    lf_in.close()
    lf_out.close()




def create_epis(lf_in, epi_out, hdf5_dataset_in="lightfield", hdf5_dataset_out="epis", dtype=np.float64, RGB=True):
    """
    Create epis for all resolutions given by the input lightfield.

    Parameters
    ----------
    lf_in : string
        The input hdf5 filename (including the directory) of the lightfield.
    epi_out : string
        The output hdf5 filename (including the directory) of the lightfield
        in all resolutions.
    hdf5_dataset_in: string
        The container name inside the hdf5 file for the lightfield. The same
        name will be used for the new file.
    hdf5_dataset_out: string, optional
        The container name inside the hdf5 file for the epis.
    dtype : numpy.dtype, optional
        The new data type for the epis. Must be either
        np.float64, np.uint8 or np.uint16.
    RGB : bool, optional
        If True, the output epis will be converted to RGB (default).
        Otherwise gray type images are stored.
    """

    # Initialze the hdf5 file objects
    lf_in = prepareLoading(lf_in)
    epi_out = prepareSaving(epi_out, extension=".hdf5")

    # Initialze the hdf5 file objects
    lf_in = h5py.File(lf_in, 'r')
    epi_out = h5py.File(epi_out, 'w')

    # Check if there is a resolution attribute. Create otherwise
    r_all = lf_in[hdf5_dataset_in].attrs.get('resolutions')[...]

    epi_grp = epi_out.create_group(hdf5_dataset_out)
    epi_grp.attrs.create('resolutions', r_all)


    # Initialize a progress bar to follow the conversion
    widgets = ['Create EPIs: ', Percentage(), ' ', Bar(),' ', ETA(), ' ']
    progress = ProgressBar(widgets=widgets, max_val=r_all.shape[0]).start()
    for r,res in enumerate(r_all):
        progress.update(r)

        set_name = str(res[0]) + 'x' + str(res[1])
        lf_data = lf_in[hdf5_dataset_in + '/' + set_name]

        # Find out what data we have
        if len(lf_data.shape) == 4 and lf_data.shape[-1] == 3:
            OLDRGB = True
        elif len(lf_data.shape) == 3:
            OLDRGB = False
        else:
            raise TypeError(
                'The given lightfield contains neither gray nor RGB images!')

        if RGB:
            epi_data = epi_grp.create_dataset(set_name, shape=(res[0],lf_data.shape[0], res[1],3), dtype=dtype)
        else:
            epi_data = epi_grp.create_dataset(set_name, shape=(res[0],lf_data.shape[0], res[1]), dtype=dtype)


        for v in range(res[0]):

            if dtype == np.float64:
                if RGB and not OLDRGB:
                    epi_data[v] = img_as_float(gray2rgb(lf_data[:,v,])).reshape(epi_data[v].shape)
                elif not RGB and OLDRGB:
                    epi_data[v] = img_as_float(rgb2gray(lf_data[:,v,])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_float(lf_data[:,v,...]).reshape(epi_data[v].shape)
            elif dtype == np.uint16:
                if RGB and not OLDRGB:
                    epi_data[v] = img_as_uint(gray2rgb(lf_data[:,v,])).reshape(epi_data[v].shape)
                elif not RGB and OLDRGB:
                    epi_data[v] = img_as_uint(rgb2gray(lf_data[:,v,])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_uint(lf_data[:,v,...]).reshape(epi_data[v].shape)
            elif dtype == np.uint8:
                if RGB and not OLDRGB:
                    epi_data[v] = img_as_ubyte(gray2rgb(lf_data[:,v,...])).reshape(epi_data[v].shape)
                elif not RGB and OLDRGB:
                    epi_data[v] = img_as_ubyte(rgb2gray(lf_data[:,v,...])).reshape(epi_data[v].shape)
                else:
                    epi_data[v] = img_as_ubyte(lf_data[:,v,...]).reshape(epi_data[v].shape)
            else:
                raise TypeError('Given dtype not supported.')

    progress.finish()

    # Cleanup
    lf_in.close()
    epi_out.close()