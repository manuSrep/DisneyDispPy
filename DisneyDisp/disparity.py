#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, \
    print_function

import os
import shutil
from itertools import repeat
from multiprocessing import Pool, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np
from miscpy import prepareLoading, prepareSaving
from progressbar import Bar, ETA, Percentage, ProgressBar
from scipy.ndimage.filters import median_filter
from skimage.color import gray2rgb
from skimage.io import imsave

from ._confidence import edge_confidence
from ._disparity import bilateral_median, propagation
from ._process_epi import process_epi, convert_process_epi
from ._scale import fine_to_course, course_to_fine
from .lf2epi import calculate_resolutions, downsample_lightfield, create_epis


class Disney():
    '''
    The class collecting all functionality and input parameters needed for the
    disparity calculation.
    '''

    def __init__(self, lightfield, lf_dataset, output_dir,
                 working_dir='work_tmp/',
                 n_cpus=-1, r_start=None, s_hat=None, DEBUG=False):

        """
        Constructor to settle up all files and parameters. Do also some pre-
        computations required

        Parameters
        ----------
        lightfield : string
            The filename of the lightfield including the directory.
        lf_dataset : string
            The group name inside the lightfield's hdf5 file.
        output_dir: string
            The directory to store the final results in.
        working_dir: string, optional
            A temporal directory to work in.
        n_cpus : int, optional
            The number of cpus to use. If -1 all cpus available will be used.
        r_start : tuble (), optional
            r_v and r_u to start with
        s_hat : int, optional
            If given, only this s-dimension will be calculated.
        DEBUG : Boolean, optional
            enable DEBUG output
        """

        lightfield = prepareLoading(lightfield)
        self.output_dir = prepareSaving(output_dir)
        self.working_dir = prepareSaving(working_dir)

        self.n_cpus = n_cpus
        if self.n_cpus == -1:
            self.n_cpus = cpu_count()

        self.DEBUG = DEBUG  # Plot intermediate results and other debugging output

        # Attributes of .hdf5 files to load or store the data
        self.lf_dataset = lf_dataset
        self.light_field = h5py.File(os.path.expanduser(lightfield), 'r')
        self.epi_field = h5py.File(os.path.join(self.working_dir, 'epis.hdf5'),'a')
        self.disp_field = h5py.File(
            os.path.join(self.working_dir, 'disparities.hdf5'), 'a')

        if self.DEBUG:
            self.score_field = h5py.File(
                os.path.join(self.working_dir, 'scores.hdf5'), 'a')
            self.DB_field = h5py.File(
                os.path.join(self.working_dir, 'disparity_bounds.hdf5'), 'a')
            self.Ce_field = h5py.File(
                os.path.join(self.working_dir, 'edge_confidences.hdf5'), 'a')
            self.Cd_field = h5py.File(
                os.path.join(self.working_dir, 'disparity_confidences.hdf5'),
                'a')

        # Runtime attributes
        self.lf_res = self.light_field[
            self.lf_dataset].shape  # The resolution of the original lightfield (s,v,u)
        self.epi_res = None  # A ndarray [r]. The different EPI resolutions (v,u)

        self.r_start = self.lf_res[1:3]
        if self.DEBUG:
            self.r_start = self.lf_res[
                           1:3] if r_start is None else r_start  # If given start calculation by this resolution
        self.s_hat = s_hat  # If not None the only scanline to process

        self.initialize()
        print("All data loaded!")

    def __del__(self):
        """
        Cleanup.
        """
        if not self.DEBUG:
            shutil.rmtree(self.working_dir)

    def initialize(self):
        """
        Load data from files or initialize if not available.
        """
        # 2) We need all epi data:
        if not self.epi_field.get('epis', default=False):
            lf_file = self.light_field.filename
            self.light_field.close()
            epi_file = self.epi_field.filename
            self.epi_field.close()
            tmp_file = os.path.join(self.working_dir, 'tmp.hdf5')
            epi_res = calculate_resolutions(self.lf_res[1], self.lf_res[2])
            downsample_lightfield(lf_file, tmp_file, self.lf_dataset, epi_res)
            create_epis(tmp_file, epi_file, self.lf_dataset, dtype=np.uint8,RGB=False)
            os.remove(tmp_file)
            self.light_field = h5py.File(lf_file, 'r')
            self.epi_field = h5py.File(epi_file, 'r')
        self.epi_res = self.epi_field['epis'].attrs.get('resolutions')

        # 3) Check r-start
        if not np.all(np.less_equal(self.epi_res[-1], self.r_start)):
            raise IOError('r-start dimension not found')
        SCRATCH = False
        if np.array_equal(self.lf_res[1:3], self.r_start) or not self.DEBUG:
            SCRATCH = True

        # 4) We need all disparity data
        for res in self.epi_res:
            res_name = str(res[0]) + 'x' + str(res[1])
            if res[0] >= self.r_start[0] and res[1] >= self.r_start[
                1] and not SCRATCH:  # There should be data in file
                if not self.disp_field['disparities'].get(res_name,
                                                          default=False):
                    raise IOError(
                        'Diparity file object does not contain enough data sets.')

        if self.DEBUG:
            # 5) We need the initial disparitiy bounds for that resolution we start
            for res in self.epi_res:
                res_name = str(res[0]) + 'x' + str(res[1])
                if res[0] >= self.r_start[0] and res[1] >= self.r_start[
                    1] and not SCRATCH:  # There should be data in file
                    if not self.DB_field['disp_bounds'].get(res_name,
                                                            default=False):
                        raise IOError(
                            'Diparity file object does not contain enough data sets.')

            # 6) We need to initialize the edge confidence array
            for res in self.epi_res:
                res_name = str(res[0]) + 'x' + str(res[1])
                if res[0] > self.r_start[0] and res[1] > self.r_start[
                    1] and not SCRATCH:  # There must be data in file for at least all resolutions larger than r_start:
                    if not self.Ce_field['edge_conf'].get(res_name,
                                                          default=False):
                        raise IOError(
                            'Edge confidence file object does not contain enough data sets.')

            # 9) We need to initialize the disparity confidence array
            for res in self.epi_res:
                res_name = str(res[0]) + 'x' + str(res[1])
                if res[0] > self.r_start[0] and res[1] > self.r_start[
                    1] and not SCRATCH:  # There must be data in file for at least all resolutions larger than r_start:
                    if not self.Cd_field['disp_conf'].get(res_name,
                                                          default=False):
                        raise IOError(
                            'Disparity confidence file object does not contain enough data sets.')

    def getEpi(self, res, v=None, s=None, u=None):
        """
        Get the required epi data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.

        Returns
        -------
        : numpy.array [v,s,u]
            The sub(data) requested.

        """
        res_name = str(res[0]) + 'x' + str(res[1])
        data = self.epi_field['epis/' + res_name]

        if data is None:
            raise IOError('EPI data not found!')

        if v is None and s is None and u is None:
            return data[...]
        elif v is None and s is None:
            return data[:, :, u]
        elif v is None and u is None:
            return data[:, s]
        elif s is None and u is None:
            return data[v]
        elif v is None:
            return data[:, s, u]
        elif s is None:
            return data[v, :, u]
        elif u is None:
            return data[v, s]

    def getDisp(self, res, v=None, s=None, u=None, level=None):
        """
        Get the required required disparity data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        level : int, optional
            Determine which disparity data to get in DEBUG mode.
            0) after refinement,
            1) raw data,
            2) after bilateral median,
            3) after confidence selection

        Returns
        -------
        : numpy.array [v,s,u, (level)]
            The sub(data) requested.
        """
        grp = self.disp_field.require_group('disparities')
        res_name = str(res[0]) + 'x' + str(res[1])
        if self.DEBUG:
            grp.require_dataset(res_name,
                                (res[0], self.lf_res[0], res[1], 4),
                                dtype=np.float32, fillvalue=np.nan)
        else:
            grp.require_dataset(res_name,
                                (res[0], self.lf_res[0], res[1]),
                                dtype=np.float32, fillvalue=np.nan)
            level = None

        data = self.disp_field['disparities/' + res_name]

        if level is None:

            if v is None and s is None and u is None:
                return data[...]
            elif v is None and s is None:
                return data[:, :, u]
            elif v is None and u is None:
                return data[:, s]
            elif s is None and u is None:
                return data[v]
            elif v is None:
                return data[:, s, u]
            elif s is None:
                return data[v, :, u]
            elif u is None:
                return data[v, s]
        else:
            if v is None and s is None and u is None:
                return data[:, :, :, level]
            elif v is None and s is None:
                return data[:, :, u, level]
            elif v is None and u is None:
                return data[:, s, :, level]
            elif s is None and u is None:
                return data[v, :, :, level]
            elif v is None:
                return data[:, s, u, level]
            elif s is None:
                return data[v, :, u, level]
            elif u is None:
                return data[v, s, :, level]

    def getScore(self, res, v=None, s=None, u=None, level=None):
        """
        Get the required required score data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        level : int, optional
            Determine which score data to get in DEBUG mode.
            0) S_max,
            1) S_mean,
            2) S_argmax,

        Returns
        -------
        : numpy.array [v,s,u, (level)]
            The sub(data) requested.
        """
        grp = self.disp_field.require_group('scores')
        res_name = str(res[0]) + 'x' + str(res[1])

        grp.require_dataset(res_name,
                            (res[0], self.lf_res[0], res[1], 3),
                            dtype=np.float64, fillvalue=np.nan)

        data = self.score_field['scores/' + res_name]

        if level is None:
            if v is None and s is None and u is None:
                return data[...]
            elif v is None and s is None:
                return data[:, :, u]
            elif v is None and u is None:
                return data[:, s]
            elif s is None and u is None:
                return data[v]
            elif v is None:
                return data[:, s, u]
            elif s is None:
                return data[v, :, u]
            elif u is None:
                return data[v, s]
        else:
            if v is None and s is None and u is None:
                return data[:, :, :, level]
            elif v is None and s is None:
                return data[:, :, u, level]
            elif v is None and u is None:
                return data[:, s, :, level]
            elif s is None and u is None:
                return data[v, :, :, level]
            elif v is None:
                return data[:, s, u, level]
            elif s is None:
                return data[v, :, u, level]
            elif u is None:
                return data[v, s, :, level]

    def getDBs(self, res, v=None, s=None, u=None):
        """
        Get the required required disparity bound data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.

        Returns
        -------
        : numpy.array [v,s,u]
            The sub(data) requested.
        """
        res_name = str(res[0]) + 'x' + str(res[1])
        data = self.DB_field['disp_bounds/' + res_name]

        if data is None:
            raise IOError('Disparity data not found!')

        if v is None and s is None and u is None:
            return data[...]
        elif v is None and s is None:
            return data[:, :, u]
        elif v is None and u is None:
            return data[:, s]
        elif s is None and u is None:
            return data[v]
        elif v is None:
            return data[:, s, u]
        elif s is None:
            return data[v, :, u]
        elif u is None:
            return data[v, s]

    def getCe(self, res, v=None, s=None, u=None):
        """
        Get the required required edge confidence data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.

        Returns
        -------
        : numpy.array [v,s,u]
            The sub(data) requested.
        """
        res_name = str(res[0]) + 'x' + str(res[1])
        grp = self.Ce_field['edge_conf']
        threshold = grp.attrs.get('threshold', default=None)
        data = self.Ce_field['edge_conf/' + res_name]

        if data is None:
            raise IOError('Edge confidence data not found!')

        if v is None and s is None and u is None:
            return data[...], threshold
        elif v is None and s is None:
            return data[:, :, u], threshold
        elif v is None and u is None:
            return data[:, s], threshold
        elif s is None and u is None:
            return data[v], threshold
        elif v is None:
            return data[:, s, u], threshold
        elif s is None:
            return data[v, :, u], threshold
        elif u is None:
            return data[v, s], threshold

    def getCd(self, res, v=None, s=None, u=None):
        """
        Get the required required disparity confidence data.

        Parameters
        ----------
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.

        Returns
        -------
        : numpy.array [v,s,u]
            The sub(data) requested.
        """
        res_name = str(res[0]) + 'x' + str(res[1])
        grp = self.Cd_field['disp_conf']
        threshold = grp.attrs.get('threshold', default=None)

        data = self.Cd_field['disp_conf/' + res_name]

        if data is None:
            raise IOError('Disparity confidence data not found!')

        if v is None and s is None and u is None:
            return data[...], threshold
        elif v is None and s is None:
            return data[:, :, u], threshold
        elif v is None and u is None:
            return data[:, s, ...], threshold
        elif s is None and u is None:
            return data[v], threshold
        elif v is None:
            return data[:, s, u], threshold
        elif s is None:
            return data[v, :, u], threshold
        elif u is None:
            return data[v, s], threshold

    def saveDisp(self, data, res, v=None, s=None, u=None, level=None):
        """
        Save the selected disparity data.

        Parameters
        ----------
        data: numpy.array
            The data to save.
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        level : int, optional
            Determine which disparity data to get in DEBUG mode.
            0) after refinement,
            1) raw data,
            2) after bilateral median,
            3) after confidence selection.
        """
        grp = self.disp_field.require_group('disparities')
        res_name = str(res[0]) + 'x' + str(res[1])
        if self.DEBUG:
            d_set = grp.require_dataset(res_name,
                                        (res[0], self.lf_res[0], res[1], 4),
                                        dtype=np.float32, fillvalue=np.nan)
        else:
            level = None
            d_set = grp.require_dataset(res_name,
                                        (res[0], self.lf_res[0], res[1]),
                                        dtype=np.float32, fillvalue=np.nan)

        if level is None:
            if v is None and s is None and u is None:
                d_set[...] = data
            elif v is None and s is None:
                d_set[:, :, u] = data
            elif v is None and u is None:
                d_set[:, s] = data
            elif s is None and u is None:
                d_set[v] = data
            elif v is None:
                d_set[:, s, u] = data
            elif s is None:
                d_set[v, :, u] = data
            elif u is None:
                d_set[v, s] = data
        else:
            if v is None and s is None and u is None:
                d_set[:, :, :, level] = data
            elif v is None and s is None:
                d_set[:, :, u, level] = data
            elif v is None and u is None:
                d_set[:, s, :, level] = data
            elif s is None and u is None:
                d_set[v, :, :, level] = data
            elif v is None:
                d_set[:, s, u, level] = data
            elif s is None:
                d_set[v, :, u, level] = data
            elif u is None:
                d_set[v, s, :, level] = data

    def saveScore(self, data, res, v=None, s=None, u=None, level=0):
        """
        Save the selected score data.

        Parameters
        ----------
        data: numpy.array
            The data to save.
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        level : int, optional
            Determine which disparity data to get in DEBUG mode.
            0) S_max,
            1) S_mean,
            2) S_argmax,
        """
        grp = self.score_field.require_group('scores')
        res_name = str(res[0]) + 'x' + str(res[1])

        d_set = grp.require_dataset(res_name,
                                    (res[0], self.lf_res[0], res[1], 3),
                                    dtype=np.float32, fillvalue=np.nan)

        if v is None and s is None and u is None:
            d_set[:, :, :, level] = data
        elif v is None and s is None:
            d_set[:, :, u, level] = data
        elif v is None and u is None:
            d_set[:, s, :, level] = data
        elif s is None and u is None:
            d_set[v, :, :, level] = data
        elif v is None:
            d_set[:, s, u, level] = data
        elif s is None:
            d_set[v, :, u, level] = data
        elif u is None:
            d_set[v, s, :, level] = data

    def saveDBs(self, data, res, v=None, s=None, u=None):
        """
        Save the selected disparity bound data.

        Parameters
        ----------
        data: numpy.array
            The data to save.
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        """

        grp = self.DB_field.require_group('disp_bounds')
        res_name = str(res[0]) + 'x' + str(res[1])
        d_set = grp.require_dataset(res_name,
                                    (res[0], self.lf_res[0], res[1], 2),
                                    dtype=np.float32, fillvalue=0)

        if v is None and s is None and u is None:
            d_set[...] = data
        elif v is None and s is None:
            d_set[:, :, u] = data
        elif v is None and u is None:
            d_set[:, s] = data
        elif s is None and u is None:
            d_set[v] = data
        elif v is None:
            d_set[:, s, u] = data
        elif s is None:
            d_set[v, :, u] = data
        elif u is None:
            d_set[v, s] = data

    def saveCe(self, data, res, v=None, s=None, u=None, threshold=None):
        """
        Save the selected edge confidence data.

        Parameters
        ----------
        data: numpy.array
            The data to save.
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        threshold : float, optional
            Confidence threshold for binary mask.
        """
        grp = self.Ce_field.require_group('edge_conf')
        if threshold is not None:
            grp.attrs.create('threshold', threshold)
        res_name = str(res[0]) + 'x' + str(res[1])
        d_set = grp.require_dataset(res_name, (res[0], self.lf_res[0], res[1]),
                                    dtype=np.float32, fillvalue=np.nan)

        if v is None and s is None and u is None:
            d_set[...] = data
        elif v is None and s is None:
            d_set[:, :, u] = data
        elif v is None and u is None:
            d_set[:, s] = data
        elif s is None and u is None:
            d_set[v] = data
        elif v is None:
            d_set[:, s, u] = data
        elif s is None:
            d_set[v, :, u] = data
        elif u is None:
            d_set[v, s] = data

    def saveCd(self, data, res, v=None, s=None, u=None, threshold=None):
        """
        Save the selected disparity confidence data.

        Parameters
        ----------
        data: numpy.array
            The data to save.
        res : tuple
             The current v- and u-dimension.
        v : int, optional
            The only v-dimension to return.
        s : int, optional
            The only s-dimension to return.
        u : int, optional
            The only u-dimension to return.
        threshold : float, optional
            Confidence threshold for binary mask.
        """

        grp = self.Cd_field.require_group('disp_conf')
        if threshold is not None:
            grp.attrs.create('threshold', threshold)
        res_name = str(res[0]) + 'x' + str(res[1])
        d_set = grp.require_dataset(res_name, (res[0], self.lf_res[0], res[1]),
                                    dtype=np.float32, fillvalue=np.nan)

        if v is None and s is None and u is None:
            d_set[...] = data
        elif v is None and s is None:
            d_set[:, :, u] = data
        elif v is None and u is None:
            d_set[:, s] = data
        elif s is None and u is None:
            d_set[v] = data
        elif v is None:
            d_set[:, s, u] = data
        elif s is None:
            d_set[v, :, u] = data
        elif u is None:
            d_set[v, s] = data

    def generate_s_hat_order(self):
        """
        Generate a list with each entry being the next s-entry to work with.

        Returns
        -------
         : list
         A ordered list with the s-dimensions to go through.

        """
        s = []
        if self.s_hat is not None:
            s.append(self.s_hat)  # We just use one scanline
        else:
            s.append(self.lf_res[0] // 2)
            for i in range(1, self.lf_res[0] // 2 + 1):
                s.append(self.lf_res[0] // 2 - i)  # The line under s_hat
                s.append(self.lf_res[0] // 2 + i)  # The line above s_hat
            if self.lf_res[
                0] % 2 == 0:  # For even numbers we have added one line too much
                s.pop()

        return s

    def calculateDisp(self, min_disp, max_disp, stepsize, Ce_t, Cd_t, S_t, NOISEFREE=False):
        """
        The main method to calculate the disparity estimates.

        Parameters
        ----------
        min_disp : float
            The minimal disparity to sample for.
        max_disp : float
            The maximal disparity to sample for.
        stepsize : float
            The stepsize used during the sampling procedure.
        Ce_t : float
            The threshold for the edge confidence.
        Cd_t : float:
            The threshold for the disparity confidence.
        S_t : float
            The similarity threshold e.g. for the bilateral median filter.
        NOISEFREE : bool, optional
            True means not to iteratively smooth the mean radiance. This should
            only be enabled for noisy data.
        """
        s_list = self.generate_s_hat_order()  # generate list of s_hat lines to go through
        r_list = [r for r, res in enumerate(self.epi_res)
                  if res[0] <= self.r_start[0] and res[1] <= self.r_start[1]]  # generate the indices of r we need to compute

        for r, res in enumerate(self.epi_res):  # adjust disparity bounds
            if res[0] > self.r_start[0] and res[1] > self.r_start[1]:
                u_scale = self.epi_res[r + 1][1] / self.epi_res[r][1]
                assert u_scale <= 1, 'The scaling factor of u-dimension is larger 1.'
                min_disp *= u_scale
                max_disp *= u_scale
                stepsize *= u_scale

        # Initialize a progress bar to track the calculation process
        max_val = 0
        for r in r_list:
            max_val += self.epi_res[r, 0] * len(s_list)
        widgets = ['Calculating disparities: ', Percentage(), ' ', Bar(), ' ',
                   ETA(), ' ']
        progress = ProgressBar(widgets=widgets, maxval=max_val).start()
        current_val = 0

        ###############################################################
        #                                                             #
        #                        Resolution level                     #
        #                                                             #
        ###############################################################
        for r in r_list:  # go through all epi resolutions; r is only the index of the resolution

            # create some variables and temporal arrays
            s_dim = self.lf_res[0]
            v_dim = self.epi_res[r, 0]
            u_dim = self.epi_res[r, 1]
            res = self.epi_res[r]

            if v_dim == self.lf_res[1] and u_dim == self.lf_res[2]:
                FINEST = True
            else:
                FINEST = False
            if v_dim == self.epi_res[-1, 0] and u_dim == self.epi_res[-1, 1]:
                COARSEST = True
            else:
                COARSEST = False

            # Create some directories
            sample_dir = os.path.join(
                self.output_dir,'Plots/Radiance/{v_dim}x{u_dim}/'.format(
                    v_dim=v_dim, u_dim=u_dim))
            if self.DEBUG and not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            propagation_dir = os.path.join(
                self.output_dir, 'Plots/Propagation/{v_dim}x{u_dim}/'.format(
                    v_dim=v_dim, u_dim=u_dim))
            if self.DEBUG and not os.path.exists(propagation_dir):
                os.makedirs(propagation_dir)

            # Find number of jobs to start
            max_jobs = self.n_cpus
            if v_dim < 300:
                n_jobs = 1
            else:
                n_jobs = max_jobs

            n_tasks = v_dim // n_jobs

            epis = self.getEpi(res)  # The epis; ndarray[v,s,u].
            Ds = self.getDisp(res, level=0)  # The disparities; ndarray[v,s,u].

            ###############################################################
            #                                                             #
            #                        scan line level                      #
            #                                                             #
            ###############################################################
            for s in s_list:  # scan through all lines in an epi; s is only the index of the line

                r_bars = np.zeros((v_dim, u_dim), dtype=np.float32)  # The scanline updated radiances of all epis; ndarray[v,u].
                Md = np.full((v_dim, u_dim), False, dtype=np.bool)  # The scanline disparity confidences; ndarray[v,u].

                # 1. edge confidence (2):
                # The edge mask is needed on resolution level per scanline.
                # Thus we do the calculation ones per resolution and store the
                # mask in memory for efficient access. 
                if not COARSEST:  # Only calculate EPI-pixels with high edge confidence
                    Ce, Me = edge_confidence(epis[:, s], window=9,
                                             threshold=Ce_t)
                else:  # except for the coarsest resolution
                    Ce, Me = edge_confidence(epis[:, s], window=9, threshold=-1)
                    assert np.all(Me),\
                        'Unvalide edge confidence at coarsest resolution.'

                if self.DEBUG:
                    self.saveCe(Ce, res, s=s, threshold=Ce_t)

                Mc = np.isnan(Ds[:, s])
                M = np.logical_and(Me, Mc)

                if n_jobs > 1:  # only then perform multiprocessing
                    pool = Pool(processes=n_jobs)

                    ###############################################################
                    #                                                             #
                    #                       epi level                             #
                    #                                                             #
                    ###############################################################
                    for v in range(0, v_dim, n_tasks):  # go through all epis; v is only the index of the epi

                        n = min(n_jobs * n_tasks, v_dim - v)
                        v_jobs = range(v, v + n)
                        s_hat_jobs = repeat(s, n)
                        Cd_t_jobs = repeat(Cd_t, n)
                        min_disp_jobs = repeat(min_disp, n)
                        max_disp_jobs = repeat(max_disp, n)
                        stepsize_jobs = repeat(stepsize, n)
                        NOISEFREE_jobs = repeat(NOISEFREE, n)
                        COARSEST_jobs = repeat(COARSEST, n)
                        DEBUG_jobs = repeat(self.DEBUG, n)
                        s_hat_DEBUG_jobs = repeat(s_list[0], n)
                        DEBUG_dir_jobs = repeat(sample_dir, n)
                        epi_jobs = iter([epis[v_] for v_ in v_jobs])
                        D_jobs = iter([Ds[v_, s] for v_ in v_jobs])
                        Ce_jobs = iter([Ce[v_] for v_ in v_jobs])
                        M_jobs = iter([M[v_] for v_ in v_jobs])

                        args = zip(epi_jobs, D_jobs, Ce_jobs, M_jobs,
                                   s_hat_jobs, min_disp_jobs, max_disp_jobs,
                                   stepsize_jobs, Cd_t_jobs, NOISEFREE_jobs,
                                   COARSEST_jobs, DEBUG_jobs, s_hat_DEBUG_jobs,
                                   DEBUG_dir_jobs, v_jobs)

                        results_jobs = pool.imap(convert_process_epi, args,
                                                 chunksize=n_tasks)

                        for i, results in enumerate(results_jobs):
                            v_ = v_jobs[i]
                            Ds[v_, s] = results[0]
                            r_bars[v_] = results[1]
                            Md[v_] = results[2]

                            if self.DEBUG:
                                self.saveDisp(Ds[v_, s], res, v=v_, s=s,
                                              level=1)
                                self.saveCd(results[3], res, v=v_,
                                            threshold=Cd_t)
                                self.saveDBs(results[4], res, v=v_, s=s)
                                self.saveScore(results[5], res, v=v_, s=s,
                                               level=0)
                                self.saveScore(results[6], res, v=v_, s=s,
                                               level=1)
                                self.saveScore(results[7], res, v=v_, s=s,
                                               level=2)
                    pool.close()
                    pool.join()

                else:  # no multiprocessing
                    ###############################################################
                    #                                                             #
                    #                       epi level                             #
                    #                                                             #
                    ###############################################################
                    for v in range(v_dim):  # go through all epis; v is only the index of the epi

                        # 2. Disparity bounds
                        # 3. Radiance sampling(3)
                        # 4. Score computation (4, 5)
                        # 5. Disparity confidence (7)
                        # 6. Disparity estiamte (6)

                        Ds[v, s], r_bars[v], Md[
                            v], Cd, DB, S_max, S_mean, S_argmax = process_epi(
                            epis[v], Ds[v, s], Ce[v], M[v], s, min_disp,
                            max_disp, stepsize, Cd_t, NOISEFREE, COARSEST,
                            DEBUG=self.DEBUG, s_hat_DEBUG=s_list[0],
                            DEBUG_dir=sample_dir, v_DEBUG=v)

                        if self.DEBUG:
                            self.saveDisp(Ds[v, s], res, v=v, s=s, level=1)
                            self.saveCd(Cd, res, v=v, threshold=Cd_t)
                            self.saveDBs(DB, res, v=v, s=s)
                            self.saveScore(S_max, res, v=v, s=s, level=0)
                            self.saveScore(S_mean, res, v=v, s=s, level=1)
                            self.saveScore(S_argmax, res, v=v, s=s, level=2)

                ###############################################################
                #                                                             #
                #                        scan line level                      #
                #                                                             #
                ###############################################################

                # 7. bilateral median filter

                if not COARSEST:  # using the threshold
                    Ds[:, s] = bilateral_median(Ds[:, s], epis[:, s], M, Me,
                                                threshold=S_t)
                else:  # except for the lowest resolution, which is necessary not to introduce new NaNs
                    Ds[:, s] = bilateral_median(Ds[:, s], epis[:, s], M, Me,
                                                threshold=float('inf'))
                if self.DEBUG:
                    self.saveDisp(Ds[:, s], res, s=s, level=2)

                M = np.logical_and(~Mc, ~Md)
                if COARSEST:
                    assert not np.any(M)
                Ds[:, s][M] = np.nan  # remove unvalide values
                if self.DEBUG:
                    self.saveDisp(Ds[:, s], res, s=s, level=3)

                # 8. Propagate from scanline
                if self.s_hat is None:
                    Ds, epi_plot = propagation(Ds, epis, r_bars, s,
                                               threshold=S_t, DEBUG=False)
                    if self.DEBUG:
                        if s == s_list[0]:
                            imsave(os.path.join(
                                propagation_dir,'Propagation_v={v}_s={s}.png'.format(
                                    v=v_dim // 2, s=s)),epi_plot[:, s, :])

                if COARSEST:
                    assert not np.any(np.isnan(Ds[:, s]))
                self.saveDisp(Ds, res, level=0)

                current_val += v_dim
                progress.update(current_val)

            ###############################################################
            #                                                             #
            #                        Resolution level                     #
            #                                                             #
            ###############################################################


            # One resolution is done, so we update which values need recomputation and save output

            if not COARSEST:
                res_next = self.epi_res[r + 1]
                u_scale = res_next[1] / res[1]
                assert u_scale <= 1, 'The scaling factor of u-dimension is larger 1.'
                min_disp *= u_scale
                max_disp *= u_scale
                stepsize *= u_scale

                Ds = fine_to_course(Ds, res_next)
                self.saveDisp(Ds, res_next, level=0)

        progress.finish()

        # at last we need to sample  up
        for r in range(len(self.epi_res) - 1, 0, -1):
            Ds = course_to_fine(self.getDisp(self.epi_res[r - 1], level=0),
                                self.getDisp(self.epi_res[r], level=0))
            self.saveDisp(Ds, self.epi_res[r - 1], level=0)

    def calculateMap(self):
        """
        Calculate the final disparity map for each s-dimension and save output
        into a hdf5 file container. In DEBUG mode lot's of furhter plot's are
        generated.
        """

        disp_dir = os.path.join(self.output_dir,
                                'Plots/Disparity/{v_dim}x{u_dim}/'.format(
                                    v_dim=self.lf_res[1], u_dim=self.lf_res[2]))
        if not os.path.exists(disp_dir):
            os.makedirs(disp_dir)

        Disp_map_f = h5py.File(
            os.path.join(self.output_dir, 'disparity_map.hdf5'), 'w')
        Disp_map = Disp_map_f.require_dataset('disparity_map',
                                              shape=self.lf_res[0:3],
                                              dtype=np.float64)

        s_list = self.generate_s_hat_order()
        # Disp_map[:] = self.getDisp(self.epi_res[0], s=s level=0).swapaxes(0,1)#self.lf_res[0:3])

        for s in s_list:
            Disp_map[s] = median_filter(
                self.getDisp(self.epi_res[0], s=s, level=0), size=(
                3, 3))  # we apply a median filter to remove remaining speccles
            plt.plot()
            plt.imshow(Disp_map[s], cmap='gray', interpolation='None')
            plt.colorbar()
            plt.savefig(
                os.path.join(disp_dir, str(s) + '_final_disparityMap.png'))
            plt.close()

        if self.DEBUG:

            for r, res in enumerate(self.epi_res):
                s_dim = self.lf_res[0]
                v_dim = res[0]
                u_dim = res[1]

                lf_dir = os.path.join(self.output_dir,
                                      'Plots/lightfield/{v_dim}x{u_dim}/'.format(
                                          v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(lf_dir):
                    os.makedirs(lf_dir)

                disp_dir = os.path.join(self.output_dir,
                                        'Plots/Disparity/{v_dim}x{u_dim}/'.format(
                                            v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(disp_dir):
                    os.makedirs(disp_dir)

                score_dir = os.path.join(self.output_dir,
                                         'Plots/Scores/{v_dim}x{u_dim}/'.format(
                                             v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(score_dir):
                    os.makedirs(score_dir)

                bounds_dir = os.path.join(self.output_dir,
                                          'Plots/DisparityBounds/{v_dim}x{u_dim}/'.format(
                                              v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(bounds_dir):
                    os.makedirs(bounds_dir)

                dispConf_dir = os.path.join(self.output_dir,
                                            'Plots/DisparityConfidence/{v_dim}x{u_dim}/'.format(
                                                v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(dispConf_dir):
                    os.makedirs(dispConf_dir)

                edge_dir = os.path.join(self.output_dir,
                                        'Plots/Edges/{v_dim}x{u_dim}/'.format(
                                            v_dim=v_dim, u_dim=u_dim))
                if not os.path.exists(edge_dir):
                    os.makedirs(edge_dir)

                s = self.s_hat if self.s_hat is not None else s_list[
                    len(s_list) // 2]

                Lf_map = self.getEpi(self.epi_res[r], s=s)
                Disp_map = self.getDisp(self.epi_res[r], s=s)
                Score_map = self.getScore(self.epi_res[r], s=s)
                DB_map = self.getDBs(self.epi_res[r], s=s)
                Cd_map, Cd_t = self.getCd(self.epi_res[r], s=s)
                Md_map = Cd_map > Cd_t
                Ce_map, Ce_t = self.getCe(self.epi_res[r], s=s)
                Me_map = Ce_map > Ce_t

                plt.plot()
                plt.imsave(os.path.join(lf_dir, str(s) + '_ligthfield.png'),
                           gray2rgb(Lf_map[...]))
                plt.close()

                plt.plot()
                plt.imshow(Disp_map[:, :, 0], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(disp_dir, str(
                    s) + '_upsampled_disparityMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Disp_map[:, :, 1], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(
                    os.path.join(disp_dir, str(s) + '_raw_disparityMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Disp_map[:, :, 2], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(
                    os.path.join(disp_dir, str(s) + '_median_disparityMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Disp_map[:, :, 3], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(disp_dir, str(
                    s) + '_confidendt_disparityMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Score_map[:, :, 0], cmap='gray',
                           interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(score_dir, str(s) + '_S_max.png'))
                plt.close()

                plt.plot()
                plt.imshow(Score_map[:, :, 1], cmap='gray',
                           interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(score_dir, str(s) + '_S_mean.png'))
                plt.close()

                plt.plot()
                plt.imshow(Score_map[:, :, 2], cmap='gray',
                           interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(score_dir, str(s) + '_S_argmax.png'))
                plt.close()

                plt.plot()
                plt.imshow(DB_map[:, :, 0], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(bounds_dir, str(
                    s) + '_lowerDisparityBoundsMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(DB_map[:, :, 1], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(bounds_dir, str(
                    s) + '_upperDisparityBoundsMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Cd_map[...], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(dispConf_dir, str(
                    s) + '_disparityConfidenceMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Md_map[...], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(os.path.join(dispConf_dir, str(
                    s) + '_disparityConfidenceMask.png'))
                plt.close()

                plt.plot()
                plt.imshow(Ce_map[...], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(
                    os.path.join(edge_dir, str(s) + '_egeConfidenceMap.png'))
                plt.close()

                plt.plot()
                plt.imshow(Me_map[...], cmap='gray', interpolation='none')
                plt.colorbar()
                plt.savefig(
                    os.path.join(edge_dir, str(s) + '_egeConfidenceMask.png'))
                plt.close()
