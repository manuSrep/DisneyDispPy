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

from skimage.io import imsave

from ._confidence import disparity_confidence
from ._radiance import sample_radiance, find_disp_bounds
from ._score import score_computation
from ._disparity import estimate_disp


def process_epi(epi, D, Ce, M, s_hat, min_disp, max_disp, stepsize, Cd_t,
                NOISEFREE, COARSEST, DEBUG=False, s_hat_DEBUG=0,
                DEBUG_dir='/tmp', v_DEBUG=0):
    """
    Depth estimates according to eq. 6 for one scanline  s.
    Function covering all steps which can be performed on one EPI.


    Parameters
    ----------
    epi : numpy.array [s,u]
        One gray-value epi.
    D: numpy.array [u].
        Already calculated disparities.
    Ce :  numpy.array [u]
        Edge confidence values for each EPI pixel.
    M : numpy.array [u] boolean.
       Mask which values to consider.
    s_hat : int
        The current scanline s to sample for.
    min_disp : int
        The minimal disparity to sample for.
    max_disp : int
        The maximal disparity to sample for.
    stepsize : float
        The stepsize used during the sampling procedure.
    Cd_t : float, optional
        The threshold which must be overcome for the depth confidence.
    NOISEFREE : boolean, optional
        Improve performance of noise-free EPIs.
    COARSEST : boolean, optional
        True means we are at the coarsest resolution.
    DEBUG : boolean, optional
        Enable plotting to visualize the sampling process
    s_hat_DEBUG : int, optional
        s_hat used for DEBUG output.
    DEBUG_dir : string, optional
        Directory used for DEBUG output.
    v_DEBUG : int, optional
        v used for DEBUG output.

    DB : numpy.array [u,2]
        The minimal and maximal disparity bounds to sample in between.
    M : numpy.array [u] boolean.
        Mask which values to consider.
    plots : numpy.array [d,s,u].
        If plotting was enabled epis to mark on the sampling process.
    DEBUG : boolean, optional
        Enable plotting to visualize the sampling process


    S : numpy.array [u,d]
        The calculated scores for each sampled radiance at each pixel u.

    R_bar : numpy.array [u,d].
        The updated convergence radiance for each EPI pixel.
    disp_range : numpy.array [d]
        The range of disparities used during the sampling.

    Returns
    -------
    D : numpy.array [u]
        The updated estimated disparities for each u at the current s-dimension.
    R_bar : numpy.array [u]
        The best updated convergence radiance for each EPI pixel.
    Md : numpy.array [u] of boolean.
        True means edge confidence is sufficient.
    Cd :  numpy.array [u]
        The calculated confidence values for each pixel.
    DB : numpy.array [u,2]
        The minimal and maximal disparity bounds to sample in between.
    S_max : numpy.array [u]
        The maximal score over all disparities per pixel.
    S_mean : numpy.array [u]
        The mean score over all disparities per pixel.
    S_argmax : numpy.array [u]
        The index of the maximal score.
    """

    # 2. Disparity bounds
    DB = find_disp_bounds(D, min_disp, max_disp,
                          stepsize)  # calcualte the current disparity bounds; ndarray[u]
    # 3. Radiance sampling(3)
    R, Mr, disp_range, plot = sample_radiance(epi, s_hat, min_disp, max_disp,
                                          stepsize, DB, M, DEBUG=DEBUG)
    # The sampled radiances; ndarray[u,d,s]. The range of disparities; ndarray[d]. The plots; ndarray[d,s,u]
    if DEBUG:  # We plot the sampling results but only in an exemplary fashion
        if s_hat == s_hat_DEBUG:
            imsave(os.path.join(DEBUG_dir,
                                'Sampling_v={v}_s={s}_disparity={d}.png'.format(
                                    v=v_DEBUG, s=s_hat_DEBUG, d=disp_range[0])),
                   plot[0])
            imsave(os.path.join(DEBUG_dir,
                                'Sampling_v={v}_s={s}_disparity={d}.png'.format(
                                    v=v_DEBUG, s=s_hat_DEBUG,
                                    d=disp_range[-1])), plot[-1])

    # 4. Score computation (4, 5)
    S, R_bar = score_computation(R, epi, s_hat, M, Mr, h=0.02, NOISEFREE=NOISEFREE)
    # The scores; ndarray[d,u]. The R_bar values; ndarray[d,u]


    # 5. Disparity confidence (7)
    if not COARSEST:  # only accept values with high depth confidence
        Cd, Md, S_max, S_mean = disparity_confidence(S, Ce, threshold=Cd_t)
    else:  # except for the lowest resolution
        Cd, Md , S_max, S_mean = disparity_confidence(S, Ce, threshold=-1)
        assert np.all(Md), 'Unvalide disparity confidence at coarsest resolution.'
    # The disparity confidence score; ndarray[u]. disparity confidence mask; ndarray[u].


    # 6. Disparity estiamte (6)
    D, R_bar, S_argmax = estimate_disp(D, S, M, R_bar, disp_range)
    # The best scored disparity estimate; ndarray[u]. The best scored R_bar; ndarray[u].

    return D, R_bar, Md, Cd, DB,  S_max, S_mean, S_argmax


def convert_process_epi(args):
    """
    Function to allow multiprocessing with multiple arguments.
    """

    return process_epi(*args)
