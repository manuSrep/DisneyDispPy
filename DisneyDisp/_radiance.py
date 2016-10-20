#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

from math import floor, ceil

import numpy as np
from numba import jit


@jit(nopython=True)
def find_disp_bounds(Disps, min_disp, max_disp, stepsize):
    """
    Calculate the disparity bounds for the next iteration

    Parameters
    ----------
    Disps : numpy.array [u]
        Reliable disparity estimates. All other values must be NaN.
    min_disp : int
        The minimal disparity to sample for.
    max_disp : int
        The maximal disparity to sample for.
    stepsize : float
        The stepsize used during the sampling procedure.

    Returns
    -------
    DB : numpy.array [u,2]
        The estimated disparity bounds for each u at line s in each EPI.
    """

    u_dim = Disps.shape[0]

    DB = np.zeros((u_dim, 2), dtype=np.float32)
    DB[:, 0] = min_disp
    DB[:, 1] = max_disp

    if np.all(np.isnan(Disps)):
        # if there are no disparity estiamtes yet
        return DB
    else:

        # counter = np.sum(~np.isnan(Disps)) # Does not work in numba yet
        counter = 0  # count how many valid estimates there are already
        for u in range(u_dim):
            if not np.isnan(Disps[u]):
                counter += 1
        ind = np.zeros((counter), dtype=np.uint)
        i = 0
        for u in range(u_dim):  # we seek for non nan indeces
            if not np.isnan(Disps[u]):
                ind[i] = u
                i += 1

        if len(ind) > 0:
            for i in range(len(ind)):
                DB[ind[i], 0] = Disps[
                    ind[i]]  # reliable estimates keep their bound
                DB[ind[i], 1] = Disps[ind[i]]
                assert DB[ind[i], 0] == DB[ind[i], 1], 'Depth bounds for reliable estimates are nor equal in function \'disp_bounds\'.'

        if len(ind) > 1:
            i = 0
            first = ind[0]
            while i < len(
                    ind) - 1:  # we assign boarders only if pixels are in between two valid estimates
                i += 1
                second = ind[i]

                min_DB = min(Disps[first], Disps[second]) - stepsize
                max_DB = max(Disps[first], Disps[second]) + stepsize
                for j in range(int(first + 1), int(second)):
                    DB[j, 0] = min_DB
                    DB[j, 1] = max_DB
                    assert DB[j, 0] < DB[j, 1], 'First depth bound not smaller than second in function \'disp_bounds\'.'
                first = second

        return DB


def sample_radiance(epi, s_hat, min_disp, max_disp, stepsize, DB, M, DEBUG=False):
    """
    Sample radiances according to eq. 3:
    R(u,d) = { E( u+(s_hat - s)d ,s ) | s = 1, ..., n }

    This function works on one scanline s per EPI. For each EPI-pixel u in that
    line, radiances are sampled for all disparities d. If disparity bounds are
    given, only disparities in between the bounds or equal are taken into
    account.

    Parameters
    ----------
    epi : numpy.array [s,u]
        One gray-value epi.
    s_hat : int
        The current scanline s to sample for.
    min_disp : int
        The minimal disparity to sample for.
    max_disp : int
        The maximal disparity to sample for.
    stepsize : float
        The stepsize used during the sampling procedure.
    DB : numpy.array [u,2]
        The minimal and maximal disparity bounds to sample in between.
    M : numpy.array [u] boolean.
        Mask which values to consider.
    DEBUG : boolean, optional
        Enable plotting to visualize the sampling process

    Returns
    -------
    R : numpy.array [u,d,s]
        The set of sampled radiances. For each pixel u and disparity d the gray
        value for each scanline s is stored.
    Mr : numpy.array [u,d,s]
        Mask to indicate if value was sampled.
    disp_range : numpy.array [d].
        The range of disparities used during the sampling.
    plots : ndarray [d,s,u] or None.
        If plotting was enabled epis with markings of the sampling process.
    """

    # Calculate all radiances to sample for
    n_disp = int((max_disp - min_disp) / stepsize) + 1
    disp_range = np.linspace(min_disp, max_disp, n_disp)

    u_dim = epi.shape[1]
    d_dim = n_disp
    s_dim = epi.shape[0]
    # Check dimensions of input data
    assert epi.shape == (s_dim, u_dim,), 'Input EPI has wrong shape in function \'sample_radiance\'.'
    assert DB.shape == (u_dim, 2), 'Input DB has wrong shape in function \'sample_radiance\'.'
    assert M.shape == (u_dim,), 'Input M has wrong shape in function \'sample_radiance\'.'
    assert d_dim > 0, 'There are no disparities to samle for in function \'sample_radiance\'.'
    assert min_disp < max_disp, 'minmimal disparity is not smaller than maximal disparity in function \'sample_radiance\'.'
    assert s_hat >= 0 and s_hat < s_dim, 's_hat {s_hat} not in range 0 - s-dim in function \'sample_radiance\'.'

    if DEBUG:
        plots = np.tile(epi.reshape((1, s_dim, u_dim,)), (n_disp, 1, 1,))
    else:
        plots = np.zeros((n_disp, s_dim, u_dim,), dtype=np.uint8)

    R, Mr, plots = sample_radiance_inner(epi, s_hat, disp_range, DB, M, plots, DEBUG=DEBUG)

    # Let's see if our results have reasonable meaning'
    assert R.shape == (u_dim, d_dim, s_dim,), 'Output R has wrong shape in function \'sample_radiance\'.'
    assert Mr.shape == (u_dim, d_dim, s_dim,), 'Output Mr has wrong shape in function \'sample_radiance\'.'
    assert plots.shape == (d_dim, s_dim, u_dim,), 'Output R has wrong shape in function \'sample_radiance\'.'

    return R, Mr, disp_range, plots


@jit(nopython=True)
def sample_radiance_inner(epi, s_hat, disp_range, DB, M, plots, DEBUG=False):
    """
    A fast inner function for sampling the radinaces according to eq. 3:

    Parameters
    ----------
    epi : numpy.array [s,u]
        One gray-value epi.
    s_hat : int
        The current scanline s to sample for.
    min_disp : int
        The minimal disparity to sample for.
    max_disp : int
        The maximal disparity to sample for.
    stepsize : float
        The stepsize used during the sampling procedure.
    DB : numpy.array [u,2]
        The minimal and maximal disparity bounds to sample in between.
    M : numpy.array [u] boolean.
        Mask which values to consider.
    plots : numpy.array [d,s,u].
        If plotting was enabled epis to mark on the sampling process.
    DEBUG : boolean, optional
        Enable plotting to visualize the sampling process

    Returns
    -------
    R : numpy.array [u,d,s]
        The set of sampled radiances. For each pixel u and disparity d the gray
        value for each scanline s is stored.
    Mr : numpy.array [u,d,s]
        Mask to indicate if value was sampled.
    plots : ndarray [d,s,u] or None.
        If plotting was enabled epis with markings of the sampling process.
    """

    u_dim = epi.shape[1]
    d_dim = len(disp_range)
    s_dim = epi.shape[0]

    R = np.zeros((u_dim, d_dim, s_dim,), dtype=np.uint8)
    Mr = np.full((u_dim, d_dim, s_dim,), fill_value=False, dtype=np.bool_)

    for u in range(u_dim):
        if not M[u]:
            continue

        for d in range(d_dim):
            if disp_range[d] < DB[u, 0]:
                continue
            if disp_range[d] > DB[u, 1]:
                break # for some disparities there will be NaNs

            for s in range(s_dim):
                x = u + (s_hat - s) * disp_range[d]
                x0 = int(floor(x))
                x1 = int(ceil(x))

                if x0 >= 0 and x1 < u_dim:

                    if x0 == x1:  # if we match an epi pixel
                        a = 1
                        b = 0
                    else:  # otherwise interpolate
                        a = (x1 - x) / (x1 - x0)
                        b = (x - x0) / (x1 - x0)
                    R[u, d, s] = ((a * epi[s, x0]) + (b * epi[s, x1]))
                    Mr[u, d, s] = True

                if DEBUG:
                    plots[d, s, x0] = 255
                    plots[d, s, x1] = 255

    return R, Mr, plots
