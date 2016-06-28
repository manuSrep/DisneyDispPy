#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Disney Disparity.
:author: Manuel Tuschen
:date: 24.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
from numba import jit


def score_computation(R, epi, s_hat, M, h=0.02, NOISEFREE=False):
    """
    Calculates a score for each disparity estimate as in eq. 4:
    Afterwards for each EPI pixel u in the currently processed s-dimension each
    disparity is scored.

    Parameters
    ----------
    R : numpy.array [u,d,s]
       The set of sampled radiances. For each pixel u and disparity d the gray
       value for each scanline s is stored.
    epi : numpy.array [s,u]
       One gray-value epi.
    s_hat : int
       The current scanline s to sample for.
    M : numpy.array [u] boolean.
       Mask which values to consider.
    h : float
        A bandwidth parameter for the kernel function.
    NOISEFREE: boolean, optional
        Improve performance of noise-free EPIs.

    Returns
    ----------
    S_norm : numpy.array [u,d]
        The calculated scores for each sampled radiance at each pixel u.
    R_bar : numpy.array [u,d].
        The updated convergence radiance for each EPI pixel.
    """

    u_dim = R.shape[0]
    d_dim = R.shape[1]
    s_dim = R.shape[2]
    # Check dimensions of input data
    assert epi.shape == (s_dim, u_dim,), 'Input EPI has wrong shape in function \'score_computation\'.'
    assert M.shape == (u_dim,), 'Input Mc has wrong shape in function \'score_computation\'.'
    assert s_hat >= 0 and s_hat < s_dim, 's_hat not in range 0 - s-dim in function \'sample\'.'

    # alculate initial R_bar. There must be one value for each disparity later.
    R_bar = np.tile(epi[s_hat].reshape((u_dim, 1, 1)), (1, d_dim, s_dim))
    assert R_bar.shape == (u_dim, d_dim, s_dim), 'Initial R_bar has wrong shape in fucntion \'score_computation\'.'
    assert not np.any(np.isnan(R_bar)), 'NaN detected in initial R_bar in function \'score_computation\'.'

    if not NOISEFREE:
        R_bar = calc_r_bar_iter(R_bar, R, M, h)
        assert R_bar.shape == (u_dim, d_dim, s_dim), 'Iterative R_bar has wrong shape in function \'score_computation\'.'
        assert not np.any(np.isnan(R_bar[:, :, s_hat][M])), 'NaN detected in final R_bar in function \'score_computation\'.'

    # calculate R - R_bar
    R_minus_R_bar = R - R_bar
    assert R_minus_R_bar.shape == (u_dim, d_dim,s_dim,), 'R - R_bar difference has wrong shape in function \'score_computation\'.'


    # compute the norm for the scoring function
    R_norm = np.sum(~np.isnan(R), axis=-1)  # count along s-axis ignoring NaNs
    all_nan = R_norm == 0
    R_norm += 1
    #R_norm[~all_nan] = 1.0 / R_norm[~all_nan] # Not working why ??
    R_norm = 1.0 / R_norm
    assert R_norm.shape == (u_dim, d_dim), 'R_norm output has wrong shape in fucntion \'score_computation\'.'

    # calculate the score
    S = np.sum(epanechnikov_kernel(R_minus_R_bar, M, h), axis=-1)
    S_norm = R_norm * S
    R_bar = R_bar[:, :, s_hat]

    # Let's see if our final results have reasonable meaning'
    assert S_norm.shape == (u_dim, d_dim), 'S_norm output has wrong shape in fucntion \'score_computation\'.'
    assert not np.any(S_norm[all_nan] != 0), 'Score not zero if all entries are NaN in fucntion \'score_computation\'.'
    assert not np.any(np.isnan(S_norm)), 'NaN detected in S_norm output in fucntion \'score_computation\'.'
    assert R_bar.shape == (u_dim, d_dim), 'R_bar output has wrong shape in fucntion \'score_computation\'.'
    return S_norm, R_bar


@jit(nopython=True)
def epanechnikov_kernel(R, M, h):
    """
    The epanechnikov kernel function as in eq. 4

    Parameters
    ----------
    R : numpy.array [u,d,s]
       The set of sampled radiances. For each pixel u and disparity d the gray
       value for each scanline s is stored.
    M : numpy.array [u] boolean.
       Mask which values to consider.
    h : float
        A bandwidth parameter for the kernel function.

    Returns
    ----------
    kernel_val: numpy.array  [u,d,s].
        Float values between 0 and 1 which 1 being the highest score possible.
    """

    u_dim = R.shape[0]
    d_dim = R.shape[1]
    s_dim = R.shape[2]

    kernel_vals = np.zeros(R.shape, dtype=np.float64)

    for u in range(u_dim):
        if not M[u]:
            continue
        for d in range(d_dim):
            for s in range(s_dim):
                if not np.isnan(R[u, d, s]) and abs(R[u, d, s] / h) <= 1:  # can still be zero due to sampling outside of epi
                    kernel_vals[u, d, s] = 1 - (R[u, d, s] / h) ** 2

    return kernel_vals


@jit(nopython=True)
def calc_r_bar_iter(R_bar, R, M, h, n_iter=10):
    """
    Calculate the radiance mean according to eq. 5
    For each EPI pixel in the current s-dimension a new convergence radiance is
    sampled.

    Parameters
    ----------
    R_bar : numpy.array [u,d]
        The updated convergence radiance for each EPI pixel.
    R : numpy.array [u,d,s]
       The set of sampled radiances. For each pixel u and disparity d the gray
       value for each scanline s is stored.
    M : numpy.array [u] boolean.
        Mask which values to consider.
    h : float
        A bandwidth parameter for the kernel function.
    n_iter : int, optional
        The number of iterations for the convergence calculation.

    Returns
    ----------
    R_bar : numpy.array [u,d]
        The updated convergence radiance for each EPI pixel.
    """

    u_dim = R.shape[0]
    d_dim = R.shape[1]
    s_dim = R.shape[2]

    n = 0
    while n < n_iter:
        R_minus_R_bar = R - R_bar
        assert R_minus_R_bar.shape == (u_dim, d_dim, s_dim), 'R - R_bar difference has wrong shape in function \'calc_r_bar_iter_loop\'.'

        kernel_vals = epanechnikov_kernel(R_minus_R_bar, M, h)
        assert kernel_vals.shape == (u_dim, d_dim, s_dim), 'kernel_vals has wrong shape in function \'calc_r_bar_iter\'.'
        assert not np.any(np.isnan(kernel_vals)), 'NaN detected in kernel output in function \'calc_r_bar_iter\'.'

        # sum_numerator = np.nansum(kernel_vals * R, axis=-1) # not supported yet
        sum_numerator = sum_along_s_axis(kernel_vals * R, M)  # resulting array will have NaNs as R !
        assert sum_numerator.shape == (u_dim, d_dim), 'sum_numerator has wrong shape in function \'calc_r_bar_iter\'.'

        # sum_denominator = np.nansum(kernel_vals, axis=-1) # not supported yet
        sum_denominator = sum_along_s_axis(kernel_vals, M)
        sum_denominator += 1  # add pseudocount to avoid division by zero
        assert sum_denominator.shape == (u_dim, d_dim), 'sum_denominator has wrong shape in function \'calc_r_bar_iter\'.'
        assert not np.any(np.isnan(sum_denominator[M])), 'NaN detected in sum_denominator output in function \'calc_r_bar_iter\'.'

        _R_bar = sum_numerator / sum_denominator
        assert _R_bar.shape == (u_dim, d_dim), 'R_bar has wrong shape in function \'calc_r_bar_iter\'.'

        #R_bar = np.tile(_R_bar[:,:,s_hat], (1, 1, s_dim)) # not supported yet
        R_bar = tile_R_bar(_R_bar, s_dim)
        assert R_bar.shape == (u_dim, d_dim,s_dim), 'R_bar output has wrong shape in function \'calc_r_bar_iter\'.'

        n += 1

    return R_bar




    ########################################################################
    #                                                                      #
    #                           Helper functions                           #
    #                                                                      #
    ########################################################################


# These functions are necessary as the standard way of doing it is not supported by numba right now


@jit(nopython=True)
def tile_R_bar(R_bar, s_dim):
    """
    Extend dimension of R_bar by an s-axis.

    Parameters
    ----------
    R_bar : numpy.array [u,d]
        The updated convergence radiance for each EPI pixel.
    s_dim : int
        The dimension of the s-axis.

    Returns
    ----------
    R_bar_tiled : numpy.array [u,d, s]
        The extended R_bar array.
    """

    u_dim = R_bar.shape[0]
    d_dim = R_bar.shape[1]

    R_bar_tiled = np.zeros((u_dim, d_dim, s_dim), dtype=np.float64)

    for u in range(u_dim):
        for d in range(d_dim):
            for s in range(s_dim):
                R_bar_tiled[u, d, s] = R_bar[u, d]
    return R_bar_tiled


@jit(nopython=True)
def sum_along_s_axis(kernel_output, M):
    """
    Sum over the s dimension of the kernel values output.

    Parameters
    ----------
    kernel_output : numpy.array [u,d,s]
        Float values between 0 and 1 which 1 being the highest score possible.
    M : numpy.array [u] boolean
       Mask which values to consider.

    Returns
    ----------
    kernel_sum : numpy.array [u,d]
        The sum of the kernel values over s dimension.
    """

    u_dim = kernel_output.shape[0]
    d_dim = kernel_output.shape[1]
    s_dim = kernel_output.shape[2]

    kernel_sum = np.full((u_dim, d_dim), np.nan, dtype=np.float64)
    for u in range(u_dim):
        if not M[u]:
            continue
        for d in range(d_dim):
            _sum = 0
            for s in range(s_dim):
                if not np.isnan(kernel_output[u, d, s]):
                    _sum += kernel_output[u, d, s]
            kernel_sum[u, d] = _sum
    return kernel_sum
