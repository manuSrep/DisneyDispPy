#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
from numba import jit


def score_computation(R, epi, s_hat, M, Mr, h=0.02, NOISEFREE=False):
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
    Mr : numpy.array [u,d,s]
        Mask to indicate if value was sampled.
    h : float
        A bandwidth parameter for the kernel function.
    NOISEFREE: boolean, optional
        Improve performance of noise-free EPIs.

    Returns
    -------
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
    assert M.shape == (u_dim,), 'Input M has wrong shape in function \'score_computation\'.'
    assert Mr.shape == (u_dim,d_dim,s_dim), 'Input Mr has wrong shape in function \'score_computation\'.'
    assert s_hat >= 0 and s_hat < s_dim, 's_hat not in range 0 - s-dim in function \'sample\'.'

    # calculate initial R_bar. There must be one value for each disparity later.
    R_bar = np.tile(epi[s_hat].reshape((u_dim, 1)), (1, d_dim))
    assert R_bar.shape == (u_dim,d_dim), 'Initial R_bar has wrong shape in fucntion \'score_computation\'.'

    if not NOISEFREE:
        R_bar = r_bar_iter(R_bar, R, M, Mr, h)
        assert R_bar.shape == (u_dim,d_dim), 'Iterative R_bar has wrong shape in function \'score_computation\'.'
    # compute the norm for the scoring function
    R_norm = np.sum(Mr, axis=-1) # count number of valid entries along s-axis
    R_norm += 1 # add pseudocount to avoid division by zero
    R_norm = 1.0 / R_norm
    assert R_norm.shape == (u_dim, d_dim), 'R_norm output has wrong shape in fucntion \'score_computation\'.'

    # calculate the score
    S = epanechnikov_kernel(R, R_bar, M, Mr, h)
    S_norm = R_norm * S

    # Let's see if our final results have reasonable meaning'
    assert S_norm.shape == (u_dim, d_dim), 'S_norm output has wrong shape in fucntion \'score_computation\'.'
    assert R_bar.shape == (u_dim,d_dim), 'R_bar output has wrong shape in fucntion \'score_computation\'.'
    return S_norm, R_bar


@jit(nopython=True)
def epanechnikov_kernel(R, R_bar, M, Mr, h):
    """
    The epanechnikov kernel function as in eq. 4

    Parameters
    ----------
    R : numpy.array [u,d,s]
       The set of sampled radiances. For each pixel u and disparity d the gray
       value for each scanline s is stored.
    R_bar : numpy.array [u,d].
        The updated convergence radiance for each EPI pixel.
    M : numpy.array [u] boolean.
       Mask which values to consider.
    Mr : numpy.array [u,d,s]
        Mask to indicate if value was sampled.
    h : float
        A bandwidth parameter for the kernel function.

    Returns
    -------
    K: numpy.array  [u,d,s].
        Float values between 0 and 1 which 1 being the highest score possible.
    """

    u_dim = R.shape[0]
    d_dim = R.shape[1]
    s_dim = R.shape[2]

    K = np.zeros((u_dim,d_dim), dtype=np.float32)

    for u in range(u_dim):
        if not M[u]:
            continue
        for d in range(d_dim):
            for s in range(s_dim):
                if Mr[u, d, s]:
                    R_Rbar = R[u, d, s] - R_bar[u, d]
                    if abs(R_Rbar / h) <= 1:
                        K[u, d] += 1 - (R_Rbar / h)**2
    return K


@jit(nopython=True)
def r_bar_iter(R_bar, R, M, Mr, h, n_iter=10):
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
    Mr : numpy.array [u,d,s]
        Mask to indicate if value was sampled.
    h : float
        A bandwidth parameter for the kernel function.
    n_iter : int, optional
        The number of iterations for the convergence calculation.

    Returns
    -------
    R_bar : numpy.array [u,d]
        The updated convergence radiance for each EPI pixel.
    """

    u_dim = R.shape[0]
    d_dim = R.shape[1]
    s_dim = R.shape[2]

    for n in range(n_iter):
        for u in range(u_dim):
            if not M[u]:
                continue
            for d in range(d_dim):
                numerator_sum = 0
                denominator_sum = 0
                for s in range(s_dim):
                    if Mr[u, d, s]:
                        R_Rbar = R[u, d, s] - R_bar[u, d]
                        if abs(R_Rbar / h) <= 1:
                            k = 1 - (R_Rbar / h)**2
                            numerator_sum += k * R[u, d, s]
                            denominator_sum += k

                if denominator_sum > 0: # avoid values not sampled
                    R_bar[u,d] = numerator_sum / denominator_sum

        assert R_bar.shape == (u_dim, d_dim), 'R_bar has wrong shape in function \'calc_r_bar_iter\'.'
    return R_bar