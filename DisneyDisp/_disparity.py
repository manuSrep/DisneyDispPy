#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import warnings
from math import ceil, floor

import numpy as np
from numba import jit


def estimate_disp(D, S, M, R_bar, disp_range):
    """
    Depth estimates according to eq. 6 for one scanline  s.
    Estimates are only valid, if the values satisfy the edge and depth
    confidence and recomputation is required.

    Parameters
    ----------
    D : numpy.array [u].
        Already calculated disparities.
    S : numpy.array [u,d]
        The calculated scores for each sampled radiance at each pixel u.
    M : numpy.array [u] boolean.
       Mask which values to consider.
    R_bar : numpy.array [u,d].
        The updated convergence radiance for each EPI pixel.
    disp_range : numpy.array [d].
        The range of disparities used during the sampling.

    Returns
    -------
    D : numpy.array [u]
        The updated estimated disparities for each u at the current s-dimension.
    R_bar_best : numpy.array [u].
        The best updated convergence radiance for each EPI pixel.
    S_argmax : numpy.array [u]
        The index of the maximal score.
    """

    u_dim = S.shape[0]
    d_dim = S.shape[1]
    # Check dimensions of input data
    assert D.shape == (u_dim,), 'Input D has wrong number of dimensions in function \'estimate_disp\'.'
    assert S.shape == (u_dim, d_dim), 'Input S has wrong number of dimensions in function \'estimate_disp\'.'
    assert M.shape == (u_dim,), 'Input M has wrong number of dimensions in function \'estimate_disp\'.'
    assert R_bar.shape == (u_dim,d_dim,), 'Input R_bar has wrong number of dimensions in function \'estimate_disp\'.'
    assert disp_range.shape == (d_dim,), 'Input disp has wrong number of dimensions in function \'estimate_disp\'.'

    # We need to find the disparity of maximal score
    S_argmax = np.argmax(S, axis=-1)
    D_best = disp_range[S_argmax]
    R_bar_best = R_bar[np.arange(u_dim), S_argmax]

    # Let's see if our results have reasonable meaning
    assert S_argmax.shape == (u_dim,), 'S_argmax has wrong shape in funcion \'estimate_disp\'.'
    assert D_best.shape == (u_dim,), 'D_best output has wrong shape in funcion \'estimate_disp\'.'
    assert not np.any(np.isnan(D_best)), 'NaN detected in D_best output in fucntion \'estimate_disp\'.'
    assert R_bar_best.shape == (u_dim,), 'R_bar_best output has wrong shape in function \'estimate_disp\'.'

    D[M] = D_best[M]

    return D, R_bar_best, S_argmax


def bilateral_median(Ds, epis, M, Me, window=11, threshold=0.1):
    """
    Smooth the calculated disparities according to eq. 8
    It is assumed that epis and edge_masks are symmetrically distributed around
    the current epi/mask.

    Parameters
    ----------
    Ds : numpy.array [v,u].
        The set of already calculated disparities for each EPI at one scanline.
    epi : numpy.array [v,u]
        Set of all gray-value epis for one scanline.
    M: numpy.array [u] boolean.
       Mask which values to consider.
    Me : numpy.array [v,u] of boolean.
        True means an edge was discovered for at that pixel.
    window : int, optional
        The window size for the bilateral median filter. As the window should
        be centered symetrically around the pixel to process the value shpuld
        be odd. For even numbers the next higher odd number is chosen.
    threshold : float, optional
        The threshold for the filter to determine if two EPI pixels are
        regarded as similar.

    Returns
    -------
    Ds: numpy.array [v,u].
        The smoothed disparities at the current s-dimension.
    """

    v_dim = epis.shape[0]
    u_dim = epis.shape[1]

    # Make window size odd
    if window % 2 == 0:
        warnings.warn(
            'window should be an odd number in function \'bilateral_median\'.')
        window += 1

    # Check dimensions of input data
    assert epis.shape == (v_dim, u_dim,), 'Input epis has wrong number of dimensions in function \'bilateral_median\'.'
    assert Ds.shape == (v_dim, u_dim,), 'Input Ds has wrong number of dimensions in function \'bilateral_median\'.'
    assert Me.shape == (v_dim,u_dim,), 'Input Mes has wrong number of dimensions in function \'bilateral_median\'.'
    assert M.shape == (v_dim, u_dim,), 'Input Ms has wrong number of dimensions in function \'bilateral_median\'.'

    # The epis must be extended in u and v-dimension to avoid a boarder problem
    # pad epis in u-dimension
    epis_padded = np.pad(epis, ((0, 0), (int(window / 2), int(window / 2))),
                         'edge')
    Me_padded = np.pad(Me, ((0, 0), (int(window / 2), int(window / 2))), 'edge')
    Ds_padded = np.pad(Ds, ((0, 0), (int(window / 2), int(window / 2))), 'edge')

    # pad epis in v-dimension
    epis_padded = np.pad(epis_padded,
                         ((int(window / 2), int(window / 2)), (0, 0)), 'edge')
    Me_padded = np.pad(Me_padded, ((int(window / 2), int(window / 2)), (0, 0)),
                       'edge')
    Ds_padded = np.pad(Ds_padded, ((int(window / 2), int(window / 2)), (0, 0)),
                       'edge')

    assert epis_padded.shape == (v_dim + window - 1,
                                 u_dim + window - 1), 'Padded epis array has wrong shape in function \'bilateral_median\'.'
    assert Me_padded.shape == (v_dim + window - 1,
                               u_dim + window - 1), 'Padded Mes array has wrong shape in function \'bilateral_median\'.'
    assert Ds_padded.shape == (v_dim + window - 1,
                               u_dim + window - 1), 'Padded Disps array has wrong shape in function \'bilateral_median\'.'


    smoothed_Disps = bilateral_median_inner(Ds_padded, epis_padded, Me_padded,
                                            M, np.array([v_dim, u_dim]),
                                            window, threshold)

    # Let's see if our results have reasonable meaning'
    assert smoothed_Disps.shape == (v_dim,
                                    u_dim), 'values_to_smooth_D output has wrong shape in function \'bilateral_median\'.'

    Ds[M] = smoothed_Disps[M]  # update values

    return Ds


@jit(nopython=True)
def bilateral_median_inner(Ds_padded, epis_padded, Me_padded, M, v_u_dim,
                           window, threshold):
    """
    Fast inner loop for the bilateral median filter.

    Parameters
    ----------
     Ds_padded : numpy.array [v+window,u+window].
        The set of already calculated disparities for each EPI at one scanline.
        The array was extended to avoid border problems.
    epis_padded : numpy.array [v+window,u+window]
        Set of all gray-value epis for one scanline. The array was extended
        to avoid border problems.
    Me_padded : numpy.array [v+window,u+window] of boolean.
        True means an edge was discovered for at that pixel.
    v_u_dim: numpy.array.
        The original v- and u-dimensions.
    M: numpy.array [u] boolean.
       Mask which values to consider.
    window : int, optional
        The window size for the bilateral median filter. As the window should
        be centered symetrically around the pixel to process the value shpuld
        be odd. For even numbers the next higher odd number is chosen.
    threshold : float, optional
        The threshold for the filter to determine if two EPI pixels are
        regarded as similar.

    Returns
    -------
    smoothed_Ds: numpy.array [v,u].
        The smoothed disparities at the current s-dimension.
    """

    v_dim = v_u_dim[0]
    u_dim = v_u_dim[1]

    smoothed_Ds = np.full((v_dim, u_dim), np.nan,dtype=np.float32)  # we collect all the values

    for v in range(v_dim):
        for u in range(u_dim):

            if not M[v, u]:
                continue
            v_hat = v + int(window // 2)
            u_hat = u + int(window // 2)

            values_to_smooth = []
            for v_ in range(v_hat - int(window // 2),
                            v_hat + int(window // 2) + 1):
                for u_ in range(u_hat - int(window // 2),
                                u_hat + int(window // 2) + 1):
                    assert not np.isnan(epis_padded[
                                            v_hat, u_hat]), 'NaN detected in EPI in function \'bilateral_median_inner\'.'
                    assert not np.isnan(epis_padded[
                                            v_, u_]), 'NaN detected in EPI in function \'bilateral_median_inner\'.'

                    diff = abs(epis_padded[v_hat, u_hat] - epis_padded[v_, u_])
                    if diff < threshold and Me_padded[v_, u_] and not np.isnan(
                            Ds_padded[
                                v_, u_]):  # There can be NaNs in other epis
                        values_to_smooth.append(Ds_padded[v_, u_])

            if len(values_to_smooth) > 0:
                tmp_array = np.zeros((len(values_to_smooth),), dtype=np.float32)
                for i in range(len(values_to_smooth)):
                    tmp_array[i] = values_to_smooth[i]
                smoothed_Ds[v, u] = np.median(tmp_array)

    return smoothed_Ds


def propagation(Ds, epi, R_bar_best, s_hat, threshold=0.1, DEBUG=False):
    """
    Propagate depth estimate for each entry in the current s-dimension.

    Parameters
    ----------
    Ds : numpy.array [v,s,u]
        The set of already calculated disparities for each EPI.
    epi : numpy.array [v,s,u]
        Set of all gray-value epis.
    R_bar_best : numpy.array [v,u]
        The best updated convergence radiance for each EPI pixel.
    s_hat : int
       The current scanline s to sample for.
    threshold : float, optional
        The threshold for the filter to determine if two EPI pixels are
        regarded as similar.
    DEBUG: boolean, optional
        Enable plotting to visualize the propagation process

    Returns
    -------
    Ds : numpy.array [v,s,u.
        The propagated disparities.
    plots : ndarray [v,s,u]
        If plotting was enabled epis with markings of the propagation process.
    """

    v_dim = epi.shape[0]
    s_dim = epi.shape[1]
    u_dim = epi.shape[2]
    # Check dimensions of input data
    assert Ds.shape == (v_dim, s_dim, u_dim,), 'Input Ds has wrong shape in function \'propagation\'.'
    assert epi.shape == (v_dim, s_dim, u_dim,), 'Input epi has wrong shape in function \'propagation\'.'
    assert R_bar_best.shape == (v_dim, u_dim,), 'Input R_bar_best has wrong shape in function \'propagation\'.'
    assert s_hat >= 0 and s_hat < s_dim, 's_hat not in range 0 - s-dim in function \'propagation\'.'

    if DEBUG:
        plots = epi.copy()
    else:
        plots = np.zeros(epi.shape)

    Disps, plot = propagation_inner(Ds, epi, R_bar_best, s_hat, threshold, plots, DEBUG=DEBUG)
    assert Ds.shape == (v_dim, s_dim, u_dim), 'Output Ds has wrong shape in function \'propagation\'.'

    return Ds, plot


@jit(nopython=True)
def propagation_inner(Ds, epi, R_bar_best, s_hat, threshold, plots, DEBUG=False):
    """
    Propagate depth estimate for each entry in the current s-dimension.

    Parameters
    ----------
    Ds : numpy.array [v,s,u]
        The set of already calculated disparities for each EPI.
    epi : numpy.array [v,s,u]
        Set of all gray-value epis.
    R_bar_best : numpy.array [u]
        The best updated convergence radiance for each EPI pixel.
    s_hat: int
       The current scanline s to sample for.
    threshold : float, optional
        The threshold for the filter to determine if two EPI pixels are
        regarded as similar.
    plots : numpy.array [d,s,u]
        If plotting was enabled epis to mark on the sampling process.
    DEBUG : boolean, optional
        Enable plotting to visualize the propagation process

    Returns
    -------
    Ds: numpy.array [v,s,u]
        The propagated disparities.
    plots: ndarray [v,s,u]
        If plotting was enabled epis with markings of the propagation process.
    """

    v_dim = epi.shape[0]
    s_dim = epi.shape[1]
    u_dim = epi.shape[2]

    # Go through all the entries in Disps
    for v in range(v_dim):
        for u in range(u_dim):
            if np.isnan(Ds[v, s_hat, u]):  # we must not propagate NaNs
                continue

            # we need to check if a pixel can be reached by propagation
            # If there is an overlap both pixels are tried
            # go in u direction relative to current u
            for s in range(s_dim):
                x = u + (s_hat - s) * Ds[v, s_hat, u]
                x_floor = int(floor(x))
                x_ceil = int(ceil(x))

                # check the propagation conditions
                if x_floor >= 0 and x_floor < u_dim and np.isnan(
                        Ds[v, s, x_floor]):

                    diff = abs(epi[v, s, x_floor] - R_bar_best[v, u])
                    if diff < threshold:
                        Ds[v, s, x_floor] = Ds[v, s_hat, u]

                    if DEBUG:
                        plots[v, s, x_floor] = 1

                if x_ceil >= 0 and x_ceil < u_dim and np.isnan(
                        Ds[v, s, x_ceil]):

                    diff = abs(epi[v, s, x_ceil] - R_bar_best[v, u])
                    if diff < threshold:
                        Ds[v, s, x_ceil] = Ds[v, s_hat, u]

                    if DEBUG:
                        plots[v, s, x_ceil] = 1

    return Ds, plots
