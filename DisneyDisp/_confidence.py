#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, \
    print_function

import warnings

import numpy as np
from skimage.morphology import binary_opening, square


def edge_confidence(epi, window=9, threshold=0.02):
    """
    Calculates the edge confidence according to eq. 2.
    This is a simple measurement which designates an edge if pixel intensities
    of gray value changes. An edge is determined by the sum of difference in
    pixel intensity of a central pixel to all pixels in a 1D window of size
    window. It is assumed that there is an edge if the sum is greater than a
    given threshold.

    Parameters
    ----------
    epi : numpy.array [v,u]
        Set of all gray-value epis for scanline s_hat.
    window_size : int, optional
        The 1D window siz in pixels. As the window should be centered
        symmetrically around the pixel to process the value shpuld be odd. For
        even numbers the next higher odd number is chosen.
    threshold : float, optional
        The threshold giving the smallest difference in EPI luminescence which
        must be overcome to designate an edge.

    Returns
    -------
    Ce : numpy.array [v,u]
         Edge confidence values for each EPI pixel.
    Me : numpy.array [v,u] of boolean.
        True means an edge was discovered for at that pixel.
    """

    v_dim = epi.shape[0]
    u_dim = epi.shape[1]
    # Check dimensions of input data
    assert epi.shape == (v_dim, u_dim,), 'Input EPI has wrong shape in function \'edge_confidence\'.'

    # Make window size odd
    if window % 2 == 0:
        warnings.warn(
            'window should be an odd number in function \'edge_confidence\'. Window size {g} was given but {u} is used instead.'.format(
                g=window, u=window + 1))
        window += 1

    # We avoid the border problem by padding the epi.
    padded_epi = np.pad(epi, ((0, 0), (int(window // 2), int(window // 2))), 'edge')
    assert padded_epi.shape == (v_dim, u_dim + window - 1,), 'Padded epi has wrong shape in function \'edge_confidence\'.'

    # Calculate confidence values
    Ce = np.zeros(epi.shape, dtype=np.float32)  # initiate array
    for k in range(window):
        Ce += (epi[...] - padded_epi[:, k:epi.shape[1] + k]) ** 2
    Me = Ce > threshold  # create confidence Mask
    Me = binary_opening(Me, selem=square(2), out=Me)  # work with square to avoid aliasing

    # Let's see if our results have reasonable meaning'
    assert np.all(
        Ce >= 0), 'Negative edge confidence found in function \'edge_confidence\'.'
    assert Ce.shape == (v_dim,
                        u_dim,), 'Ce output has incorrect shape in fucntion \'edge_confidence\'.'
    assert Me.shape == (v_dim,
                        u_dim,), 'Me output has incorrect shape in fucntion \'edge_confidence\'.'
    return Ce, Me


def disparity_confidence(S, Ce, threshold=0.1):
    """
    Calculate the disparity confidence according to eq. 7 Disparity confidence
    is calculated for each pixel u in one line s from an EPI by combining the
    estimated scores found during radiance sampling and the edge confidence
    for each pixel.

    Parameters
    ----------
    S : numpy.array [u,d]
        The calculated scores for each sampled radiance at each pixel u.
    Ce :  numpy.array [u]
         Edge confidence values for each EPI pixel.
    threshold : float, optional
        The threshold which must be overcome for the depth confidence.

    Returns
    -------
    Cd :  numpy.array [u]
         The calculated confidence values for each pixel.
    Md : numpy.array [u] of boolean.
        True means edge confidence is sufficient.
    S_max : numpy.array [u]
        The maximal score over all disparities per pixel.
    S_mean : numpy.array [u]
        The mean score over all disparities per pixel.
    """

    u_dim = S.shape[0]
    d_dim = S.shape[1]
    # Check dimensions of input data
    assert S.shape == (u_dim, d_dim), 'Input S has wrong shape in function \'disparity_confidence\'.'
    assert Ce.shape == (u_dim,), 'Input Ce has wrong shape in function \'disparity_confidence\'.'

    S_max = np.max(S, axis=-1)
    S_mean = np.sum(S, axis=-1)  # might be np.mean but is written as sum in the paper
    Cd = Ce * abs(S_max - S_mean)

    Md = Cd > threshold  # create confidence Mask

    # Let's see if our results have reasonable meaning'
    assert np.all(
        Cd >= 0), 'Negative disparity confidence found in function \'disparity_confidence\'.'
    assert Cd.shape == (
        u_dim,), 'Cd output has incorrect shape in fucntion \'disparity_confidence\'.'
    assert Md.shape == (
        u_dim,), 'Md output has incorrect shape in fucntion \'disparity_confidence\'.'
    return Cd, Md, S_max, S_mean
