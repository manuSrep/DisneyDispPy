#!/usr/bin/python
# -*- coding: utf8 -*-

"""
:author: Manuel Tuschen
:date: 20.06.2016
:license: GPL3
"""

from __future__ import division, absolute_import, unicode_literals, print_function

import numpy as np
from skimage.transform import resize


def fine_to_course(D, course_res):
    """
    Downsample the disparities for the next resolution. Already calculated
    values are propagated.

    Parameters
    ----------
    D : numpy.array [v,s,u].
        Already calculated disparities for the finer resolution
    course_res : numpy.array  []
        The v,u dimension of the courser resolution.

    Returns
    -------
    D_ : numpy.array [v_,s,u_,]
        The update disparities for the courser resolution.
    """

    v_dim = D.shape[0]
    s_dim = D.shape[1]
    u_dim = D.shape[2]

    # Check dimensions of input data
    assert D.shape == (v_dim, s_dim, u_dim), \
        'Input D has wrong input shape in function \'fine_to_course\'.'
    assert course_res[0] < v_dim, \
        'v-dimension of course_res larger than v_dim in function \'fine_to_course\'.'
    assert course_res[1] < u_dim,\
        'u-dimension of course_res larger than u_dim in function \'fine_to_course\'.'

    u_scale = course_res[1] / u_dim
    assert u_scale <= 1, \
        'The scaling factor of u-dimension is larger 1 in function \'fine_to_course\'.'

    D_ = resize(D, (course_res[0], s_dim, course_res[1]), order=0, mode='edge', preserve_range=True)
    D_ *= u_scale

    return D_


def course_to_fine(D_epi_fine, D_epi_course):
    """
    In a last step disparity estimates from lower resolution must be transferred
    to the higher resolutions if there was no value calculated yet.

    Parameters
    ----------
    D_epi_fine : numpy.array [v,s,u]
        Disparity estimates at the finer resolution.
    D_epi_course : numpy.array [v,s,u]
        Disparity estimates at the courser resolution.

    Returns
    -------
    D_epi_fine : numpy.array [v,s,u]
        Updated disparity estimates at the finer resolution.
    """

    v_dim_fine = D_epi_fine.shape[0]
    v_dim_course = D_epi_course.shape[0]
    u_dim_fine = D_epi_fine.shape[2]
    u_dim_course = D_epi_course.shape[2]
    s_dim = D_epi_course.shape[1]

    # Check dimensions of input data
    assert D_epi_fine.shape == (v_dim_fine, s_dim,u_dim_fine), \
        'Input D_epi_fine has wrong input shape in function \'course_to_fine\'.'
    assert D_epi_course.shape == (v_dim_course, s_dim, u_dim_course), \
        'Input D_epi_course has wrong input shape in function \'course_to_fine\'.'
    assert v_dim_course < v_dim_fine, \
        'v-dimension of course image larger than v_dim of fine image in function \'course_to_fine\'.'
    assert u_dim_course < u_dim_fine, \
        'u-dimension of course image larger than v_dim of fine image in function \'course_to_fine\'.'

    u_scale = u_dim_course / u_dim_fine
    assert u_scale <= 1, \
        'The scaling factor of u-dimension is larger 1 in function \'course_to_fine\'.'

    D_tmp = np.zeros((v_dim_fine, s_dim, u_dim_fine), dtype=np.float32)

    for s in range(s_dim):
        D_tmp[:, s] = resize(D_epi_course[:, s], (v_dim_fine, u_dim_fine),
                             order=0, mode='edge', preserve_range=True)
        if not np.all(np.isnan(D_tmp[:, s])):
            assert not np.any(np.isnan(D_tmp[:,s])), 'NaN detected in D_tmp in function \'course_to_fine\'.'
    D_tmp /= u_scale

    replace = np.where(np.isnan(D_epi_fine))
    D_epi_fine[replace] = D_tmp[replace]
    if not np.all(np.isnan(D_epi_fine[:, s])):
        assert not np.any(np.isnan(D_epi_fine[:,s])), 'NaN detected in output D_epi_fine in function \'course_to_fine\'.'

    return D_epi_fine
