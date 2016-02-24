# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import numpy as np
import scipy.linalg

from .. import Compressed_from_dense, ILU, get_include, Compressed_from_data

def test_get_include():
    assert get_include().endswith('include')
    assert os.listdir(get_include())[0] == 'block_diag_ilu.hpp'

def test_Compressed_from_data():
    A = np.array([
        [  9,  3, .1,  0,.01,  0],
        [ -4, 18,  0, .5,  0,.02],
        [ .2,  0,  7, -2,  1,  0],
        [  0,-.1,  3,  5,  0,-.1],
        [.03,  0, .1,  0, -9,  2],
        [  0,.04,  0,.05, -3,  7]
    ])
    b = np.array([-7, 3, 5, -4, 6, 1.5])
    lu, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((lu, piv), b)

    data = np.array([
        9, -4, 3, 18, 7, 3, -2, 5, -9, -3, 2, 7,
        .2, -.1, .1, .05, .03, .04,
        .1, .5, 1, -.1, .01, .02
    ], dtype=np.float64)
    cmprs = Compressed_from_data(data, 3, 2, 2)
    ilu = ILU(cmprs)
    ix = ilu.solve(b)
    assert np.allclose(x, ix, rtol=0.05, atol=1e-6)


def test_Compressed_from_dense():
    A = np.array([
        [  9,  3, .1,  0,.01,  0],
        [ -4, 18,  0, .5,  0,.02],
        [ .2,  0,  7, -2,  1,  0],
        [  0,-.1,  3,  5,  0,-.1],
        [.03,  0, .1,  0, -9,  2],
        [  0,.04,  0,.05, -3,  7]
    ])
    b = np.array([-7, 3, 5, -4, 6, 1.5])
    lu, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((lu, piv), b)

    cmprs = Compressed_from_dense(A, 3, 2, 2)
    ilu = ILU(cmprs)
    ix = ilu.solve(b)
    assert np.allclose(x, ix, rtol=0.05, atol=1e-6)


def _get_test_m1():
    A = np.array([
        [1, 3, 5],
        [2, 4, 7],
        [1, 1, 0]
    ])
    ref = np.array([
        [ 2.0, 4.0, 7.0],
        [ 0.5, 1.0, 1.5],
        [ 0.5,-1.0,-2.0]
    ])
    return A, ref


def _get_test_m2():
    A = np.array([
        [  5,  3,  2,  0,  0,  0],
        [  5,  8,  0,  3,  0,  0],
        [  1,  0,  8,  4,  4,  0],
        [  0,  2,  4,  4,  0,  5],
        [  0,  0,  3,  0,  6,  9],
        [  0,  0,  0,  4,  2,  7]
    ], dtype=np.float64)
    ref = np.array([
        [  5,  3,  2,  0,  0,  0],
        [  1,  5,  0,  3,  0,  0],
        [1/5,  0,  8,  4,  4,  0],
        [  0,2/5,1/2,  2,  0,  5],
        [  0,  0,3/8,  0,  6,  9],
        [  0,  0,  0,4/2,1/3,  4]
    ], dtype=np.float64)
    return A, ref


def _get_test_m3():
    A = np.array([
        [-17, 63, .2,  0],
        [ 37, 13,  0, .3],
        [ .1,  0, 11, 72],
        [  0, .2,-42, 24]
    ], dtype=np.float64)
    pivref = [[1, 1], [1, 1]]

        # [ 37, 13,  0, .3],
        # [-17, 63, .2,  0],
        # [  0, .2,-42, 24]
        # [ .1,  0, 11, 72],

    ref = np.array([
        [ 37,                     13,      0,          .3],
        [-17/37,         63+17/37*13,     .2,           0],
        [     0,     .2/(63+17/37*13),    -42,          24],
        [0.1/37,                   0, -11/42, 72+11/42*24]
    ], dtype=np.float64)
    return A, ref, pivref


def _get_test_m4():
    A = np.array([
        [-17, 63, .2,  0,.02,  0],
        [ 37, 13,  0, .3,  0,.03],
        [ .1,  0, 11, 72,-.1,  0],
        [  0, .2,-42, 24,  0, .2],
        [.03,  0,-.1,  0, 72,-13],
        [  0,-.1,  0,.08, 14,-57]
    ], dtype=np.float64)
    pivref = [[1, 1], [1, 1], [0, 1]]

        # [ 37, 13,  0, .3,  0,.03],
        # [-17, 63, .2,  0,.02,  0],
        # [  0, .2,-42, 24,  0, .2],
        # [ .1,  0, 11, 72,-.1,  0],
        # [.03,  0,-.1,  0, 72,-13],
        # [  0,-.1,  0,.08, 14,-57]

    a =  63 + 17/37*13
    b =  72 + 11/42*24
    c = -57 + 14/72*13
    ref = np.array([
        [    37,        13,      0,          .3,        0,        .03],
        [-17/37,         a,     .2,           0,      .02,          0],
        [     0,      .2/a,     -42,          24,       0,         .2],
        [ .1/37,         0,  -11/42,           b,     -.1,          0],
        [.03/37,         0,   .1/42,           0,      72,        -13],
        [     0,     -.1/a,       0,       .08/b,   14/72,          c],
    ], dtype=np.float64)
    return A, ref, pivref


def FakeLU(A, n, ndiag=0):
    N = A.shape[0]//n
    cmprs = Compressed_from_dense(np.asarray(A, dtype=np.float64), N, n, ndiag)
    return ILU(cmprs)

def test_ilu_solve__1():
    A1, ref1 = _get_test_m1()
    fLU1 = FakeLU(A1, 3)
    xref = [2, 3, 5]
    b = np.array([2+9+25, 4+12+35, 2+3], dtype=np.float64)
    x = fLU1.solve(b)
    assert np.allclose(x, xref)

def test_FakeLU_solve_2():
    A2, ref2 = _get_test_m2()
    fLU2 = FakeLU(A2, 2, 1)
    b = np.array([65, 202, 11, 65, 60, 121], dtype=np.float64)
    # scipy.linalg.lu_solve(scipy.linalg.lu_factor(A2), b) gives xtrue:
    xtrue = [11, 12, -13, 17, 9, 5]

    # but here we verify the errornous result `xref` from the incomplete LU
    # factorization of A2 which is only midly diagonally dominant:
    # LUx = b
    # Ly = b
    yref = [b[0]]
    yref = yref + [b[1] - 1*yref[0]]
    yref = yref + [b[2] - yref[0]/5]
    yref = yref + [b[3] - 2/5*yref[1] - 1/2*yref[2]]
    yref = yref + [b[4] - 3/8*yref[2]]
    yref = yref + [b[5] - 2*yref[3] - 1/3*yref[4]]

    # Ux = y
    xref = [(yref[5]           )/4]
    xref = [(yref[4]-9*xref[-1])/6] + xref
    xref = [(yref[3]-5*xref[-1])/2] + xref
    xref = [(yref[2]-4*xref[-2]-4*xref[-3])/8] + xref
    xref = [(yref[1]-3*xref[-3])/5] + xref
    xref = [(yref[0]-3*xref[-5]-2*xref[-4])/5] + xref
    x = fLU2.solve(b)
    assert np.allclose(xref, x)
    assert not np.allclose(xtrue, x) # <-- shows that the error is intentional


def test_FakeLU_solve_3():
    A3, ref3, pivref3 = _get_test_m3()
    fLU3 = FakeLU(A3, 2, 1)

    b = np.array([-62, 207, 11, -14], dtype=np.float64)
    xtrue = scipy.linalg.lu_solve(scipy.linalg.lu_factor(A3), b.copy())

    # LUx = b
    # Ly = b
    yref = [b[1]]
    yref = yref + [b[0] + 17/37*yref[0]]
    yref = yref + [b[3] - .2/(63+17/37*13)*yref[1]]
    yref = yref + [b[2] - 0.1/37*yref[0] + 11/42*yref[2]]

    # Ux = y
    xref = [(yref[3]           )/(72+11/42*24)]
    xref = [(yref[2]-24*xref[-1])/(-42)] + xref
    xref = [(yref[1]-0.2*xref[-2])/(63+17/37*13)] + xref
    xref = [(yref[0]-0.3*xref[-1] - 13*xref[-3])/37] + xref
    xref = np.array(xref)
    x = fLU3.solve(b.copy())
    assert np.allclose(xref, x)
    assert np.allclose(xtrue, x, rtol=0.01, atol=1e-6)


def test_FakeLU_solve_4():
    A4, ref4, pivref4 = _get_test_m4()
    fLU4 = FakeLU(A4, 2, 2)

    b = np.array([-62, 207, 11, -14, 25, -167], dtype=np.float64)
    xtrue = scipy.linalg.lu_solve(scipy.linalg.lu_factor(A4), b.copy())
    a_ =  63 + 17/37*13
    b_ =  72 + 11/42*24
    c_ = -57 + 14/72*13
    # LUx = b
    # Ly = b
    yref = [b[1]]
    yref = yref + [b[0] + 17/37*yref[0]]
    yref = yref + [b[3] - .2/a_*yref[1]]
    yref = yref + [b[2] - 0.1/37*yref[0] +  11/42*yref[2]]
    yref = yref + [b[4] - .03/37*yref[0] -  .1/42*yref[2]]
    yref = yref + [b[5] + .1/a_*yref[1]   - .08/b_*yref[3]  - 14/72*yref[4]]

    # Ux = y
    xref = [(yref[5]           )/c_]
    xref = [(yref[4]+13*xref[-1])/72] + xref
    xref = [(yref[3]+.1*xref[-2])/b_] + xref
    xref = [(yref[2]-24*xref[-3]-.2*xref[-1])/(-42)] + xref
    xref = [(yref[1]-0.2*xref[-4]-0.02*xref[-2])/a_] + xref
    xref = [(yref[0]-0.03*xref[-1]-0.3*xref[-3] - 13*xref[-5])/37] + xref
    xref = np.array(xref)
    x = fLU4.solve(b.copy())
    assert np.allclose(xref, x)
    assert np.allclose(xtrue, x, rtol=0.03, atol=1e-6)
