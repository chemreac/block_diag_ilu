#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script aims to demonstrate the
benefit (with respect to precision)
of the ILU implementation compared
to e.g. ignoring sub and super diagonals
completely.

Note that the python wrapper is quite inefficient and
hence not suitable for benchmarking.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *

import argh
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

from fakelu import fast_FakeLU

def main(N=5, n=5, ndiag=1, factor=1e-2, seed=42):
    """
    Ax = b
    """
    np.random.seed(seed)
    rnd = np.random.random
    A = np.zeros((N*n, N*n))
    blocks, sub, sup, x_blk = [], [], [], []
    b = rnd(N*n)

    for bi in range(N):
        cur_block = rnd((n, n))
        blocks.append(cur_block)
        slc = slice(n*bi, n*(bi+1))
        A[slc, slc] = cur_block
        x_blk.append(lu_solve(lu_factor(cur_block), b[slc]))

    for di in range(ndiag):
        sub.append(rnd((N-di-1)*n))
        sup.append(rnd((N-di-1)*n))

    fLU = fast_FakeLU(A, n, ndiag)
    x_ref = lu_solve(lu_factor(A), b)
    x_ilu = fLU.solve(b)
    plt.plot(x_ref, 'Ref')
    plt.plot(x_ilu, 'ILU')
    plt.plot(x_blk, 'block')
    plt.show()

if __name__ == '__main__':
    argh.dispatch_command(main)
