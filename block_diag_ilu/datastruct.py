# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np


def diag_data_len(N, n, ndiag):
    return n*(N*ndiag - ((ndiag+1)*(ndiag+1) - (ndiag+1))//2)


def alloc_compressed(nblocks, blockw, ndiag, nsat):
    n_block_elem = blockw*blockw*nblocks
    n_diag_elem = 2*diag_data_len(nblocks, blockw, ndiag)
    n_sat_elem = (nsat*nsat + nsat)*blockw
    return np.zeros(n_block_elem + n_diag_elem + n_sat_elem)


def get_compressed(A, N, n, ndiag, cmaj=True):
    """
    Turns a dense matrix (n*N)*(n*N) into a packed storage of
    block diagonal submatrices (column major order) followed by
    sub diagonal data (sequential starting with diagonals closest to
    main diagonal), followed by super diagonal data.

    Parameters
    ----------
    A: 2-dimensional square matrix
    N: int
        number of super-blocks
    n: int
        sub-block dimension
    ndiag: int
        number of sub diagonals (also implies number of
        super diagonals)
    cmaj: bool
        column major ordering in diagonal blocks.

    Raises
    ------
    ValueError on mismatch of A.shape and n*N

    """
    if A.shape != (n*N, n*N):
        raise ValueError("Shape of A != (n*N, n*N)")
    B = np.zeros(n*n*N + 2*diag_data_len(N, n, ndiag))
    idx = 0
    for bi in range(N):
        for imaj in range(n):
            for imin in range(n):
                if cmaj:
                    B[idx] = A[bi*n+imin, bi*n+imaj]
                else:
                    B[idx] = A[bi*n+imaj, bi*n+imin]
                idx += 1
    for di in range(ndiag):
        for bi in range(N-di-1):
            for ci in range(n):
                B[idx] = A[n*(bi+di+1) + ci, n*bi + ci]
                idx += 1
    for di in range(ndiag):
        for bi in range(N-di-1):
            for ci in range(n):
                B[idx] = A[n*bi + ci, n*(bi+di+1) + ci]
                idx += 1
    return B
