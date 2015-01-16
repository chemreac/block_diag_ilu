# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -DUNIT_TEST
# distutils: libraries = ['lapack']

# This wrapper is for debugging / testing purposes only,
# it is fragile and can easily cause segfaults.

cimport numpy as cnp
import numpy as np
from block_diag_ilu cimport ILU

cdef class PyILU:
    cdef ILU *thisptr
    cdef double[::1, :] A
    cdef double[::1] sup, sub
    cdef public int nblocks, blockw, ndiag

    def __cinit__(self, double[::1, :] A, double[::1] sub,
                  double[::1] sup, int blockw, int ndiag):
        assert A.shape[0] == blockw
        assert A.shape[1] % blockw == 0
        self.nblocks = A.shape[1] // blockw
        self.blockw = blockw
        self.ndiag = ndiag

        cdef int diag_len = 0
        for i in range(ndiag):
            diag_len += (self.nblocks-i-1)*blockw
        assert sub.size >= diag_len
        assert sup.size >= diag_len
        # Make sure data isn't free:ed while still possibly
        # referenced by ILU object:
        self.A = A
        self.sup = sup
        self.sub = sub
        self.thisptr = new ILU(&self.A[0,0], &sub[0], &self.sup[0],
                               self.nblocks, blockw, ndiag, blockw)

    def __dealloc__(self):
        del self.thisptr

    def get_LU(self):
        return self.A

    def solve(self, double[::1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.empty(b.size)
        assert b.shape[0] == self.thisptr.nblocks*self.thisptr.blockw
        self.thisptr.solve(&b[0], &x[0])
        return x

    def sub_get(self, int di, int bi, int ci):
        if ci > self.blockw:
            raise ValueError
        if bi > self.nblocks - di - 1:
            raise ValueError
        return self.thisptr.sub_get(di, bi, ci)

    def sup_get(self, int di, int bi, int ci):
        if ci > self.blockw:
            raise ValueError
        if bi > self.nblocks - di - 1:
            raise ValueError
        return self.thisptr.sup_get(di, bi, ci)

    def piv_get(self, int idx):
        return self.thisptr.piv_get(idx) - 1 # Fortran indices starts at 1

    def rowbycol_get(self, int idx):
        if idx > self.nblocks*self.blockw:
            raise ValueError
        return self.thisptr.rowbycol_get(idx)

    def colbyrow_get(self, int idx):
        if idx > self.nblocks*self.blockw:
            raise ValueError
        return self.thisptr.colbyrow_get(idx)
