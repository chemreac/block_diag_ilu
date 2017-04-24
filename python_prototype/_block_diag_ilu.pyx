# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

# This wrapper is for debugging / testing purposes only,
# it is fragile and can easily cause segfaults.

cimport numpy as cnp
import numpy as np
from block_diag_ilu cimport ColMajBlockDiagMatrixView, ILU

from cython.operator cimport dereference as deref

cdef class PyILU:
    cdef ILU[double] *thisptr
    cdef ColMajBlockDiagMatrixView[double] *viewptr
    cdef double[::1, :] A
    cdef double[::1] sup, sub
    cdef public int nblocks, blockw, ndiag

    def __cinit__(self,
                  cnp.ndarray[cnp.float64_t, mode='fortran', ndim=2] A,
                  cnp.ndarray[cnp.float64_t, mode='c', ndim=1] sub,
                  cnp.ndarray[cnp.float64_t, mode='c', ndim=1] sup,
                  int blockw, int ndiag):
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
        self.viewptr = new ColMajBlockDiagMatrixView[double](
            NULL, self.nblocks, blockw, ndiag, 0, blockw)
        cdef double * Adata = &A[0, 0]
        for i in range(A.size):
            self.viewptr.m_data[i] = Adata[i]
        for i in range(sub.size):
            self.viewptr.m_data[A.size + i] = sub[i]
        for i in range(sup.size):
            self.viewptr.m_data[A.size + sub.size + i] = sup[i]
        self.thisptr = new ILU[double](deref(self.viewptr))

    @property
    def ny(self):
        return self.nblocks*self.blockw

    def __getitem__(self, key):
        ri, ci = key
        if self.viewptr.valid_index(ri, ci):
            return deref(self.viewptr)(ri, ci)
        else:
            return 0.0

    @property
    def A(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] A = np.empty((self.ny, self.ny))
        for ri in range(self.ny):
            for ci in range(self.ny):
                A[ri, ci] = self[ri, ci]
        return A

    def __dealloc__(self):
        del self.thisptr
        del self.viewptr

    def get_LU(self):
        return self.A

    def solve(self, double[::1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.empty(b.size)
        assert b.shape[0] == self.viewptr.m_nblocks*self.viewptr.m_blockw
        self.thisptr.solve(&b[0], &x[0])
        return x

    def sub_get(self, int di, int bi, int ci):
        if ci > self.blockw:
            raise ValueError
        if bi > self.nblocks - di - 1:
            raise ValueError
        return self.viewptr.sub(di, bi, ci)

    def sup_get(self, int di, int bi, int ci):
        if ci > self.blockw:
            raise ValueError
        if bi > self.nblocks - di - 1:
            raise ValueError
        return self.viewptr.sup(di, bi, ci)

    def piv_get(self, int idx):
        return self.thisptr.m_ilu_inplace.m_ipiv[idx] - 1 # Fortran indices starts at 1

    def rowbycol_get(self, int idx):
        if idx > self.nblocks*self.blockw:
            raise ValueError
        return self.thisptr.m_ilu_inplace.m_rowbycol[idx]

    def colbyrow_get(self, int idx):
        if idx > self.nblocks*self.blockw:
            raise ValueError
        return self.thisptr.m_ilu_inplace.m_colbyrow[idx]
