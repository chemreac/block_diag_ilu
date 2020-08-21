# -*- coding: utf-8 -*-
# -*- mode: cython -*-
# cython: language_level=3
# distutils: language = c++

from libcpp.memory cimport unique_ptr
cimport numpy as cnp
from cython.operator cimport dereference as deref
from block_diag_ilu cimport BlockDiagMatrix, ILU_inplace, BandedMatrix, BandedLU

import numpy as np
from .datastruct import alloc_compressed, diag_data_len


cdef class PyBlockDiagMatrix:
    cdef BlockDiagMatrix[double] *thisptr
    #cdef public double[::1] data

    def __cinit__(self, int nblocks, int blockw, int ndiag, int nsat=0, int ld=0):
        self.thisptr = new BlockDiagMatrix[double](NULL, nblocks, blockw, ndiag, nsat, ld)

    def copy(self):
        return Compressed_from_data(self.data, self.thisptr.m_nblocks, self.thisptr.m_blockw,
                                    self.thisptr.m_ndiag, self.thisptr.m_nsat, self.thisptr.m_ld)

    @property
    def data(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.empty(self.thisptr.m_ndata)
        for i in range(self.thisptr.m_ndata):
            arr[i] = self.thisptr.m_data[i]
        return arr

    def __dealloc__(self):
        del self.thisptr

    @property
    def nsat(self):
        return self.thisptr.m_nsat

    def get_block(self, bi, ri, ci):
        return self.thisptr.block(bi, ri, ci)

    def get_sub(self, di, bi, ci):
        return self.thisptr.sub(di, bi, ci)

    def get_sup(self, di, bi, ci):
        return self.thisptr.sup(di, bi, ci)

    def get_bot(self, si, bi, ci):
        return self.thisptr.bot(si, bi, ci)

    def get_top(self, si, bi, ci):
        return self.thisptr.top(si, bi, ci)

    def set_block(self, bi, ri, ci, value):
        self.thisptr.set_block(bi, ri, ci, value)

    def set_sub(self, di, bi, ci, value):
        self.thisptr.set_sub(di, bi, ci, value)

    def set_sup(self, di, bi, ci, value):
        self.thisptr.set_sup(di, bi, ci, value)

    def set_bot(self, si, bi, ci, value):
        self.thisptr.set_bot(si, bi, ci, value)

    def set_top(self, si, bi, ci, value):
        self.thisptr.set_top(si, bi, ci, value)

    def dot_vec(self, cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] vec):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(vec.size)
        self.thisptr.dot_vec(&vec[0], &out[0])
        return out

    def scale_diag_add(self, PyBlockDiagMatrix other, scale, diag_add):
        self.thisptr.scale_diag_add(deref(other.thisptr), scale, diag_add)

    def __getitem__(self, key):
        ri, ci = key
        if self.thisptr.valid_index(ri, ci):
            return deref(self.thisptr)(ri, ci)
        else:
            return 0.0

    def to_dense(self):
        cdef int dim = self.thisptr.m_nblocks*self.thisptr.m_blockw
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] A = np.empty((dim, dim))
        for ri in range(dim):
            for ci in range(dim):
                A[ri, ci] = self[ri, ci]
        return A

Compressed = PyBlockDiagMatrix

def Compressed_from_dense(cnp.ndarray[cnp.float64_t, ndim=2] A, nblocks, blockw, ndiag, nsat=0):
    if A.shape[0] != A.shape[1]:
        raise ValueError("A not square")
    if A.shape[0] != nblocks*blockw:
        raise ValueError("A shape does not match nblocks & blockw")
    if ndiag > nblocks - 1:
        raise ValueError("too many diagonals")
    cmprs = PyBlockDiagMatrix(nblocks, blockw, ndiag, nsat, ld=blockw)
    for bi in range(nblocks):
        for ci in range(blockw):
            for ri in range(blockw):
                cmprs.set_block(bi, ri, ci, A[blockw*bi + ri, blockw*bi + ci])
    for di in range(ndiag):
        for bi in range(nblocks - di - 1):
            for ci in range(blockw):
                cmprs.set_sub(di, bi, ci, A[(bi+di+1)*blockw + ci, bi*blockw + ci])
                cmprs.set_sup(di, bi, ci, A[bi*blockw + ci, (bi+di+1)*blockw + ci])
    for si in range(nsat):
        for bi in range(si+1):
            for ci in range(blockw):
                cmprs.set_top(si, bi, ci, A[bi*blockw + ci, (nblocks - si - 1 + bi)*blockw + ci])
                cmprs.set_bot(si, bi, ci, A[(nblocks - si - 1 + bi)*blockw + ci, bi*blockw + ci])
    return cmprs


def Compressed_from_data(cnp.ndarray[cnp.float64_t, ndim=1] data, nblocks, blockw, ndiag, nsat, ld):
    cdef PyBlockDiagMatrix cmprs = PyBlockDiagMatrix(
        nblocks, blockw, ndiag, nsat, ld)
    if (data.size != cmprs.data.size):
        raise ValueError('Incompatible sizes')

    # cmprs.data[:] = data[:]  # <-- does not work
    for i in range(data.size):
        cmprs.thisptr.m_data[i] = data[i]
    return cmprs


cdef _check_solve_flag(int flag, int N):
    if flag != 0:
        if flag < N:
            raise ValueError("NaN in b")
        else:
            raise ZeroDivisionError("Rank deficient matrix")


cdef class PyILU:
    cdef ILU_inplace[double] *thisptr
    cdef PyBlockDiagMatrix pbdm

    def __cinit__(self, PyBlockDiagMatrix pbdm):
        self.pbdm = pbdm.copy()
        self.thisptr = new ILU_inplace[double](self.pbdm.thisptr)

    def __dealloc__(self):
        del self.thisptr

    def solve(self, cnp.ndarray[cnp.float64_t, ndim=1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros_like(b)
        _check_solve_flag(self.thisptr.solve(&b[0], &x[0]), b.size)
        return x


cdef class PyLU:
    cdef unique_ptr[BandedMatrix[double]] bndv
    cdef BandedLU[double] *thisptr

    def __cinit__(self, PyBlockDiagMatrix cmprs):
        self.bndv = cmprs.thisptr.as_banded_padded()
        self.thisptr = new BandedLU[double](self.bndv.get())
        self.thisptr.factorize()

    def __dealloc__(self):
        del self.thisptr

    def solve(self, cnp.ndarray[cnp.float64_t, ndim=1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros_like(b)
        _check_solve_flag(self.thisptr.solve(&b[0], &x[0]), b.size)
        return x
