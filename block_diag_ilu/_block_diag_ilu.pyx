# -*- coding: utf-8 -*-
# -*- mode: cython -*-
# distutils: language = c++

cimport numpy as cnp
from cython.operator cimport dereference as deref
from block_diag_ilu cimport ColMajBlockDiagMatrixView, ILU, LU

import numpy as np
from .datastruct import alloc_compressed, diag_data_len


cdef class Compressed:
    cdef ColMajBlockDiagMatrixView[double] *thisptr
    #cdef public double[::1] data

    def __cinit__(self, int nblocks, int blockw, int ndiag, int nsat=0):
        self.view = new ColMajBlockDiagView[double](NULL, nblocks, blockw, ndiag, nsat)

    @property
    def data(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.empty(self.thisptr.m_ndata)
        for i in range(self.thisptr.m_ndata):
            arr[i] = self.thisptr.m_data[i]
        return arr

    def __dealloc__(self):
        del self.view

    @property
    def nsat(self):
        return self.view.m_nsat

    def get_block(self, bi, ri, ci):
        return self.view.block(bi, ri, ci)

    def get_sub(self, di, bi, ci):
        return self.view.sub(di, bi, ci)

    def get_sup(self, di, bi, ci):
        return self.view.sup(di, bi, ci)

    def get_bot(self, si, bi, ci):
        return self.view.bot(si, bi, ci)

    def get_top(self, si, bi, ci):
        return self.view.top(si, bi, ci)

    def set_block(self, bi, ri, ci, value):
        self.view.set_block(bi, ri, ci, value)

    def set_sub(self, di, bi, ci, value):
        self.view.set_sub(di, bi, ci, value)

    def set_sup(self, di, bi, ci, value):
        self.view.set_sup(di, bi, ci, value)

    def set_bot(self, si, bi, ci, value):
        self.view.set_bot(si, bi, ci, value)

    def set_top(self, si, bi, ci, value):
        self.view.set_top(si, bi, ci, value)

    def dot_vec(self, cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] vec):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(vec.size)
        self.view.dot_vec(&vec[0], &out[0])
        return out

    def scale_diag_add(self, Compressed other, scale, diag_add):
        self.view.scale_diag_add(deref(other.view), scale, diag_add)

    def to_dense(self):
        cdef int dim = self.view.m_nblocks*self.view.m_blockw
        cdef cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] A = np.empty((dim, dim))
        for ri in range(dim):
            for ci in range(dim):
                A[ri, ci] = self.view(ri, ci)
        return A


def Compressed_from_dense(cnp.ndarray[cnp.float64_t, ndim=2] A, nblocks, blockw, ndiag, nsat=0):
    if A.shape[0] != A.shape[1]:
        raise ValueError("A not square")
    if A.shape[0] != nblocks*blockw:
        raise ValueError("A shape does not match nblocks & blockw")
    if ndiag > nblocks - 1:
        raise ValueError("too many diagonals")
    cmprs = Compressed(nblocks, blockw, ndiag, nsat)
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


def Compressed_from_data(cnp.ndarray[cnp.float64_t, ndim=1] data, nblocks, blockw, ndiag):
    cdef Compressed cmprs = Compressed(nblocks, blockw, ndiag)
    if (data.size != cmprs.data.size):
        raise ValueError('Incompatible sizes')

    # cmprs.data[:] = data[:]  # <-- does not work
    for i in range(data.size):
        cmprs.data[i] = data[i]
    return cmprs


cdef _check_solve_flag(int flag, int N):
    if flag != 0:
        if flag < N:
            raise ValueError("NaN in b")
        else:
            raise ZeroDivisionError("Rank deficient matrix")


cdef class PyILU:
    cdef ILU[double] *thisptr

    def __cinit__(self, Compressed cmprs):
        self.thisptr = new ILU[double](deref(cmprs.view))

    def __dealloc__(self):
        del self.thisptr

    def solve(self, cnp.ndarray[cnp.float64_t, ndim=1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros_like(b)
        _check_solve_flag(self.thisptr.solve(&b[0], &x[0]), b.size)
        return x


cdef class PyLU:
    cdef LU[double] *thisptr

    def __cinit__(self, Compressed cmprs):
        self.thisptr = new LU[double](deref(cmprs.view))

    def __dealloc__(self):
        del self.thisptr

    def solve(self, cnp.ndarray[cnp.float64_t, ndim=1] b):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.zeros_like(b)
        _check_solve_flag(self.thisptr.solve(&b[0], &x[0]), b.size)
        return x
