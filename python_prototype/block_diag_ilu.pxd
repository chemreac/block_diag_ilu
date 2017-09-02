# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass BlockDiagMatrix[T]:
        int m_ndata, m_nblocks, m_blockw, m_ndiag, m_nsat, m_ld
        T * m_data
        BlockDiagMatrix(T*, int, int, int, int, int)
        bool valid_index(int, int)
        T& operator()(int, int)
        T& sub(int, int, int)
        T& sup(int, int, int)

    cdef cppclass ILU_inplace[T]:
        ILU_inplace(BlockDiagMatrix[T] *)
        void solve(const T * const, T * const)
        int* m_ipiv
        int* m_rowbycol
        int* m_colbyrow
