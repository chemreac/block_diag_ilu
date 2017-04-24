# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagMatrixView[T]:
        int m_nsat, m_blockw, m_nblocks, m_ndata
        T * m_data
        ColMajBlockDiagMatrixView(T*, int, int, int, int, int)
        bool valid_index(int, int)
        T& operator()(int, int)
        T& sub(int, int, int)
        T& sup(int, int, int)


    cdef cppclass ILU[T]:
        ILU(ColMajBlockDiagMatrixView[T])
        int solve(T *, T *)
        ILU_inplace m_ilu_inplace

    cdef cppclass ILU_inplace[T]:
        int nblocks()
        int blockw()
        int ndiag()
        ILU_inplace(ColMajBlockDiagMatrixView[T])
        void solve(const T * const, T * const)
        int* m_ipiv
        int* m_rowbycol
        int* m_colbyrow
