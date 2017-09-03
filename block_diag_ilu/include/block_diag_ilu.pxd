# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "anyode/anyode_matrix.hpp" namespace "AnyODE":
    cdef cppclass MatrixView[T]:
        pass

    cdef cppclass BandedMatrix[T]:
        pass

cdef extern from "anyode/anyode_decomposition.hpp" namespace "AnyODE":
    cdef cppclass BandedLU[T]:
        BandedLU(BandedMatrix[T]*)
        int solve(T *, T *)
        int factorize()

cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass BlockDiagMatrix[T]:
        int m_ndata, m_nblocks, m_blockw, m_ndiag, m_nsat, m_ld
        T * m_data
        BlockDiagMatrix(T*, int, int, int, int, int)
        T& block(size_t, int, int)
        T& sub(int, int, int)
        T& sup(int, int, int)
        T& bot(int, int, int)
        T& top(int, int, int)
        void dot_vec(T*, T*)
        void scale_diag_add(BlockDiagMatrix[T]&, T, T)
        T& operator()(int, int)
        void set_block(size_t, int, int, T)
        void set_sub(int, int, int, T)
        void set_sup(int, int, int, T)
        void set_bot(int, int, int, T)
        void set_top(int, int, int, T)
        unique_ptr[BandedMatrix[T]] as_banded_padded()
        bool valid_index(int, int)

    cdef cppclass ILU_inplace[T]:
        BlockDiagMatrix[T]* m_view
        ILU_inplace(BlockDiagMatrix[T]*)
        int solve(T *, T *)
