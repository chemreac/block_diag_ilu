# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagView[T]:
        ColMajBlockDiagView(T * const, T * const, T * const, const size_t, const int, const int)
    cdef cppclass ILU_inplace[T]:
        int nblocks()
        int blockw()
        int ndiag()
        ILU_inplace(ColMajBlockDiagView[T])
        void solve(const T * const, T * const)
        T sub_get(const int, const int, const int)
        T sup_get(const int, const int, const int)
        int piv_get(const int)
        int rowbycol_get(const int)
        int colbyrow_get(const int)
