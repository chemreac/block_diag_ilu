# -*- coding: utf-8 -*-
# -*- mode: cython-mode -*-

cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ILU:
        const int nblocks, blockw, ndiag
        ILU(double * const, 
            double * const,
            const double * const,
            int, int, int, int)
        void solve(const double * const, double * const)
        double sub_get(const int, const int, const int)
        double sup_get(const int, const int, const int)
        int piv_get(const int)
        int rowbycol_get(const int)
        int colbyrow_get(const int)