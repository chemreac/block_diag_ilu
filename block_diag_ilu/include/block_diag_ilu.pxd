cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagMatrixView[T]:
        int m_nsat, m_blockw, m_nblocks
        ColMajBlockDiagMatrixView(T*, int, int, int, int)
        T& block(size_t, int, int)
        T& sub(int, int, int)
        T& sup(int, int, int)
        T& bot(int, int, int)
        T& top(int, int, int)
        void dot_vec(T*, T*)
        void scale_diag_add(ColMajBlockDiagMatrixView[T]&, T, T)
        T& operator()(int, int)

        void set_block(size_t, int, int, T)
        void set_sub(int, int, int, T)
        void set_sup(int, int, int, T)
        void set_bot(int, int, int, T)
        void set_top(int, int, int, T)

    cdef cppclass ILU_inplace[T]:
        ColMajBlockDiagMatrixView[T] m_view

    cdef cppclass ILU[T]:
        ILU(ColMajBlockDiagMatrixView[T])
        int solve(T *, T *)
        ILU_inplace m_ilu_inplace


cdef extern from "block_diag_ilu/banded.hpp" namespace "block_diag_ilu":
    cdef cppclass LU[T]:
        LU(ColMajBlockDiagView[T])
        int solve(T *, T *)
