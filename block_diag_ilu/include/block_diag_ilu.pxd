cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagView[T]:
        int m_nsat, m_blockw, m_nblocks
        ColMajBlockDiagView(T*, T*, T*, int, int, int, T*, int)
        T& block(size_t, int, int)
        T& sub(int, int, int)
        T& sup(int, int, int)
        T& sat(int, int, int) except +
        void set_block(size_t, int, int, T)
        void set_sub(int, int, int, T)
        void set_sup(int, int, int, T)
        void set_sat(int, int, int, T) except +
        void dot_vec(T*, T*)
        void scale_diag_add(ColMajBlockDiagView[T]&, T, T)
        T get_global(int, int)

    cdef cppclass ILU_inplace[T]:
        ColMajBlockDiagView[T] m_view

    cdef cppclass ILU[T]:
        ILU(ColMajBlockDiagView[T])
        int solve(T *, T *)
        ILU_inplace m_ilu_inplace


cdef extern from "block_diag_ilu/banded.hpp" namespace "block_diag_ilu":
    cdef cppclass LU[T]:
        LU(ColMajBlockDiagView[T])
        int solve(T *, T *)
