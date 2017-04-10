cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagView[T]:
        ColMajBlockDiagView(T*, T*, T*, size_t, int, int)
        T& block(size_t, int, int)
        T& sub(int, size_t, int)
        T& sup(int, size_t, int)
        void set_block(size_t, int, int, T)
        void set_sub(int, size_t, int, T)
        void set_sup(int, size_t, int, T)

    cdef cppclass ILU[T]:
        ILU(ColMajBlockDiagView[T])
        int solve(T *, T *)

    cdef cppclass LU[T]:
        LU(ColMajBlockDiagView[T])
        int solve(T *, T *)
