cdef extern from "block_diag_ilu.hpp" namespace "block_diag_ilu":
    cdef cppclass ColMajBlockDiagView[T]:
        ColMajBlockDiagView(T*, T*, T*, size_t, unsigned, unsigned)
        T& block(size_t, unsigned, unsigned)
        T& sub(unsigned, size_t, unsigned)
        T& sup(unsigned, size_t, unsigned)
        void set_block(size_t, unsigned, unsigned, T)
        void set_sub(unsigned, size_t, unsigned, T)
        void set_sup(unsigned, size_t, unsigned, T)

    cdef cppclass ILU[T]:
        ILU(ColMajBlockDiagView[T])
        void solve(T *, T *)

    cdef cppclass LU[T]:
        LU(ColMajBlockDiagView[T])
        void solve(T *, T *)
