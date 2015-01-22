#ifndef BLOCK_DIAG_ILU_GOB3CSYR2HBHUEX4HJGA3584
#define BLOCK_DIAG_ILU_GOB3CSYR2HBHUEX4HJGA3584
#include <algorithm> // std::max
#include <type_traits>
#include <utility>
#include <memory>
#include <cmath> // std::abs for float and double
#include <cstdlib> // std::abs for int (must include!!)

#ifndef NDEBUG
#include <vector>
#endif

// block_diag_ilu
// ==============
// Algorithm: Incomplete LU factorization of block diagonal matrices with weak sub-/super-diagonals
// Language: C++11
// License: Open Source, see LICENSE.txt (BSD 2-Clause license)
// Author: Björn Dahlgren 2015
// URL: https://github.com/chemreac/block_diag_ilu


namespace block_diag_ilu {

    // make_unique<T[]>() only in C++14, work around:
    // begin copy paste from http://stackoverflow.com/a/10150181/790973
    template <class T, class ...Args>
    typename std::enable_if
    <
        !std::is_array<T>::value,
        std::unique_ptr<T>
        >::type
    make_unique(Args&& ...args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template <class T>
    typename std::enable_if
    <
        std::is_array<T>::value,
        std::unique_ptr<T>
        >::type
    make_unique(std::size_t n)
    {
        typedef typename std::remove_extent<T>::type RT;
        return std::unique_ptr<T>(new RT[n]);
    }
    // end copy paste from http://stackoverflow.com/a/10150181/790973

    // Let's define an alias template for a buffer type which may
    // use (conditional compilation) either std::unique_ptr or std::vector 
    // as underlying data structure.

#define NDEBUG // Force use of std::vector implementation
#ifndef NDEBUG
    template<typename T> using buffer_t = std::unique_ptr<T[]>;
    template<typename T> using buffer_ptr_t = T*;
    // For use in C++14:
    // template<typename T> constexpr auto buffer_factory = make_unique<T>;
    // Work around in C++11:
    template<typename T> inline constexpr buffer_t<T> buffer_factory(std::size_t n) {
        return make_unique<T[]>(n); 
    }
    template<typename T> inline constexpr T* buffer_get_raw_pointer(buffer_t<T>& buf) {
        return buf.get();
    }
#else
    template<typename T> using buffer_t = std::vector<T>;
    template<typename T> using buffer_ptr_t = T*;
    // For use in C++14:
    // template<typename T> constexpr auto buffer_factory = make_unique<T>;
    // Work around in C++11:
    template<typename T> inline constexpr buffer_t<T> buffer_factory(std::size_t n) {
        return buffer_t<T>(n); 
    }    
    template<typename T> inline constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return &buf[0];
    }
#endif
    

#if defined(WITH_BLOCK_DIAG_ILU_DGETRF)
    inline int dgetrf_square(const int dim, double * const __restrict__ a,
                             const int lda, int * const __restrict__ ipiv) noexcept;
#else
    extern "C" void dgetrf_(const int* dim1, const int* dim2, double* a, int* lda, int* ipiv, int* info);
#endif

    extern "C" void dgbtrf_(const int *nrows, const int* ncols, const int* nsub,
                            const int *nsup, double *ab, const int *ldab, int *ipiv, int *info);

    extern "C" void dgbtrs_(const char *trans, const int *dim, const int* nsub,
                            const int *nsup, const int *nrhs, double *ab,
                            const int *ldab, const int *ipiv, double *b, const int *ldb, int *info);

    template <typename Real_t = double>
    class ColMajBlockDiagView {
        Real_t *block_data, *sub_data, *sup_data;
    public:
        const std::size_t nblocks, blockw, ndiag, ld_blocks, block_stride,
            ld_diag, block_data_len, diag_data_len, nouter, dim;
        // ld_block for cache alignment and avoiding false sharing
        // block_stride for avoiding false sharing
        ColMajBlockDiagView(Real_t * const block_data_, Real_t * const sub_data_,
                            Real_t * const sup_data_, const std::size_t nblocks_,
                            const std::size_t blockw_, const std::size_t ndiag_,
                            const std::size_t ld_blocks_=0, const std::size_t block_stride_=0,
                            const std::size_t ld_diag_=0) :
            block_data(block_data_), sub_data(sub_data_), sup_data(sup_data_),
            nblocks(nblocks_), blockw(blockw_), ndiag(ndiag_),
            ld_blocks((ld_blocks_ == 0) ? blockw_ : ld_blocks_),
            block_stride((block_stride_ == 0) ? ld_blocks*blockw : block_stride_),
            ld_diag((ld_diag_ == 0) ? ld_blocks : ld_diag_),
            block_data_len(nblocks*block_stride),
            diag_data_len(nblocks*ndiag - (ndiag*ndiag + ndiag)/2),
            nouter((ndiag == 0) ? blockw-1 : blockw*ndiag),
            dim(blockw*nblocks)
        {}
        inline void set_data_pointers(Real_t *block_data_, Real_t *sub_data_, Real_t *sup_data_){
            this->block_data = block_data_;
            this->sub_data = sub_data_;
            this->sup_data = sup_data_;
        }

        void dot_vec(const Real_t * const vec, Real_t * const out){
            // out need not be zeroed out before call
            const std::size_t nblk = this->nblocks;
            const std::size_t blkw = this->blockw;
            for (std::size_t i=0; i<nblk*blkw; ++i)
                out[i] = 0.0;
            for (std::size_t bri=0; bri<nblk; ++bri)
                for (std::size_t lci=0; lci<blkw; ++lci)
                    for (std::size_t lri=0; lri<blkw; ++lri)
                        out[bri*blkw + lri] += vec[bri*blkw + lci]*\
                            (this->block(bri, lri, lci));
            for (std::size_t di=0; di<this->ndiag; ++di)
                for (std::size_t bi=0; bi<nblk-di-1; ++bi)
                    for (std::size_t ci=0; ci<blkw; ++ci){
                        out[bi*blkw + ci] += this->sup(di, bi, ci)*vec[(bi+di+1)*blkw+ci];
                        out[(bi+di+1)*blkw + ci] += this->sub(di, bi, ci)*vec[bi*blkw+ci];
                    }
    }

    private:
        inline std::size_t diag_idx(const std::size_t diagi, const std::size_t blocki,
                                    const std::size_t coli) const noexcept {
            const std::size_t n_diag_blocks_skip = this->nblocks*diagi - (diagi*diagi + diagi)/2;
            return (n_diag_blocks_skip + blocki)*(this->ld_diag) + coli;
        }
    public:
        inline Real_t& block(const std::size_t blocki, const std::size_t rowi,
                             const std::size_t coli) const noexcept {
            return this->block_data[blocki*this->block_stride + coli*(this->ld_blocks) + rowi];
        }
        inline Real_t& sub(const std::size_t diagi, const std::size_t blocki,
                              const std::size_t coli) const noexcept {
            return this->sub_data[diag_idx(diagi, blocki, coli)];
        }
        inline Real_t& sup(const std::size_t diagi, const std::size_t blocki,
                              const std::size_t coli) const noexcept {
            return this->sup_data[diag_idx(diagi, blocki, coli)];
        }
        inline Real_t get_global(const std::size_t rowi,
                                  const std::size_t coli) const noexcept{
            const int bri = rowi / this->blockw;
            const int bci = coli / this->blockw;
            const int lri = rowi - bri*this->blockw;
            const int lci = coli - bci*this->blockw;
            if (bri == bci)
                return this->block(bri, lri, lci);
            if ((unsigned)std::abs(bri - bci) > ndiag){
                return 0.0;
            }
            if (lri != lci){
                return 0.0;
            }
            if (bri - bci > 0)
                return this->sub(bri-bci-1, bci, lci);
            return this->sup(bci-bri-1, bri, lri);
        }
        inline std::size_t get_banded_ld() const noexcept {
            return 1 + 3*this->nouter; // padded for use with LAPACK's dgbtrf
        }
        inline buffer_t<Real_t> to_banded() const { // std::unique_ptr
            const auto ld_result = this->get_banded_ld();
            auto result = buffer_factory<Real_t>(ld_result*this->dim); //make_unique<Real_t[]>
            for (std::size_t ci=0; ci<this->dim; ++ci){
                const std::size_t row_lower = (ci < this->nouter) ? 0 : ci - this->nouter;
                const std::size_t row_upper = (ci + this->nouter + 1 > this->dim) ? this->dim :
                    ci + this->nouter + 1;
                for (std::size_t ri=row_lower; ri<row_upper; ++ri){
                    result[ld_result*ci + 2*this->nouter + ri - ci] = this->get_global(ri, ci);
                }
            }
            return result;
        }
    };

    class LU {
#ifdef UNIT_TEST
    public:
#endif
        const int dim, nouter, ld;
        buffer_t<double> data;
        buffer_t<int> ipiv;
    public:
        LU(const ColMajBlockDiagView<double>& view) :
            dim(view.dim),
            nouter(view.nouter),
            ld(view.get_banded_ld()),
            data(view.to_banded()),
            ipiv(buffer_factory<int>(view.dim))
        {
            int info;
            dgbtrf_(&this->dim, &this->dim, &this->nouter, &this->nouter, 
                    buffer_get_raw_ptr(this->data),
                    &this->ld, 
                    buffer_get_raw_ptr(this->ipiv), &info);
            if (info){
                throw std::runtime_error("DGBTRF failed.");
            }
        }
        inline void solve(const double * const b, double * const x){
            const char trans = 'N'; // no transpose
            memcpy(x, b, sizeof(double)*this->dim);
            int info, nrhs=1;
            dgbtrs_(&trans, &this->dim, &this->nouter, &this->nouter, &nrhs,
                    buffer_get_raw_ptr(this->data), &this->ld, 
            buffer_get_raw_ptr(this->ipiv), x, &this->dim, &info);
            if (info)
                throw std::runtime_error("DGBTRS failed.");
        };
    };

    template <typename Real_t = double>
    class ColMajBlockDiagMat {
    public:
        buffer_t<Real_t> block_data, sub_data, sup_data;
        ColMajBlockDiagView<Real_t> view;
        ColMajBlockDiagMat(const std::size_t nblocks_,
                           const std::size_t blockw_,
                           const std::size_t ndiag_,
                           const std::size_t ld_blocks_=0,
                           const std::size_t block_stride_=0,
                           const std::size_t ld_diag_=0) :
            view(nullptr, nullptr, nullptr, nblocks_, blockw_,
                 ndiag_, ld_blocks_, block_stride_, ld_diag_)
        {
            this->block_data = buffer_factory<Real_t>(view.block_data_len);
            this->sub_data = buffer_factory<Real_t>(view.diag_data_len);
            this->sup_data = buffer_factory<Real_t>(view.diag_data_len);
            this->view.set_data_pointers(buffer_get_raw_ptr(this->block_data),
                                         buffer_get_raw_ptr(this->sub_data),
                                         buffer_get_raw_ptr(this->sup_data));
        }
        LU factor_lu () const {
            return LU(this->view);
        }
        void dot_vec(const Real_t * const vec, Real_t * const out){
            this->view.dot_vec(vec, out);
        }
    };

    inline void rowpiv2rowbycol(int n, const int * const piv, int * const rowbycol) {
        for (int i = 0; i < n; ++i){
            rowbycol[i] = i;
        }
        for (int i=0; i<n; ++i){
            int j = piv[i] - 1; // Fortran indexing starts with 1
            if (i != j){
                int tmp = rowbycol[j];
                rowbycol[j] = rowbycol[i];
                rowbycol[i] = tmp;
            }
        }
    }

    inline void rowbycol2colbyrow(int n, const int * const rowbycol, int * const colbyrow){
        for (int i=0; i<n; ++i){
            for (int j=0; j<n; ++j){
                if (rowbycol[j] == i){
                    colbyrow[i] = j;
                    break;
                }
            }
        }
    }

    constexpr int diag_store_len(int N, int n, int ndiag) {
        return n*(N*ndiag - (ndiag*ndiag + ndiag)/2);
    }

    class ILU {
        double * const __restrict__ block_data;
        double * const __restrict__ sub_data;
        const double * const __restrict__ sup_data;
    public:
        const int nblocks, blockw, ndiag;
    private:
        int ld_block_data;
        buffer_t<int> piv, rowbycol, colbyrow;

        inline double lu_get(const int blocki, const int rowi,
                             const int coli) const {
            const int blockspan = (this->ld_block_data)*(this->blockw);
            return this->block_data[blocki*blockspan + coli*(this->ld_block_data) + rowi];
        }
    public:
        inline double sub_get(const int diagi, const int blocki,
                              const int coli) const {
            const int skip_ahead = diag_store_len(this->nblocks, this->blockw, diagi);
            return this->sub_data[skip_ahead + blocki*(this->blockw) + coli];
        }
        inline double sup_get(const int diagi, const int blocki, const int coli) const {
            const int skip_ahead = diag_store_len(this->nblocks, this->blockw, diagi);
            return this->sup_data[skip_ahead + blocki*(this->blockw) + coli];
        }
#ifdef UNIT_TEST
        int piv_get(const int idx) { return this->piv[idx]; }
        int rowbycol_get(const int idx) { return this->rowbycol[idx]; }
        int colbyrow_get(const int idx) { return this->colbyrow[idx]; }
#endif
        ILU(double * const __restrict__ block_data,
            double * const __restrict__ sub_data,
            const double * const __restrict__ sup_data,
            int nblocks,
            int blockw, int ndiag,
            int ld_block_data=0)
            // block_data : column major ordering (Fortran style),
            //     (will be overwritten)
            // sub_data : sub[0], sub[1], ... sup[ndiag-1]
            // sup_data : sup[0], sup[1], ... sup[ndiag-1]
            // use ld_block_data to avoid false sharing in parallelized
            // execution (TODO: guard against false sharing in diagdata..)
            : block_data(block_data), sub_data(sub_data), sup_data(sup_data),
              nblocks(nblocks), blockw(blockw), ndiag(ndiag),
              ld_block_data((ld_block_data) ? ld_block_data : blockw),
            piv(buffer_factory<int>(blockw*nblocks)),
              rowbycol(buffer_factory<int>(blockw*nblocks)),
              colbyrow(buffer_factory<int>(blockw*nblocks)) {
            int info_ = 0;
#if defined(WITH_BLOCK_DIAG_ILU_OPENMP)
#pragma omp parallel for
#endif
            for (int bi=0; bi<nblocks; ++bi){
#if defined(WITH_BLOCK_DIAG_ILU_DGETRF)
                int info = dgetrf_square(
                           this->blockw,
                           &block_data[bi*blockw*(this->ld_block_data)],
                           this->ld_block_data,
                           &(this->piv[bi*blockw]));
#else
                int info;
                dgetrf_(&(this->blockw),
                        &(this->blockw),
                        &block_data[bi*blockw*(this->ld_block_data)],
                        &(this->ld_block_data),
                        &(this->piv[bi*blockw]),
                        &info);
#endif
                if ((info != 0) && (info_ == 0))
                    info_ = info;
                for (int ci = 0; ci < blockw; ++ci){
                    for (int di = 0; (di < (this->ndiag)) && (bi+di < (this->nblocks) - 1); ++di){
                        const int skip_ahead = diag_store_len(this->nblocks, this->blockw, di);
                        const int gi = skip_ahead + bi*(this->blockw) + ci;
                        this->sub_data[gi] = sub_data[gi]/(this->lu_get(bi, ci, ci));
                    }
                }
                rowpiv2rowbycol(blockw, &piv[bi*blockw], &rowbycol[bi*blockw]);
                rowbycol2colbyrow(blockw, &rowbycol[bi*blockw], &colbyrow[bi*blockw]);
            }
            if (info_)
                throw std::runtime_error("ILU failed!");
        }
        void solve(const double * const __restrict__ b, double * const __restrict__ x) const {
            // before calling solve: make sure that the
            // block_data and sup_data pointers are still valid.
            auto y = buffer_factory<double>((this->nblocks)*(this->blockw));
            for (int bri = 0; bri < (this->nblocks); ++bri){
                for (int li = 0; li < (this->blockw); ++li){
                    double s = 0.0;
                    for (int lci = 0; lci < li; ++lci){
                        s += this->lu_get(bri, li, lci)*y[bri*(this->blockw) + lci];
                    }
                    for (int di = 1; di < (this->ndiag) + 1; ++di){
                        if (bri >= di) {
                            int ci = this->colbyrow[bri*(this->blockw) + li];
                            s += (this->sub_get(di-1, bri-di, ci) * y[(bri-di)*(this->blockw) + ci]);
                        }
                    }
                    y[bri*(this->blockw) + li] = b[bri*(this->blockw)
                                                   + this->rowbycol[bri*(this->blockw) + li]
                                                   ] - s;
                }
            }
            for (int bri = this->nblocks - 1; bri >= 0; --bri){
                for (int li = this->blockw - 1; li >= 0; --li){
                    double s = 0.0;
                    for (int ci = li + 1; ci < (this->blockw); ++ci)
                        s += this->lu_get(bri, li, ci)*x[bri*(this->blockw) + ci];
                    for (int di = 1; di <= this->ndiag; ++di) {
                        if (bri < this->nblocks - di){
                            int ci = this->colbyrow[bri*this->blockw + li];
                            s += this->sup_get(di-1, bri, ci)*x[(bri+di)*(this->blockw) + ci];
                        }
                    }
                    x[bri*this->blockw+li] = (y[bri*(this->blockw) + li] - s)/(this->lu_get(bri, li, li));
                }
            }
        }
    };

    class BlockDiagMat {
    public:
        const int nblocks, blockw, ndiag, sub_offset, sup_offset;
        const std::size_t data_len;
        buffer_t<double> data;
        BlockDiagMat(int nblocks, int blockw, int ndiag) :
            nblocks(nblocks), blockw(blockw), ndiag(ndiag),
            sub_offset(nblocks*blockw*blockw),
            sup_offset(nblocks*blockw*blockw + diag_store_len(nblocks, blockw, ndiag)),
            data_len(nblocks*blockw*blockw+2*diag_store_len(nblocks, blockw, ndiag)),
            data(buffer_factory<double>(data_len))
            {}
        inline void zero_out_all() noexcept {
            for (std::size_t i=0; i<(this->data_len); ++i){
                this->data[i] = 0.0;
            }
        }
        inline void zero_out_diags() noexcept {
            for (std::size_t i=(this->sub_offset); i<(this->data_len); ++i){
                this->data[i] = 0.0;
            }
        }
        inline double& block(int bi, int ri, int ci) noexcept {
            return data[bi*(this->blockw)*(this->blockw) + ci*(this->blockw) + ri];
        }
        inline double& sub(int di, int bi, int lci) noexcept {
            return data[this->sub_offset + diag_store_len(this->nblocks, this->blockw, di) + bi*(this->blockw) + lci];
        }
        inline double& sup(int di, int bi, int lci) noexcept {
            return data[this->sup_offset + diag_store_len(this->nblocks, this->blockw, di) + bi*(this->blockw) + lci];
        }

#ifdef UNIT_TEST
        double get(int ri, int ci) {
            const int bri = ri / this->blockw;
            const int bci = ci / this->blockw;
            const int lri = ri - bri*this->blockw;
            const int lci = ci - bci*this->blockw;
            if (bri == bci)
                return this->block(bri, lri, lci);
            if (std::abs(bri - bci) > ndiag)
                return 0.0;
            if (lri != lci)
                return 0.0;
            if (bri - bci > 0)
                return this->sub(bri-bci-1, bci, lci);
            return this->sup(bci-bri-1, bri, lri);
        }
#endif

        void set_to_1_minus_gamma_times_other(double gamma, BlockDiagMat &other) {
            // Scale main blocks by -gamma
            for (int bi=0; bi<this->nblocks; ++bi)
                for (int ci=0; ci<this->blockw; ++ci)
                    for (int ri=0; ri<this->blockw; ++ri)
                        this->block(bi, ri, ci) = -gamma*other.block(bi, ri, ci);

            // Add the identiy matrix
            for (int bi=0; bi<this->nblocks; ++bi)
                for (int ci=0; ci<this->blockw; ++ci)
                    this->block(bi, ci, ci) += 1;

            // Scale diagonals by -gamma
            for (int di=0; di<this->ndiag; ++di)
                for (int bi=0; bi<this->nblocks-di-1; ++bi)
                    for (int ci=0; ci<this->blockw; ++ci){
                        this->sub(di, bi, ci) = -gamma*other.sub(di, bi, ci);
                        this->sup(di, bi, ci) = -gamma*other.sup(di, bi, ci);
                    }
        }

        // The end user must assure that the underlying data is not freed.
        ILU ilu_inplace() {
            return ILU(&this->data[0],
                       &this->data[this->sub_offset],
                       &this->data[this->sup_offset],
                       this->nblocks, this->blockw, this->ndiag);
        }
        void dot_vec(const double * const vec, double * const out){
            // out need not be zeroed out before call
            const int nblocks = this->nblocks;
            const int blockw = this->blockw;
            for (int i=0; i<nblocks*blockw; ++i)
                out[i] = 0.0;
            for (int bri=0; bri<nblocks; ++bri)
                for (int lci=0; lci<blockw; ++lci)
                    for (int lri=0; lri<blockw; ++lri)
                        out[bri*blockw + lri] += vec[bri*blockw + lci]*(this->block(bri, lri, lci));
            for (int di=0; di<this->ndiag; ++di)
                for (int bi=0; bi<nblocks-di-1; ++bi)
                    for (int ci=0; ci<blockw; ++ci){
                        out[bi*blockw + ci] += this->sup(di, bi, ci)*vec[(bi+di+1)*blockw+ci];
                        out[(bi+di+1)*blockw + ci] += this->sub(di, bi, ci)*vec[bi*blockw+ci];
                    }
        }
    };

}

#if defined(WITH_BLOCK_DIAG_ILU_DGETRF)
inline int block_diag_ilu::dgetrf_square(const int dim, double * const __restrict__ a,
                                         const int lda, int * const __restrict__ ipiv) noexcept {
    // Unblocked algorithm for LU decomposition of square matrices
    // employing Doolittle's algorithm with rowswaps.
    //
    // ipiv indexing starts at 1 (Fortran compability)
    if (dim == 0) return 0;

    int info = 0;
    auto A = [&](int ri, int ci) -> double& { return a[ci*lda + ri]; };
    auto swaprows = [&](int ri1, int ri2) { // this is not cache friendly
        for (int ci=0; ci<dim; ++ci){
            double temp = A(ri1, ci);
            A(ri1, ci) = A(ri2, ci);
            A(ri2, ci) = temp;
        }
    };

    for (int i=0; i<dim-1; ++i) {
        int pivrow = i;
        double absmax = std::abs(A(i, i));
        for (int j=i; j<dim; ++j) {
            // Find pivot
            double curabs = std::abs(A(j, i));
            if (curabs > absmax){
                absmax = curabs;
                pivrow = j;
            }
        }
        if ((absmax == 0) && (info == 0))
            info = pivrow+1;
        ipiv[i] = pivrow+1;
        if (pivrow != i) {
            // Swap rows
            swaprows(i, pivrow);
        }
        // Eliminate in column
        for (int ri=i+1; ri<dim; ++ri){
            A(ri, i) = A(ri, i)/A(i, i);
        }
        // Subtract from rows
        for (int ci=i+1; ci<dim; ++ci){
            for (int ri=i+1; ri<dim; ++ri){
                A(ri, ci) -= A(ri, i)*A(i, ci);
            }
        }
    }
    ipiv[dim-1] = dim;
    return info;
}
#endif

#endif
