#pragma once
#include <algorithm> // std::max
#include <type_traits>
#include <utility>
#include <memory>
#include <cmath> // std::abs for float and double, std::sqrt, std::isnan
#include <cstdlib> // std::abs for int (must include!!)
#include <cstring> // memcpy

#if !defined(NDEBUG)
#include <vector>
#endif

// block_diag_ilu
// ==============
// Algorithm: Incomplete LU factorization of block diagonal matrices with weak sub-/super-diagonals
// Language: C++14
// License: Open Source, see LICENSE (BSD 2-Clause license)
// Author: Björn Dahlgren 2015
// URL: https://github.com/chemreac/block_diag_ilu


namespace block_diag_ilu {

    // Let's define an alias template for a buffer type which may
    // use (conditional compilation) either std::unique_ptr or std::vector
    // as underlying data structure.

#ifdef NDEBUG
    template<typename T> using buffer_t = std::unique_ptr<T[]>;
    template<typename T> using buffer_ptr_t = T*;
    template<typename T> constexpr auto buffer_factory = std::make_unique<T[]>;
    template<typename T> constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return buf.get();
    }
#else
    template<typename T> using buffer_t = std::vector<T>;
    template<typename T> using buffer_ptr_t = T*;
    template<typename T> constexpr buffer_t<T> buffer_factory(int n) {
        return buffer_t<T>(n);
    }
    template<typename T> constexpr T* buffer_get_raw_ptr(buffer_t<T>& buf) {
        return &buf[0];
    }
#endif

#if defined(BLOCK_DIAG_ILU_WITH_DGETRF)
    template<typename T> int getrf_square(const int dim, T * const __restrict__ a,
                                                  const int lda, int * const __restrict__ ipiv) noexcept;
#else
    extern "C" void dgetrf_(const int* dim1, const int* dim2, double* a, int* lda, int* ipiv, int* info);
#endif

    extern "C" void dgbtrf_(const int *nrows, const int* ncols, const int* nsub,
                            const int *nsup, double *ab, const int *ldab, int *ipiv, int *info);

    extern "C" void dgbtrs_(const char *trans, const int *dim, const int* nsub,
                            const int *nsup, const int *nrhs, const double *ab,
                            const int *ldab, const int *ipiv, double *b, const int *ldb, int *info);

    constexpr int nouter_(int blockw, int ndiag) { return (ndiag == 0) ? blockw-1 : blockw*ndiag; }
    constexpr int banded_ld_(int nouter, int offset=-1) {
        return 1 + 2*nouter + ((offset == -1) ? nouter : offset); // padded for use with LAPACK's dgbtrf
    }

    template <typename T>
    int check_nan(const T * const arr, int n){
        // Returns the index of the first occurence of NaN
        // in input array (starting at 1), returns 0 if no NaN is encountered
        //
        // Parameters
        // ----------
        // arr: pointer to array of doubles to be checked for occurence of NaN
        // n: length of array
        //
        // Notes
        // -----
        // isnan is defined in cmath.h
        for (int i=0; i<n; ++i)
            if (std::isnan(arr[i]))
                return i+1;
        return 0; // if no NaN is encountered, -0is returned
    }

    template <class T, typename Real_t = double>
    struct ViewBase {
        const int blockw, ndiag;
        const int nblocks, nouter, dim;
        ViewBase(int blockw, int ndiag, int nblocks)
            : blockw(blockw), ndiag(ndiag), nblocks(nblocks),
              nouter(nouter_(blockw, ndiag)), dim(blockw*nblocks) {}

        Real_t get_global(const int rowi, const int coli) const noexcept{
            const auto& self = *static_cast<const T*>(this);  // CRTP
            const int bri = rowi / self.blockw;
            const int bci = coli / self.blockw;
            const int lri = rowi - bri*self.blockw;
            const int lci = coli - bci*self.blockw;
            if (bri == bci)
                return self.block(bri, lri, lci);
            if (lri != lci)
                return 0.0;
            if (bri > bci){ // sub diagonal
                if ((bri - bci) > ndiag)
                    return 0.0;
                else
                    return self.sub(bri-bci-1, bci, lci);
            } else { // super diagonal
                if ((bci - bri) > ndiag)
                    return 0.0;
                else
                    return self.sup(bci-bri-1, bri, lri);
            }
        }
        int get_banded_ld() const noexcept {
            return banded_ld_(static_cast<const T*>(this)->nouter);  // CRTP
        }
#if defined(BLOCK_DIAG_ILU_PY)
        // Cython work around: https://groups.google.com/forum/#!topic/cython-users/j58Sp3QMrD4
        void set_block(const int blocki, const int rowi,
                       const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.block(blocki, rowi, coli) = value;
        }
        void set_sub(const int diagi, const int blocki,
                     const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.sub(diagi, blocki, coli) = value;
        }
        void set_sup(const int diagi, const int blocki,
                     const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.sup(diagi, blocki, coli) = value;
        }
#endif
    };



    template <typename Real_t = double, bool col_maj = true>
    class DenseView : public ViewBase<DenseView<Real_t, col_maj>, Real_t> {
        // For use with LAPACK's dense matrix layout
    public:
        Real_t *data;
        const int ld;

        DenseView(Real_t *data, const int nblocks, const int blockw, const int ndiag, const int ld_=0)
            : ViewBase<DenseView<Real_t, col_maj>, Real_t>(blockw, ndiag, nblocks),
              data(data), ld((ld_ == 0) ? blockw*nblocks : ld_) {}
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            const int imaj = (this->blockw)*blocki + (col_maj ? coli : rowi);
            const int imin = (this->blockw)*blocki + (col_maj ? rowi : coli);
            return this->data[imaj*ld + imin];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int li) const noexcept {
            const int imaj = (this->blockw)*(blocki + (col_maj ? 0 : diagi + 1)) + li;
            const int imin = (this->blockw)*(blocki + (col_maj ? diagi + 1 : 0)) + li;
            return this->data[imaj*ld + imin];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int li) const noexcept {
            const int imaj = (this->blockw)*(blocki + (col_maj ? diagi + 1 : 0)) + li;
            const int imin = (this->blockw)*(blocki + (col_maj ? 0 : diagi + 1)) + li;
            return this->data[imaj*ld + imin];
        }
    };

    template <typename Real_t = double>
    class ColMajBandedView : public ViewBase<ColMajBandedView<Real_t>, Real_t> {
        // For use with LAPACK's banded matrix layout.
        // Note that the matrix is padded with ``mlower`` extra bands.
    public:
        Real_t *data;
        const int ld, offset;

        ColMajBandedView(Real_t *data, const int nblocks, const int blockw, const int ndiag,
                         const int ld_=0, int offset_=-1)
            : ViewBase<ColMajBandedView<Real_t>, Real_t>(blockw, ndiag, nblocks),
              data(data), ld((ld_ == 0) ? banded_ld_(nouter_(blockw, ndiag), offset_) : ld_),
              offset((offset_ == -1) ? nouter_(blockw, ndiag) : offset_) {}
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            const int imaj = blocki*(this->blockw) + coli;
            const int imin = offset + this->nouter + rowi - coli;
            return this->data[imaj*ld + imin];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int coli) const noexcept {
            const int imaj = blocki*(this->blockw) + coli;
            const int imin = offset + this->nouter + (diagi + 1)*(this->blockw);
            return this->data[imaj*ld + imin];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int coli) const noexcept {
            const int imaj = (blocki + diagi + 1)*(this->blockw) + coli;
            const int imin = offset + this->nouter - (diagi+1)*(this->blockw);
            return this->data[imaj*ld + imin];
        }
    };

    template <typename Real_t = double> class ColMajBlockDiagMat;
    template <typename Real_t = double>
    class ColMajBlockDiagView : public ViewBase<ColMajBlockDiagView<Real_t>, Real_t> {
    public:
        Real_t *block_data, *sub_data, *sup_data;
        // int will suffice, decomposition scales as N**3 even iterative methods (N**2) would need months at 1 TFLOPS
        const int ld_blocks;
        const int block_stride;
        const int ld_diag;
        const int block_data_len, diag_data_len;
        // ld_block for cache alignment and avoiding false sharing
        // block_stride for avoiding false sharing
        ColMajBlockDiagView(Real_t * const block_data, Real_t * const sub_data,
                            Real_t * const sup_data, const int nblocks,
                            const int blockw, const int ndiag,
                            const int ld_blocks_=0, const int block_stride_=0,
                            const int ld_diag_=0) :
            ViewBase<ColMajBlockDiagView<Real_t>, Real_t>(blockw, ndiag, nblocks),
            block_data(block_data), sub_data(sub_data), sup_data(sup_data),
            ld_blocks((ld_blocks_ == 0) ? blockw : ld_blocks_),
            block_stride((block_stride_ == 0) ? ld_blocks*blockw : block_stride_),
            ld_diag((ld_diag_ == 0) ? ld_blocks : ld_diag_),
            block_data_len(nblocks*block_stride),
            diag_data_len(ld_diag*(nblocks*ndiag - (ndiag*ndiag + ndiag)/2))
            {}
        void scale_diag_add(const ColMajBlockDiagView<Real_t>& source, Real_t scale=1, Real_t diag_add=0){
            const auto nblocks = this->nblocks;
            const auto blockw = this->blockw;
            for (int bi = 0; bi < nblocks; ++bi){
                for (int ci=0; ci < blockw; ++ci){
                    for (int ri = 0; ri < blockw; ++ri){
                        this->block(bi, ri, ci) = scale * source.block(bi, ri, ci);
                    }
                }
            }
            for (int bi = 0; bi < nblocks; ++bi){
                for (int ci = 0; ci < blockw; ++ci){
                    this->block(bi, ci, ci) += diag_add;
                }
            }
            for (int di = 0; di < this->ndiag; ++di) {
                for (int bi=0; bi < ((nblocks <= di+1) ? 0 : nblocks - di - 1); ++bi) {
                    for (int ci = 0; ci < blockw; ++ci){
                        this->sub(di, bi, ci) = scale * source.sub(di, bi, ci);
                        this->sup(di, bi, ci) = scale * source.sup(di, bi, ci);
                    }
                }
            }
        }
        ColMajBlockDiagMat<Real_t> copy_to_matrix() const {
            auto mat = ColMajBlockDiagMat<Real_t> {this->nblocks, this->blockw, this->ndiag, ld_blocks, block_stride, ld_diag};
            mat.view.scale_diag_add(*this);
            return mat;
        }
        void set_data_pointers(buffer_ptr_t<Real_t> block_data_,
                                      buffer_ptr_t<Real_t> sub_data_,
                                      buffer_ptr_t<Real_t> sup_data_) noexcept {
            this->block_data = block_data_;
            this->sub_data = sub_data_;
            this->sup_data = sup_data_;
        }
        void dot_vec(const Real_t * const vec, Real_t * const out){
            // out need not be zeroed out before call
            const auto nblk = this->nblocks;
            const auto blkw = this->blockw;
            for (int i=0; i<nblk*blkw; ++i){
                out[i] = 0.0;
            }
            for (int bri=0; bri<nblk; ++bri){
                for (int lci=0; lci<blkw; ++lci){
                    for (int lri=0; lri<blkw; ++lri){
                        out[bri*blkw + lri] += vec[bri*blkw + lci]*\
                            (this->block(bri, lri, lci));
                    }
                }
            }
            for (int di=0; di<this->ndiag; ++di){
                for (int bi=0; bi<nblk-di-1; ++bi){
                    for (int ci=0; ci<blkw; ++ci){
                        out[bi*blkw + ci] += this->sup(di, bi, ci)*vec[(bi+di+1)*blkw+ci];
                        out[(bi+di+1)*blkw + ci] += this->sub(di, bi, ci)*vec[bi*blkw+ci];
                    }
                }
            }
        }
        Real_t rms_diag(int diag_idx) {
            // returns the root means square of `diag_idx`:th diagonal
            // (diag_idx < 0 denotes sub diagonals, diag_idx == 0 deontes main diagonal,
            // and diag_idx > 0 denotes super diagonals)
            Real_t sum = 0;
            int nelem;
            if (diag_idx == 0){
                nelem = (this->nblocks)*(this->blockw);
                for (int bi = 0; bi < (this->nblocks); ++bi){
                    for (int ci = 0; ci < (this->blockw); ++ci){
                        const Real_t elem = this->block(bi, ci, ci);
                        sum += elem*elem;
                    }
                }
            } else if (diag_idx < 0) {
                if ((-diag_idx) >= this->nblocks)
                    return 0;
                nelem = (this->nblocks + diag_idx)*(this->blockw);
                for (int bi = 0; bi < (this->nblocks)+diag_idx ; ++bi){
                    for (int ci = 0; ci < (this->blockw); ++ci){
                        const Real_t elem = this->sub(-diag_idx - 1, bi, ci);
                        sum += elem*elem;
                    }
                }
            } else {
                if (diag_idx >= this->nblocks)
                    return 0;
                nelem = (this->nblocks - diag_idx)*(this->blockw);
                for (int bi = 0; bi < (this->nblocks)-diag_idx ; ++bi){
                    for (int ci = 0; ci < (this->blockw); ++ci){
                        const Real_t elem = this->sup(diag_idx - 1, bi, ci);
                        sum += elem*elem;
                    }
                }
            }
            return std::sqrt(sum/nelem);
        }
        Real_t average_diag_weight(int di){  // di >= 0
            Real_t off_diag_factor = 0;
            for (int bi = 0; bi < this->nblocks; ++bi){
                for (int li = 0; li < this->blockw; ++li){
                    const Real_t diag_val = this->block(bi, li, li);
                    if (bi < this->nblocks - di - 1){
                        off_diag_factor += std::abs(diag_val/this->sub(di, bi, li));
                    }
                    if (bi > di){
                        off_diag_factor += std::abs(diag_val/this->sup(di, bi - di - 1, li));
                    }
                }
            }
            return off_diag_factor / ((this->blockw) * (this->nblocks - 1 - di) * 2);
        }

    private:
        int diag_idx(const int diagi, const int blocki,
                                    const int coli) const noexcept {
            const int n_diag_blocks_skip = this->nblocks*diagi - (diagi*diagi + diagi)/2;
            return (n_diag_blocks_skip + blocki)*(this->ld_diag) + coli;
        }
    public:
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            return this->block_data[blocki*this->block_stride + coli*(this->ld_blocks) + rowi];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int coli) const noexcept {
            return this->sub_data[diag_idx(diagi, blocki, coli)];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int coli) const noexcept {
            return this->sup_data[diag_idx(diagi, blocki, coli)];
        }
        buffer_t<Real_t> to_banded() const {
            const auto ld_result = this->get_banded_ld();
            auto result = buffer_factory<Real_t>(ld_result*this->dim);
            for (int ci = 0; ci < this->dim; ++ci){
                const int row_lower = (ci < this->nouter) ? 0 : ci - this->nouter;
                const int row_upper = (ci + this->nouter + 1 > this->dim) ? this->dim :
                    ci + this->nouter + 1;
                for (int ri=row_lower; ri<row_upper; ++ri){
                    result[ld_result*ci + 2*this->nouter + ri - ci] = this->get_global(ri, ci);
                }
            }
            return result;
        }
        void set_to_1_minus_gamma_times_view(Real_t gamma, ColMajBlockDiagView &other) {
            scale_diag_add(other, -gamma, 1);
        }
        void zero_out_blocks() noexcept {
            for (int i=0; i<(this->block_data_len); ++i){
                this->block_data[i] = 0.0;
            }
        }
        void zero_out_diags() noexcept {
            for (int i=0; i<(this->diag_data_len); ++i){
                this->sub_data[i] = 0.0;
            }
            for (int i=0; i<(this->diag_data_len); ++i){
                this->sup_data[i] = 0.0;
            }
        }
    };

    template <typename Real_t = double>
    class LU {  // Wrapper around DGBTRF & DGBTRS from LAPACK
        static_assert(sizeof(Real_t) == 8, "LAPACK DGBTRF & DGBTRS operates on 64-bit IEEE 754 floats.");
#ifdef BLOCK_DIAG_ILU_UNIT_TEST
    public:
#endif
        const int dim, nouter, ld;
        buffer_t<Real_t> data;
        buffer_t<int> ipiv;
    public:
        LU(const ColMajBlockDiagView<Real_t>& view) :
            dim(view.dim),
            nouter(view.nouter),
            ld(view.get_banded_ld()),
            data(view.to_banded()),
            ipiv(buffer_factory<int>(view.dim))
        {
            factorize();
        }
        LU(const ColMajBandedView<Real_t>& view) :
            dim(view.dim),
            nouter(view.nouter),
            ld(view.ld),
            data(buffer_factory<Real_t>(view.ld*view.dim)),
            ipiv(buffer_factory<int>(view.dim))
        {
            std::memcpy(&data[0], view.data, sizeof(Real_t)*view.ld*view.dim);
            if (view.ld != banded_ld_(view.nouter)){
                throw std::runtime_error("LAPACK requires padding");
            }
            factorize();
        }
        void factorize(){
            int info;
            dgbtrf_(&this->dim, &this->dim, &this->nouter, &this->nouter,
                    buffer_get_raw_ptr(this->data),
                    &this->ld,
                    buffer_get_raw_ptr(this->ipiv), &info);
            if (info){
                throw std::runtime_error("DGBTRF failed.");
            }
        }
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x){
            const char trans = 'N'; // no transpose
            std::memcpy(x, b, sizeof(Real_t)*this->dim);
            int info, nrhs=1;
            dgbtrs_(&trans, &this->dim, &this->nouter, &this->nouter, &nrhs,
                    buffer_get_raw_ptr(this->data), &this->ld,
                    buffer_get_raw_ptr(this->ipiv), x, &this->dim, &info);
            return info;
        };
    };

    template <typename Real_t>
    class ColMajBlockDiagMat {
        buffer_t<Real_t> block_data, sub_data, sup_data;
    public:
        ColMajBlockDiagView<Real_t> view;
        const bool contiguous;
        buffer_ptr_t<Real_t> get_block_data_raw_ptr() {
            return buffer_get_raw_ptr(this->block_data);
        }
        buffer_ptr_t<Real_t> get_sub_data_raw_ptr() {
            return buffer_get_raw_ptr(this->sub_data);
        }
        buffer_ptr_t<Real_t> get_sup_data_raw_ptr() {
            return buffer_get_raw_ptr(this->sup_data);
        }
        ColMajBlockDiagMat(const int nblocks_,
                           const int blockw_,
                           const int ndiag_,
                           const int ld_blocks_=0,
                           const int block_stride_=0,
                           const int ld_diag_=0,
                           const bool contiguous=true) :
            view(nullptr, nullptr, nullptr, nblocks_, blockw_,
                 ndiag_, ld_blocks_, block_stride_, ld_diag_),
            contiguous(contiguous) {
            if (contiguous){
                this->block_data = buffer_factory<Real_t>(view.block_data_len +
                                                          2*view.diag_data_len);
                auto raw_ptr = this->get_block_data_raw_ptr();
                this->view.set_data_pointers(raw_ptr,
                                             raw_ptr + view.block_data_len,
                                             raw_ptr + view.block_data_len + view.diag_data_len);
            } else {
                this->block_data = buffer_factory<Real_t>(view.block_data_len);
                this->sub_data = buffer_factory<Real_t>(view.diag_data_len);
                this->sup_data = buffer_factory<Real_t>(view.diag_data_len);
                this->view.set_data_pointers(buffer_get_raw_ptr(this->block_data),
                                             buffer_get_raw_ptr(this->sub_data),
                                             buffer_get_raw_ptr(this->sup_data));
            }
        }
    };

    void rowpiv2rowbycol(int n, const int * const piv, int * const rowbycol) {
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

    void rowbycol2colbyrow(int n, const int * const rowbycol, int * const colbyrow){
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

    template<typename Real_t = double>
    class ILU_inplace {
    public:
        ColMajBlockDiagView<Real_t> view;
        buffer_t<int> ipiv, rowbycol, colbyrow;
#ifdef BLOCK_DIAG_ILU_UNIT_TEST
        int nblocks() { return this->view.nblocks; }
        int blockw() { return this->view.blockw; }
        int ndiag() { return this->view.ndiag; }
        Real_t sub_get(const int diagi, const int blocki,
                       const int coli) { return this->view.sub(diagi, blocki, coli); }
        Real_t sup_get(const int diagi, const int blocki,
                       const int coli) { return this->view.sup(diagi, blocki, coli); }
        int piv_get(const int idx) { return this->ipiv[idx]; }
        int rowbycol_get(const int idx) { return this->rowbycol[idx]; }
        int colbyrow_get(const int idx) { return this->colbyrow[idx]; }
#endif

        // use ld_blocks and ld_diag in view to avoid false sharing
        // in parallelized execution
        ILU_inplace(ColMajBlockDiagView<Real_t> view) :
            view(view),
            ipiv(buffer_factory<int>(view.blockw*view.nblocks)),
            rowbycol(buffer_factory<int>(view.blockw*view.nblocks)),
            colbyrow(buffer_factory<int>(view.blockw*view.nblocks)) {
            int info_ = 0;
            const auto nblocks = this->view.nblocks;
            const auto ndiag = this->view.ndiag;  // narrowing cast
#if !defined(BLOCK_DIAG_ILU_WITH_DGETRF)
            // LAPACK take pointers to integers
            int blockw = this->view.blockw;
            int ld_blocks = this->view.ld_blocks;
#else
            const auto blockw = this->view.blockw;
#endif
#if defined(BLOCK_DIAG_ILU_WITH_OPENMP)
            char * num_threads_var = std::getenv("BLOCK_DIAG_ILU_NUM_THREADS");
            int nt = (num_threads_var) ? std::atoi(num_threads_var) : 1;
            if (nt < 1)
                nt = 1;
            const int min_work_per_thread = 1; // could be increased
            if (nt > nblocks/min_work_per_thread)
                nt = nblocks/min_work_per_thread;

            #pragma omp parallel for num_threads(nt) schedule(static)  // OMP_NUM_THREADS should be 1 for openblas LU (small matrices)
#endif
            for (int bi=0; bi<nblocks; ++bi){
#if defined(BLOCK_DIAG_ILU_WITH_DGETRF)
                int info = getrf_square<Real_t>(
                           blockw,
                           &(this->view.block(bi, 0, 0)),
                           this->view.ld_blocks,
                           &(this->ipiv[bi*blockw]));
#else
                static_assert(sizeof(Real_t) == 8, "LAPACK dgetrf operates on 64-bit IEEE 754 floats.");
                int info;
                dgetrf_(&blockw,
                        &blockw,
                        &(this->view.block(bi, 0, 0)),
                        &(ld_blocks),
                        &(this->ipiv[bi*blockw]),
                        &info);
#endif
                if ((info != 0) && (info_ == 0))
                    info_ = info;
                for (int ci = 0; ci < (int)blockw; ++ci){
                    for (int di = 0; (di < ndiag) && (bi+di < (nblocks - 1)); ++di){
                        this->view.sub(di, bi, ci) /= this->view.block(bi, ci, ci);
                    }
                }
                rowpiv2rowbycol(blockw, &ipiv[bi*blockw], &rowbycol[bi*blockw]);
                rowbycol2colbyrow(blockw, &rowbycol[bi*blockw], &colbyrow[bi*blockw]);
            }
            if (info_)
                throw std::runtime_error("ILU failed!");
        }
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x) const {
            // before calling solve: make sure that the
            // block_data and sup_data pointers are still valid.
            // Returns
            // -------
            // if NaN in b:
            //     index (starting at 1) in b where first nan is found
            // if any diagonal element in U is zero:
            //     blockw*nblocks + diagonal index (starting at 1) in U where
            //     first 0 is found
            const auto nblocks = this->view.nblocks;
            const int blockw = this->view.blockw; // narrowing cast
            const int ndiag = this->view.ndiag; // narrowing cast
            auto y = buffer_factory<Real_t>(nblocks*blockw);
            int info = check_nan(b, nblocks*blockw);
            for (int bri = 0; bri < nblocks; ++bri){
                for (int li = 0; li < blockw; ++li){
                    Real_t s = 0.0;
                    for (int lci = 0; lci < li; ++lci){
                        s += this->view.block(bri, li, lci)*y[bri*blockw + lci];
                    }
                    const int ci = this->colbyrow[bri*blockw + li];
                    for (int di = 1; di <= std::min(ndiag, bri); ++di){
                        s += (this->view.sub(di-1, bri-di, ci) * y[(bri-di)*blockw + ci]);
                    }
                    y[bri*blockw + li] = b[bri*blockw + this->rowbycol[bri*blockw + li]] - s;
                }
            }
            for (int bri = nblocks; bri > 0; --bri){
                for (int li = blockw; li > 0; --li){
                    Real_t s = 0.0;
                    for (int ci = li; ci < blockw; ++ci){
                        s += this->view.block(bri-1, li-1, ci)*x[(bri-1)*blockw + ci];
                    }
                    const int ci = this->colbyrow[(bri-1)*blockw + li-1];
                    for (int di = 1; di <= std::min(nblocks - bri, ndiag); ++di) {
                        s += this->view.sup(di-1, bri-1, ci)*x[(bri-1+di)*blockw + ci];
                    }
                    x[(bri-1)*blockw+li-1] = (y[(bri-1)*blockw + li-1] - s)\
                        /(this->view.block(bri-1, li-1, li-1));
                    if (this->view.block(bri-1, li-1, li-1) == 0 && info == 0)
                        info = nblocks*blockw + (bri-1)*blockw + (li-1);
                }
            }
            return info;
        }
    };

    template<typename Real_t = double>
    class ILU{
        ColMajBlockDiagMat<Real_t> m_mat;
        ILU_inplace<Real_t> m_ilu_inplace;
    public:
        ILU(const ColMajBlockDiagView<Real_t>& view) :
            m_mat(view.copy_to_matrix()),
            m_ilu_inplace(ILU_inplace<Real_t>(m_mat.view)) {}
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x){
            return m_ilu_inplace.solve(b, x);
        }
    };

}

#if defined(BLOCK_DIAG_ILU_WITH_DGETRF)
// int will be enough (flops of a LU decomoposition scales as N**3, and besides this is unblocked)
template <typename Real_t = double>
int block_diag_ilu::getrf_square(const int dim, Real_t * const __restrict__ a,
                                          const int lda, int * const __restrict__ ipiv) noexcept {
    // Unblocked algorithm for LU decomposition of square matrices
    // employing Doolittle's algorithm with rowswaps.
    //
    // ipiv indexing starts at 1 (Fortran compability)
    // performance is exprect to be good when leading dimension
    // of the block fits in a L1 cache line (or is a small multiple thereof). (as of 2016 i.e. 8*float64)

    if (dim == 0) return 0;

    int info = 0;
    auto A = [&](int ri, int ci) -> Real_t& { return a[ci*lda + ri]; };
    auto swaprows = [&](int ri1, int ri2) { // this is not cache friendly
        for (int ci=0; ci<dim; ++ci){
            Real_t temp = A(ri1, ci);
            A(ri1, ci) = A(ri2, ci);
            A(ri2, ci) = temp;
        }
    };

    for (int i=0; i<dim-1; ++i) {
        int pivrow = i;
        Real_t absmax = std::abs(A(i, i));
        for (int j=i; j<dim; ++j) {
            // Find pivot
            Real_t curabs = std::abs(A(j, i));
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
