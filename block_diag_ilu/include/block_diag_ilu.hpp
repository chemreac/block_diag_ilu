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
        const int m_blockw, m_ndiag;
        const int m_nblocks, m_nouter, m_dim;
        ViewBase(int blockw, int ndiag, int nblocks)
            : m_blockw(blockw),
              m_ndiag(ndiag),
              m_nblocks(nblocks),
              m_nouter(nouter_(blockw, ndiag)),
              m_dim(blockw*nblocks)
        {}

        Real_t get_global(const int rowi, const int coli) const noexcept{
            const auto& self = *static_cast<const T*>(this);  // CRTP
            const int bri = rowi / self.m_blockw;
            const int bci = coli / self.m_blockw;
            const int lri = rowi - bri*self.m_blockw;
            const int lci = coli - bci*self.m_blockw;
            if (bri == bci)
                return self.block(bri, lri, lci);
            if (lri != lci)
                return 0.0;
            if (bri > bci){ // sub diagonal
                if ((bri - bci) > m_ndiag)
                    return 0.0;
                else
                    return self.sub(bri-bci-1, bci, lci);
            } else { // super diagonal
                if ((bci - bri) > m_ndiag)
                    return 0.0;
                else
                    return self.sup(bci-bri-1, bri, lri);
            }
        }
        int get_banded_ld() const noexcept {
            return banded_ld_(static_cast<const T*>(this)->m_nouter);  // CRTP
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
        Real_t *m_data;
        const int m_ld;

        DenseView(Real_t *data, const int nblocks, const int blockw, const int ndiag, const int ld_=0)
            : ViewBase<DenseView<Real_t, col_maj>, Real_t>(blockw, ndiag, nblocks),
            m_data(data),
            m_ld((ld_ == 0) ? blockw*nblocks : ld_)
        {}
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            const int imaj = this->m_blockw*blocki + (col_maj ? coli : rowi);
            const int imin = this->m_blockw*blocki + (col_maj ? rowi : coli);
            return m_data[imaj*m_ld + imin];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int li) const noexcept {
            const int imaj = (this->m_blockw)*(blocki + (col_maj ? 0 : diagi + 1)) + li;
            const int imin = (this->m_blockw)*(blocki + (col_maj ? diagi + 1 : 0)) + li;
            return m_data[imaj*m_ld + imin];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int li) const noexcept {
            const int imaj = (this->m_blockw)*(blocki + (col_maj ? diagi + 1 : 0)) + li;
            const int imin = (this->m_blockw)*(blocki + (col_maj ? 0 : diagi + 1)) + li;
            return m_data[imaj*m_ld + imin];
        }
    };

    template <typename Real_t = double>
    class ColMajBandedView : public ViewBase<ColMajBandedView<Real_t>, Real_t> {
        // For use with LAPACK's banded matrix layout.
        // Note that the matrix is padded with ``mlower`` extra bands.
    public:
        Real_t *m_data;
        const int m_ld, m_offset;

        ColMajBandedView(Real_t *data, const int nblocks, const int blockw, const int ndiag,
                         const int ld=0, int offset=-1)
            : ViewBase<ColMajBandedView<Real_t>, Real_t>(blockw, ndiag, nblocks),
            m_data(data),
            m_ld((ld == 0) ? banded_ld_(nouter_(blockw, ndiag), offset) : ld),
            m_offset((offset == -1) ? nouter_(blockw, ndiag) : offset)
        {}
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            const int imaj = blocki*(this->m_blockw) + coli;
            const int imin = m_offset + (this->m_nouter) + rowi - coli;
            return m_data[imaj*m_ld + imin];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int coli) const noexcept {
            const int imaj = blocki*(this->m_blockw) + coli;
            const int imin = m_offset + (this->m_nouter) + (diagi + 1)*(this->m_blockw);
            return m_data[imaj*m_ld + imin];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int coli) const noexcept {
            const int imaj = (blocki + diagi + 1)*(this->m_blockw) + coli;
            const int imin = m_offset + (this->m_nouter) - (diagi+1)*(this->m_blockw);
            return m_data[imaj*m_ld + imin];
        }
    };

    template <typename Real_t = double> class ColMajBlockDiagMat;
    template <typename Real_t = double>
    class ColMajBlockDiagView : public ViewBase<ColMajBlockDiagView<Real_t>, Real_t> {
    public:
        Real_t *m_block_data, *m_sub_data, *m_sup_data;
        // int will suffice, decomposition scales as N**3 even iterative methods (N**2) would need months at 1 TFLOPS
        const int m_ld_blocks;
        const int m_block_stride;
        const int m_ld_diag;
        const int m_block_data_len, m_diag_data_len;
        // ld_block for cache alignment and avoiding false sharing
        // block_stride for avoiding false sharing
        ColMajBlockDiagView(Real_t * const block_data,
                            Real_t * const sub_data,
                            Real_t * const sup_data,
                            const int nblocks,
                            const int blockw,
                            const int ndiag,
                            const int ld_blocks=0,
                            const int block_stride=0,
                            const int ld_diag=0) :
            ViewBase<ColMajBlockDiagView<Real_t>, Real_t>(blockw, ndiag, nblocks),
            m_block_data(block_data),
            m_sub_data(sub_data),
            m_sup_data(sup_data),
            m_ld_blocks((ld_blocks == 0) ? blockw : ld_blocks),
            m_block_stride((block_stride == 0) ? m_ld_blocks*blockw : block_stride),
            m_ld_diag((ld_diag == 0) ? m_ld_blocks : ld_diag),
            m_block_data_len(nblocks*m_block_stride),
            m_diag_data_len(m_ld_diag*(nblocks*ndiag - (ndiag*ndiag + ndiag)/2))
            {}
        void scale_diag_add(const ColMajBlockDiagView<Real_t>& source, Real_t scale=1, Real_t diag_add=0){
            const auto nblocks = (this->m_nblocks);
            const auto blockw = (this->m_blockw);
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
            for (int di = 0; di < (this->m_ndiag); ++di) {
                for (int bi=0; bi < ((nblocks <= di+1) ? 0 : nblocks - di - 1); ++bi) {
                    for (int ci = 0; ci < blockw; ++ci){
                        this->sub(di, bi, ci) = scale * source.sub(di, bi, ci);
                        this->sup(di, bi, ci) = scale * source.sup(di, bi, ci);
                    }
                }
            }
        }
        ColMajBlockDiagMat<Real_t> copy_to_matrix() const {
            auto mat = ColMajBlockDiagMat<Real_t> {this->m_nblocks, this->m_blockw, this->m_ndiag, m_ld_blocks, m_block_stride, m_ld_diag};
            mat.m_view.scale_diag_add(*this);
            return mat;
        }
        void set_data_pointers(buffer_ptr_t<Real_t> block_data,
                               buffer_ptr_t<Real_t> sub_data,
                               buffer_ptr_t<Real_t> sup_data) noexcept {
            m_block_data = block_data;
            m_sub_data = sub_data;
            m_sup_data = sup_data;
        }
        void dot_vec(const Real_t * const vec, Real_t * const out){
            // out need not be zeroed out before call
            const auto nblk = this->m_nblocks;
            const auto blkw = this->m_blockw;
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
            for (int di=0; di<(this->m_ndiag); ++di){
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
            const auto nblk = this->m_nblocks;
            const auto blkw = this->m_blockw;
            if (diag_idx == 0){
                nelem = nblk*blkw;
                for (int bi = 0; bi < nblk; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->block(bi, ci, ci);
                        sum += elem*elem;
                    }
                }
            } else if (diag_idx < 0) {
                if (-diag_idx >= nblk)
                    return 0;
                nelem = (nblk + diag_idx)*blkw;
                for (int bi = 0; bi < nblk+diag_idx ; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->sub(-diag_idx - 1, bi, ci);
                        sum += elem*elem;
                    }
                }
            } else {
                if (diag_idx >= nblk)
                    return 0;
                nelem = (nblk - diag_idx)*blkw;
                for (int bi = 0; bi < nblk-diag_idx ; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->sup(diag_idx - 1, bi, ci);
                        sum += elem*elem;
                    }
                }
            }
            return std::sqrt(sum/nelem);
        }
        Real_t average_diag_weight(int di){  // di >= 0
            Real_t off_diag_factor = 0;
            const auto nblk = this->m_nblocks;
            const auto blkw = this->m_blockw;
            for (int bi = 0; bi < nblk; ++bi){
                for (int li = 0; li < blkw; ++li){
                    const Real_t diag_val = this->block(bi, li, li);
                    if (bi < nblk - di - 1){
                        off_diag_factor += std::abs(diag_val/this->sub(di, bi, li));
                    }
                    if (bi > di){
                        off_diag_factor += std::abs(diag_val/this->sup(di, bi - di - 1, li));
                    }
                }
            }
            return off_diag_factor / (blkw * (nblk - 1 - di) * 2);
        }

    private:
        int diag_idx(const int diagi, const int blocki,
                                    const int coli) const noexcept {
            const int n_diag_blocks_skip = (this->m_nblocks)*diagi - (diagi*diagi + diagi)/2;
            return (n_diag_blocks_skip + blocki)*(m_ld_diag) + coli;
        }
    public:
        Real_t& block(const int blocki, const int rowi,
                             const int coli) const noexcept {
            return m_block_data[blocki*m_block_stride + coli*(m_ld_blocks) + rowi];
        }
        Real_t& sub(const int diagi, const int blocki,
                           const int coli) const noexcept {
            return m_sub_data[diag_idx(diagi, blocki, coli)];
        }
        Real_t& sup(const int diagi, const int blocki,
                           const int coli) const noexcept {
            return m_sup_data[diag_idx(diagi, blocki, coli)];
        }
        buffer_t<Real_t> to_banded() const {
            const auto ld_result = this->get_banded_ld();
            const auto ntr = this->m_nouter;
            const auto dm = this->m_dim;
            auto result = buffer_factory<Real_t>(ld_result*dm);
            for (int ci = 0; ci < dm; ++ci){
                const int row_lower = (ci < ntr) ? 0 : ci - ntr;
                const int row_upper = (ci + ntr + 1 > dm) ? dm : ci + ntr + 1;
                for (int ri=row_lower; ri<row_upper; ++ri){
                    result[ld_result*ci + 2*ntr + ri - ci] = this->get_global(ri, ci);
                }
            }
            return result;
        }
        void set_to_1_minus_gamma_times_view(Real_t gamma, ColMajBlockDiagView &other) {
            scale_diag_add(other, -gamma, 1);
        }
        void zero_out_blocks() noexcept {
            for (int i=0; i<m_block_data_len; ++i){
                m_block_data[i] = 0.0;
            }
        }
        void zero_out_diags() noexcept {
            for (int i=0; i<(m_diag_data_len); ++i){
                m_sub_data[i] = 0.0;
            }
            for (int i=0; i<(m_diag_data_len); ++i){
                m_sup_data[i] = 0.0;
            }
        }
    };

    template <typename Real_t = double>
    class LU {  // Wrapper around DGBTRF & DGBTRS from LAPACK
        static_assert(sizeof(Real_t) == 8, "LAPACK DGBTRF & DGBTRS operates on 64-bit IEEE 754 floats.");
#ifdef BLOCK_DIAG_ILU_UNIT_TEST
    public:
#endif
        const int m_dim, m_nouter, m_ld;
        buffer_t<Real_t> m_data;
        buffer_t<int> m_ipiv;
    public:
        LU(const ColMajBlockDiagView<Real_t>& view) :
            m_dim(view.m_dim),
            m_nouter(view.m_nouter),
            m_ld(view.get_banded_ld()),
            m_data(view.to_banded()),
            m_ipiv(buffer_factory<int>(view.m_dim))
        {
            factorize();
        }
        LU(const ColMajBandedView<Real_t>& view) :
            m_dim(view.m_dim),
            m_nouter(view.m_nouter),
            m_ld(view.m_ld),
            m_data(buffer_factory<Real_t>(view.m_ld*view.m_dim)),
            m_ipiv(buffer_factory<int>(view.m_dim))
        {
            std::memcpy(&m_data[0], this->m_view.m_data, sizeof(Real_t) * m_ld * m_dim);
            if (m_ld != banded_ld_(view.m_nouter)){
                throw std::runtime_error("LAPACK requires padding");
            }
            factorize();
        }
        void factorize(){
            int info;
            dgbtrf_(&m_dim, &m_dim, &m_nouter, &m_nouter,
                    buffer_get_raw_ptr(m_data),
                    &m_ld,
                    buffer_get_raw_ptr(m_ipiv), &info);
            if (info){
                throw std::runtime_error("DGBTRF failed.");
            }
        }
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x){
            const char trans = 'N'; // no transpose
            std::memcpy(x, b, sizeof(Real_t)*m_dim);
            int info, nrhs=1;
            dgbtrs_(&trans, &m_dim, &m_nouter, &m_nouter, &nrhs,
                    buffer_get_raw_ptr(m_data), &m_ld,
                    buffer_get_raw_ptr(m_ipiv), x, &m_dim, &info);
            return info;
        };
    };

    template <typename Real_t>
    class ColMajBlockDiagMat {
        buffer_t<Real_t> m_block_data, m_sub_data, m_sup_data;
    public:
        ColMajBlockDiagView<Real_t> m_view;
        const bool m_contiguous;
        buffer_ptr_t<Real_t> get_block_data_raw_ptr() {
            return buffer_get_raw_ptr(m_block_data);
        }
        buffer_ptr_t<Real_t> get_sub_data_raw_ptr() {
            return buffer_get_raw_ptr(m_sub_data);
        }
        buffer_ptr_t<Real_t> get_sup_data_raw_ptr() {
            return buffer_get_raw_ptr(m_sup_data);
        }
        ColMajBlockDiagMat(const int nblocks,
                           const int blockw,
                           const int ndiag,
                           const int ld_blocks=0,
                           const int block_stride=0,
                           const int ld_diag=0,
                           const bool contiguous=true) :
            m_view(nullptr, nullptr, nullptr, nblocks, blockw,
                   ndiag, ld_blocks, block_stride, ld_diag),
            m_contiguous(contiguous)
        {
            if (m_contiguous){
                m_block_data = buffer_factory<Real_t>(m_view.m_block_data_len +
                                                      2*m_view.m_diag_data_len);
                auto raw_ptr = this->get_block_data_raw_ptr();
                m_view.set_data_pointers(raw_ptr,
                                         raw_ptr + m_view.m_block_data_len,
                                         raw_ptr + m_view.m_block_data_len + m_view.m_diag_data_len);
            } else {
                m_block_data = buffer_factory<Real_t>(m_view.m_block_data_len);
                m_sub_data = buffer_factory<Real_t>(m_view.m_diag_data_len);
                m_sup_data = buffer_factory<Real_t>(m_view.m_diag_data_len);
                m_view.set_data_pointers(buffer_get_raw_ptr(m_block_data),
                                         buffer_get_raw_ptr(m_sub_data),
                                         buffer_get_raw_ptr(m_sup_data));
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
        ColMajBlockDiagView<Real_t> m_view;
        buffer_t<int> m_ipiv, m_rowbycol, m_colbyrow;
#ifdef BLOCK_DIAG_ILU_UNIT_TEST
        int nblocks() { return m_view.m_nblocks; }
        int blockw() { return m_view.m_blockw; }
        int ndiag() { return m_view.m_ndiag; }
        Real_t sub_get(const int diagi, const int blocki,
                       const int coli) { return m_view.sub(diagi, blocki, coli); }
        Real_t sup_get(const int diagi, const int blocki,
                       const int coli) { return m_view.sup(diagi, blocki, coli); }
        int piv_get(const int idx) { return m_ipiv[idx]; }
        int rowbycol_get(const int idx) { return m_rowbycol[idx]; }
        int colbyrow_get(const int idx) { return m_colbyrow[idx]; }
#endif

        // use ld_blocks and ld_diag in view to avoid false sharing
        // in parallelized execution
        ILU_inplace(ColMajBlockDiagView<Real_t> view) :
            m_view(view),
            m_ipiv(buffer_factory<int>(view.m_blockw*view.m_nblocks)),
            m_rowbycol(buffer_factory<int>(view.m_blockw*view.m_nblocks)),
            m_colbyrow(buffer_factory<int>(view.m_blockw*view.m_nblocks)) {
            int info_ = 0;
            const auto nblocks = m_view.m_nblocks;
            const auto ndiag = m_view.m_ndiag;  // narrowing cast
            const auto blockw = m_view.m_blockw;
            auto ld_blocks = m_view.m_ld_blocks;
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
                           &(m_view.block(bi, 0, 0)),
                           ld_blocks,
                           &(m_ipiv[bi*blockw]));
#else
                static_assert(sizeof(Real_t) == 8, "LAPACK dgetrf operates on 64-bit IEEE 754 floats.");
                int info;
                dgetrf_(&blockw,
                        &blockw,
                        &(m_view.block(bi, 0, 0)),
                        &(ld_blocks),
                        &(m_ipiv[bi*blockw]),
                        &info);
#endif
                if ((info != 0) && (info_ == 0))
                    info_ = info;
                for (int ci = 0; ci < (int)blockw; ++ci){
                    for (int di = 0; (di < ndiag) && (bi+di < (nblocks - 1)); ++di){
                        m_view.sub(di, bi, ci) /= m_view.block(bi, ci, ci);
                    }
                }
                rowpiv2rowbycol(blockw, &m_ipiv[bi*blockw], &m_rowbycol[bi*blockw]);
                rowbycol2colbyrow(blockw, &m_rowbycol[bi*blockw], &m_colbyrow[bi*blockw]);
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
            const auto nblocks = m_view.m_nblocks;
            const int blockw = m_view.m_blockw;
            const int ndiag = m_view.m_ndiag;
            auto y = buffer_factory<Real_t>(nblocks*blockw);
            int info = check_nan(b, nblocks*blockw);
            for (int bri = 0; bri < nblocks; ++bri){
                for (int li = 0; li < blockw; ++li){
                    Real_t s = 0.0;
                    for (int lci = 0; lci < li; ++lci){
                        s += m_view.block(bri, li, lci)*y[bri*blockw + lci];
                    }
                    const int ci = m_colbyrow[bri*blockw + li];
                    for (int di = 1; di <= std::min(ndiag, bri); ++di){
                        s += (m_view.sub(di-1, bri-di, ci) * y[(bri-di)*blockw + ci]);
                    }
                    y[bri*blockw + li] = b[bri*blockw + m_rowbycol[bri*blockw + li]] - s;
                }
            }
            for (int bri = nblocks; bri > 0; --bri){
                for (int li = blockw; li > 0; --li){
                    Real_t s = 0.0;
                    for (int ci = li; ci < blockw; ++ci){
                        s += m_view.block(bri-1, li-1, ci)*x[(bri-1)*blockw + ci];
                    }
                    const int ci = m_colbyrow[(bri-1)*blockw + li-1];
                    for (int di = 1; di <= std::min(nblocks - bri, ndiag); ++di) {
                        s += m_view.sup(di-1, bri-1, ci)*x[(bri-1+di)*blockw + ci];
                    }
                    x[(bri-1)*blockw+li-1] = (y[(bri-1)*blockw + li-1] - s)\
                        /(m_view.block(bri-1, li-1, li-1));
                    if (m_view.block(bri-1, li-1, li-1) == 0 && info == 0)
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
            m_ilu_inplace(ILU_inplace<Real_t>(m_mat.m_view)) {}
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
