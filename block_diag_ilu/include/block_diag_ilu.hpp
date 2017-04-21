#pragma once
#include <algorithm> // std::max
#include <type_traits>
#include <utility>
#include <cmath> // std::abs for float and double, std::sqrt, std::isnan
#include <cstdlib> // std::abs for int (must include!!)

#include <anyode/anyode_blas_lapack.hpp>
#include <anyode/anyode_buffer.hpp>
#include <anyode/anyode_matrix.hpp>

#if defined(BLOCK_DIAG_ILU_WITH_GETRF)
#include <anyode/anyode_blas_lapack.hpp>
#endif

namespace block_diag_ilu {
    using AnyODE::buffer_t;
    using AnyODE::buffer_ptr_t;
    using AnyODE::buffer_factory;
    using AnyODE::buffer_get_raw_ptr;
    using AnyODE::MatrixView;
    using AnyODE::DenseMatrixView;
    using AnyODE::BandedPaddedMatrixView;

#if defined(BLOCK_DIAG_ILU_WITH_GETRF)
    template<typename T> int getrf_square(const int dim, T * const __restrict__ a,
                                          const int lda, int * const __restrict__ ipiv) noexcept;
#endif

    constexpr int nouter_(int blockw, int ndiag) { return (ndiag == 0) ? blockw-1 : blockw*ndiag; }

    void rowpiv2rowbycol(int n, const int * const piv, int * const rowbycol) {
        for (int i = 0; i < n; ++i)
            rowbycol[i] = i;
        for (int i=0; i<n; ++i){
            int j = piv[i] - 1; // Fortran indexing starts with 1
            std::swap(rowbycol[i], rowbycol[j]);
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

#define BLOCK(bi, ri, ci, blkw) blkw*bi + ri, blkw*bi + ci
#define SUB(di, bi, ci, blkw) blkw*(bi+1+di) + ci, blkw*bi + ci
#define SUP(di, bi, ci, blkw) blkw*bi + ci, blkw*(bi+1+di) + ci
#define BOT(si, bi, ci, blkw, nblk) blkw*(nblk + bi - si - 1) + ci, blkw*bi + ci
#define TOP(si, bi, ci, blkw, nblk) blkw*bi + ci, blkw*(nblk + bi - si - 1) + ci
#define DIM nblocks*blockw

#define GET_(ri, ci) this->m_data[(ci)*(this->m_ld) + ri]
#define GET(...) GET_(__VA_ARGS__)
    template <typename Real_t = double>
    struct BlockDenseView : public DenseMatrixView<Real_t> {
        const int m_nblocks, m_blockw, m_ndiag;
        BlockDenseView(Real_t * const data, const int nblocks, const int blockw, const int ndiag) :
            DenseMatrixView<Real_t>(data, DIM, DIM, DIM),
            m_nblocks(nblocks), m_blockw(blockw), m_ndiag(ndiag)
        {}
        Real_t& block(const int bi, const int ri, const int ci) const noexcept { return GET(BLOCK(bi, ri, ci, m_blockw)); }
        Real_t& sub(const int di, const int bi, const int ci) const noexcept { return GET(SUB(di, bi, ci, m_blockw)); }
        Real_t& sup(const int di, const int bi, const int ci) const noexcept { return GET(SUP(di, bi, ci, m_blockw)); }
        Real_t& bot(const int si, const int bi, const int ci) const noexcept { return GET(BOT(si, bi, ci, m_blockw, m_nblocks)); }
        Real_t& top(const int si, const int bi, const int ci) const noexcept { return GET(TOP(si, bi, ci, m_blockw, m_nblocks)); }
    };
#undef GET
#undef GET_

#define NOUTER nouter_(blockw, ndiag)
#define GET_(ri, ci) this->m_data[this->m_kl + this->m_ku + ri + (ci)*(this->m_ld-1)]
#define GET(...) GET_(__VA_ARGS__)

    template <typename Real_t = double>
    struct BlockBandedView : public BandedPaddedMatrixView<Real_t> {
        const int m_nblocks, m_blockw, m_ndiag;
        BlockBandedView(Real_t * const data, const int nblocks, const int blockw, const int ndiag) :
            BandedPaddedMatrixView<Real_t>(data, DIM, DIM, NOUTER, NOUTER),
            m_nblocks(nblocks), m_blockw(blockw), m_ndiag(ndiag)
        {}
        Real_t& block(const int bi, const int ri, const int ci) const noexcept { return GET(BLOCK(bi, ri, ci, m_blockw)); }
        Real_t& sub(const int di, const int bi, const int ci) const noexcept { return GET(SUB(di, bi, ci, m_blockw)); }
        Real_t& sup(const int di, const int bi, const int ci) const noexcept { return GET(SUP(di, bi, ci, m_blockw)); }
    };
#undef GET
#undef GET_
#undef NOUTER

#define NITEMS MatrixView<Real_t>::alignment_items_
#define LD ((ld == 0) ? NITEMS*((blockw + NITEMS - 1)/NITEMS) : ld)
#define BLK_NDATA nblocks*blockw*LD
#define DIAG_HLF_NDATA (ndiag*nblocks - (ndiag*ndiag + ndiag)/2)*LD
#define SAT_HLF_NDATA (nsat*nsat+nsat)/2*LD
#define TOT_NDATA (BLK_NDATA + 2*(DIAG_HLF_NDATA + SAT_HLF_NDATA))

    template <typename Real_t = double>
    struct ColMajBlockDiagMatrixView : public MatrixView<Real_t> {
        const int m_nblocks, m_blockw, m_ndiag, m_nsat;
    private:
        const int m_blk_ndata, m_diag_hlf_ndata, m_sat_hlf_ndata;
    public:
        ColMajBlockDiagMatrixView(Real_t * const data,
                                  const int nblocks,
                                  const int blockw,
                                  const int ndiag=0,
                                  const int nsat=0,
                                  const int ld=0) :  // 64 byte alignment
            MatrixView<Real_t>(data, DIM, DIM, LD, TOT_NDATA),
            m_nblocks(nblocks), m_blockw(blockw), m_ndiag(ndiag), m_nsat(nsat),
            m_blk_ndata(BLK_NDATA), m_diag_hlf_ndata(DIAG_HLF_NDATA), m_sat_hlf_ndata(SAT_HLF_NDATA)
            {
                if (data != nullptr and ld == 0){
                    throw std::runtime_error("give ld when providing data pointer.");
                }
            }
        ColMajBlockDiagMatrixView(const MatrixView<Real_t>& source,
                                  const int nblocks,
                                  const int blockw,
                                  const int ndiag=0,
                                  const int nsat=0,
                                  const int ld=0) :
            MatrixView<Real_t>(nullptr, DIM, DIM, LD, TOT_NDATA),
            m_nblocks(nblocks), m_blockw(blockw), m_ndiag(ndiag), m_nsat(nsat),
            m_blk_ndata(BLK_NDATA), m_diag_hlf_ndata(DIAG_HLF_NDATA), m_sat_hlf_ndata(SAT_HLF_NDATA)
        {
            // apply_over_indices([&](const int ri, const int ci) { valid_index(ri, ci) ? (*this)(ri, ci) = source(ri, ci) : 0; });
            for (int bi = 0; bi < nblocks; ++bi)
                for (int ci=0; ci < blockw; ++ci)
                    for (int ri = 0; ri < blockw; ++ri)
                        this->block(bi, ri, ci) = source(BLOCK(bi, ri, ci, m_blockw));

            for (int di = 0; di < (this->m_ndiag); ++di)
                for (int bi=0; bi < ((nblocks <= di+1) ? 0 : nblocks - di - 1); ++bi)
                    for (int ci = 0; ci < blockw; ++ci)
                        this->sub(di, bi, ci) = source(SUB(di, bi, ci, m_blockw));

            for (int di = 0; di < (this->m_ndiag); ++di)
                for (int bi=0; bi < ((nblocks <= di+1) ? 0 : nblocks - di - 1); ++bi)
                    for (int ci = 0; ci < blockw; ++ci)
                        this->sup(di, bi, ci) = source(SUP(di, bi, ci, m_blockw));

            for (int sati=0; sati < m_nsat; ++sati)
                for (int bi=0; bi <= sati; ++bi)
                    for (int ci = 0; ci < blockw; ++ci)
                        this->bot(sati, bi, ci) = source(BOT(sati, bi, ci, m_blockw, m_nblocks));

            for (int sati=0; sati < m_nsat; ++sati)
                for (int bi=0; bi <= sati; ++bi)
                    for (int ci = 0; ci < blockw; ++ci)
                        this->top(sati, bi, ci) = source(TOP(sati, bi, ci, m_blockw, m_nblocks));
        }
#undef TOT_NDATA
#undef SAT_HLF_DATA
#undef DIAG_HLF_NDATA
#undef BLK_NDATA
#undef LD
#undef NITEMS
        template<typename F>
        void apply_over_indices(F cb) {
            for (int bi = 0; bi < m_nblocks; ++bi)
                for (int ci=0; ci < m_blockw; ++ci)
                    for (int ri = 0; ri < m_blockw; ++ri)
                        cb(BLOCK(bi, ri, ci, m_blockw));
            for (int di = 0; di < (this->m_ndiag); ++di)
                for (int bi=0; bi < ((m_nblocks <= di+1) ? 0 : m_nblocks - di - 1); ++bi)
                    for (int ci = 0; ci < m_blockw; ++ci)
                        cb(SUB(di, bi, ci, m_blockw));

            for (int di = 0; di < (this->m_ndiag); ++di)
                for (int bi=0; bi < ((m_nblocks <= di+1) ? 0 : m_nblocks - di - 1); ++bi)
                    for (int ci = 0; ci < m_blockw; ++ci)
                        cb(SUP(di, bi, ci, m_blockw));

            for (int sati=0; sati < m_nsat; ++sati)
                for (int bi=0; bi <= sati; ++bi)
                    for (int ci = 0; ci < m_blockw; ++ci)
                        cb(BOT(sati, bi, ci, m_blockw, m_nblocks));

            for (int sati=0; sati < m_nsat; ++sati)
                for (int bi=0; bi <= sati; ++bi)
                    for (int ci = 0; ci < m_blockw; ++ci)
                        cb(TOP(sati, bi, ci, m_blockw, m_nblocks));
        }
#undef DIM
#undef TOP
#undef BOT
#undef SUP
#undef SUB
#undef BLOCK

        ColMajBlockDiagMatrixView(const ColMajBlockDiagMatrixView<Real_t>& ori) :
            MatrixView<Real_t>(ori), m_nblocks(ori.m_nblocks), m_blockw(ori.m_blockw), m_ndiag(ori.m_ndiag), m_nsat(ori.m_nsat),
            m_blk_ndata(ori.m_blk_ndata), m_diag_hlf_ndata(ori.m_diag_hlf_ndata), m_sat_hlf_ndata(ori.m_sat_hlf_ndata)
        {}
        Real_t& block(const int blocki, const int rowi, const int coli) const noexcept {
            return this->m_data[(blocki*m_blockw + coli)*(this->m_ld) + rowi];
        }
#define SKIPIDXDIAG (diagi*m_nblocks - (diagi*diagi + diagi)/2 + blocki)*(this->m_ld) + coli
        Real_t& sub(const int diagi, const int blocki, const int coli) const noexcept {
            return this->m_data[SKIPIDXDIAG + m_blk_ndata];
        }
        Real_t& sup(const int diagi, const int blocki, const int coli) const noexcept {
            return this->m_data[SKIPIDXDIAG + m_blk_ndata + m_diag_hlf_ndata];
        }
#undef SKIPIDXDIAG
#define SKIPIDXSAT ((sati*sati + sati)/2 + blocki)*(this->m_ld) + coli
        Real_t& bot(const int sati, const int blocki, const int coli) const noexcept {
            return this->m_data[SKIPIDXSAT + m_blk_ndata + 2*m_diag_hlf_ndata];
        }
        Real_t& top(const int sati, const int blocki, const int coli) const noexcept {
            return this->m_data[SKIPIDXSAT + m_blk_ndata + 2*m_diag_hlf_ndata + m_sat_hlf_ndata];
        }
#undef SKIPIDXSAT
        Real_t& operator()(const int rowi, const int coli) override final {
            const int bri = rowi / this->m_blockw;
            const int bci = coli / this->m_blockw;
            const int lri = rowi - bri*this->m_blockw;
            const int lci = coli - bci*this->m_blockw;
            if (bri == bci)
                return this->block(bri, lri, lci);
            if (lri != lci)
                throw std::runtime_error("Illegal index");
            if (bri > bci){ // sub diagonal
                if (bri - bci > m_ndiag){
                    if (this->m_nblocks - bri + bci <= this->m_nsat)
                        return this->bot(this->m_nblocks - 1 - bri + bci, bci, lci);
                    else
                        throw std::runtime_error("Illegal index");
                } else{
                    return this->sub(bri-bci-1, bci, lci);
                }
            } else { // super diagonal
                if (bci - bri > m_ndiag){
                    if (this->m_nblocks - bci + bri <= this->m_nsat)
                        return this->top(this->m_nblocks - bci + bri - 1, bri, lri);
                    throw std::runtime_error("Illegal index");
                } else {
                    return this->sup(bci-bri-1, bri, lri);
                }
            }
        }
        virtual bool guaranteed_zero_index(const int ri, const int ci) const override {
            try {
                (*const_cast<ColMajBlockDiagMatrixView<Real_t>*>(this))(ri, ci);
            } catch (...){
                return true;
            }
            return false;
        }
        bool valid_index(const int ri, const int ci) {

            try {
                (*this)(ri, ci);
            } catch (...) {
                return false;
            }
            return true;
        }
        void scale_diag_add(const ColMajBlockDiagMatrixView<Real_t>& source, Real_t scale=1, Real_t diag_add=0){
            const auto nblocks = (this->m_nblocks);
            const auto blockw = (this->m_blockw);
            for (int i=0; i < this->m_ndata; ++i){
                this->m_data[i] = scale*source.m_data[i];
            }
            for (int bi = 0; bi < nblocks; ++bi){
                for (int ci = 0; ci < blockw; ++ci){
                    this->block(bi, ci, ci) += diag_add;
                }
            }
        }
        void set_to_eye_plus_scaled_mtx(Real_t scale, const ColMajBlockDiagMatrixView<Real_t>& source) {
            scale_diag_add(source, scale, 1);
        }
        void set_to_eye_plus_scaled_mtx(Real_t scale, const MatrixView<Real_t>& source) override final {
            ColMajBlockDiagMatrixView<Real_t> cpy(source, m_nblocks, m_blockw, m_ndiag, m_nsat, this->m_ld);
            scale_diag_add(cpy, scale, 1);
        }
        void dot_vec(const Real_t * const vec, Real_t * const out) override final {
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
            for (int sati=0; sati<m_nsat; ++sati){
                for (int bi=0; bi<=sati; ++bi){
                    for (int ci=0; ci<blkw; ++ci){
                        out[bi*blkw + ci] += this->top(sati, bi, ci)*vec[(nblk-1-sati+bi)*blkw + ci];
                        out[(nblk-1-sati+bi)*blkw + ci] += this->bot(sati, bi, ci)*vec[bi*blkw + ci];
                    }
                }
            }
        }
        Real_t rms_diag(int idx_diag) {
            // returns the root mean square of `idx_diag`:th diagonal
            // (idx_diag < 0 denotes sub diagonals, idx_diag == 0 deontes main diagonal,
            // and idx_diag > 0 denotes super diagonals)
            Real_t sum = 0;
            int nelem;
            const auto nblk = this->m_nblocks;
            const auto blkw = this->m_blockw;
            if (idx_diag == 0){
                nelem = nblk*blkw;
                for (int bi = 0; bi < nblk; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->block(bi, ci, ci);
                        sum += elem*elem;
                    }
                }
            } else if (idx_diag < 0) {
                if (-idx_diag >= nblk)
                    return 0;
                nelem = (nblk + idx_diag)*blkw;
                for (int bi = 0; bi < nblk+idx_diag ; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->sub(-idx_diag - 1, bi, ci);
                        sum += elem*elem;
                    }
                }
            } else {
                if (idx_diag >= nblk)
                    return 0;
                nelem = (nblk - idx_diag)*blkw;
                for (int bi = 0; bi < nblk-idx_diag ; ++bi){
                    for (int ci = 0; ci < blkw; ++ci){
                        const Real_t elem = this->sup(idx_diag - 1, bi, ci);
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
        DenseMatrixView<Real_t> as_dense() const {
            const int dim = m_blockw*m_nblocks;
            return DenseMatrixView<Real_t>(*this, dim, dim, dim);
        }
        BandedPaddedMatrixView<Real_t> as_banded_padded(const int ld=0) const {
            const int nouter = nouter_(m_blockw, m_ndiag);
            return BandedPaddedMatrixView<Real_t>(*this, nouter, nouter, ld);
        }
        BlockBandedView<Real_t> as_block_banded() const {
            auto bbv = BlockBandedView<Real_t>(nullptr, m_nblocks, m_blockw, m_ndiag);
            bbv.read(*this);
            return bbv;
        }

#if defined(BLOCK_DIAG_ILU_PY)
        // Cython work around: https://groups.google.com/forum/#!topic/cython-users/j58Sp3QMrD4
        void set_block(const int blocki, const int rowi,
                       const int coli, Real_t value) const noexcept {
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
        void set_sat(const int sati, const int blocki,
                     const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.sat(sati, blocki, coli) = value;
        }
        void set_bot(const int sati, const int blocki,
                     const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.bot(sati, blocki, coli) = value;
        }
        void set_top(const int sati, const int blocki,
                     const int coli, Real_t value) const noexcept {
            const auto& self = *static_cast<const T*>(this);  // CRTP
            self.top(sati, blocki, coli) = value;
        }
#endif
    };

    template<typename Real_t = double>
    struct ILU_inplace {
        ColMajBlockDiagMatrixView<Real_t> m_view;
        buffer_t<int> m_ipiv, m_rowbycol, m_colbyrow;
        ILU_inplace(ColMajBlockDiagMatrixView<Real_t> view) :
            m_view(view),
            m_ipiv(buffer_factory<int>(view.m_blockw*view.m_nblocks)),
            m_rowbycol(buffer_factory<int>(view.m_blockw*view.m_nblocks)),
            m_colbyrow(buffer_factory<int>(view.m_blockw*view.m_nblocks))
        {
            int info_ = 0;
            const int nblocks = m_view.m_nblocks;
            const int ndiag = m_view.m_ndiag;
            const int blockw = m_view.m_blockw;
            auto ld = m_view.m_ld;
#if defined(BLOCK_DIAG_ILU_WITH_OPENMP)
            char * num_threads_var = std::getenv("BLOCK_DIAG_ILU_NUM_THREADS");
            int nt = (num_threads_var) ? std::atoi(num_threads_var) : 1;
            if (nt < 1)
                nt = 1;
            const int min_work_per_thread = 1; // could be increased
            if (nt > nblocks/min_work_per_thread)
                nt = 1+nblocks/(min_work_per_thread+1);

            #pragma omp parallel for num_threads(nt) schedule(static)  // OMP_NUM_THREADS should be 1 for openblas LU (small matrices)
#endif
            for (int bi=0; bi<nblocks; ++bi){
#if defined(BLOCK_DIAG_ILU_WITH_GETRF)
                int info = getrf_square<Real_t>(
                           blockw,
                           &(m_view.block(bi, 0, 0)),
                           ld,
                           &(m_ipiv[bi*blockw]));
#else
                int info;
                constexpr AnyODE::getrf_callback<Real_t> getrf{};
                getrf(&blockw,
                      &blockw,
                      &(m_view.block(bi, 0, 0)),
                      &(ld),
                      &(m_ipiv[bi*blockw]),
                      &info);
#endif
                if ((info != 0) and (info_ == 0))
                    info_ = info;
                for (int ci = 0; ci < blockw; ++ci){
                    for (int di = 0; (di < ndiag) and (bi+di < (nblocks - 1)); ++di){
                        m_view.sub(di, bi, ci) /= m_view.block(bi, ci, ci);
                    }
                }
                if (bi < m_view.m_nsat)
                    for (int sati=bi; sati < m_view.m_nsat; ++sati)
                        for (int ci=0; ci < blockw; ++ci)
                            m_view.bot(sati, bi, ci) /= m_view.block(bi, ci, ci);
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
            const auto nblk = m_view.m_nblocks;
            const auto blkw = m_view.m_blockw;
            const auto ndia = m_view.m_ndiag;
            auto y = buffer_factory<Real_t>(nblk*blkw);
            auto first_nan_idx = std::distance(b, std::find_if(b, b + nblk*blkw, [](Real_t d) { return std::isnan(d); }));
            int info = (first_nan_idx == nblk*blkw) ? 0 : first_nan_idx + 1; // 0 => no NaN, 1-indexed.
            for (int bri = 0; bri < nblk; ++bri){ // Solves Ly = b (from LUx = b)
                for (int li = 0; li < blkw; ++li){
                    Real_t s = 0.0;
                    for (int lci = 0; lci < li; ++lci){
                        s += m_view.block(bri, li, lci)*y[bri*blkw + lci];
                    }
                    const int ci = m_colbyrow[bri*blkw + li];
                    for (int di = 1; di <= std::min(ndia, bri); ++di){
                        s += m_view.sub(di-1, bri-di, ci) * y[(bri-di)*blkw + ci];
                    }
                    for (int bci=0; bci <= m_view.m_nsat + bri - nblk; ++bci){
                        s += m_view.bot(nblk+bci-bri-1, bci, ci) * y[bci*blkw + ci];
                    }
                    y[bri*blkw + li] = b[bri*blkw + m_rowbycol[bri*blkw + li]] - s;
                }
            }
            for (int bri = nblk-1; bri >= 0; --bri){ // Solves Ux = y
                for (int li = blkw; li > 0; --li){
                    Real_t s = 0.0;
                    for (int ci = li; ci < blkw; ++ci){
                        s += m_view.block(bri, li-1, ci)*x[bri*blkw + ci];
                    }
                    const int ci = m_colbyrow[bri*blkw + li-1];
                    for (int di = 1; di <= std::min(nblk - bri - 1, ndia); ++di) {
                        s += m_view.sup(di-1, bri, ci)*x[(bri+di)*blkw + ci];
                    }
                    for (int sati=m_view.m_nsat; sati > bri; --sati){
                        s += m_view.top(sati-1, bri, ci)*x[(nblk - sati + bri)*blkw + ci];
                    }
                    x[bri*blkw+li-1] = (y[bri*blkw + li-1] - s)\
                        /(m_view.block(bri, li-1, li-1));
                    if (m_view.block(bri, li-1, li-1) == 0 and info == 0)
                        info = nblk*blkw + bri*blkw + (li-1);
                }
            }
            return info;
        }
    };

    template<typename Real_t = double>
    class ILU{
        ColMajBlockDiagMatrixView<Real_t> m_view_copy;
    public:
        ILU_inplace<Real_t> m_ilu_inplace;
        ILU(const ColMajBlockDiagMatrixView<Real_t>& view) :
            m_view_copy(view), m_ilu_inplace(ILU_inplace<Real_t>(m_view_copy)) {}
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x){
            return m_ilu_inplace.solve(b, x);
        }
    };

}

#if defined(BLOCK_DIAG_ILU_WITH_GETRF)
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
        for (int ci=0; ci<dim; ++ci)
            std::swap(A(ri1, ci), A(ri2, ci));
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
            A(ri, i) /= A(i, i);
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
