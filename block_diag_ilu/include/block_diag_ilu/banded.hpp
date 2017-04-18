#pragma once
#include <anyode/anyode_blas_lapack.hpp>

namespace block_diag_ilu {
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

    template <typename Real_t = double>
    class LU {  // Wrapper around DGBTRF & DGBTRS from LAPACK
        static_assert(sizeof(Real_t) == 8, "LAPACK DGBTRF & DGBTRS operates on 64-bit IEEE 754 floats.");
#ifdef BLOCK_DIAG_ILU_UNIT_TEST
    public:
#endif
        int m_dim, m_nouter, m_ld;
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
            std::copy(this->m_view.m_data, this->m_view.m_data + m_ld*m_dim, m_data);
            if (m_ld != banded_ld_(view.m_nouter)){
                throw std::runtime_error("LAPACK requires padding");
            }
            factorize();
        }
        void factorize(){
            int info;
            constexpr AnyODE::gbtrf_callback<Real_t> gbtrf{};
            gbtrf(&m_dim, &m_dim, &m_nouter, &m_nouter,
                  buffer_get_raw_ptr(m_data),
                  &m_ld,
                  buffer_get_raw_ptr(m_ipiv), &info);
            if (info){
                throw std::runtime_error("DGBTRF failed.");
            }
        }
        int solve(const Real_t * const __restrict__ b, Real_t * const __restrict__ x){
            const char trans = 'N'; // no transpose
            std::copy(b, b + m_dim, x);
            int info, nrhs=1;
            constexpr AnyODE::gbtrs_callback<Real_t> gbtrs{};
            gbtrs(&trans, &m_dim, &m_nouter, &m_nouter, &nrhs,
                  buffer_get_raw_ptr(m_data), &m_ld,
                  buffer_get_raw_ptr(m_ipiv), x, &m_dim, &info);
            return info;
        };
    };
}
