#pragma once

namespace block_diag_ilu {
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
        Real_t& sat(const int sati, const int blocki, const int li) const noexcept { // no error checking
            int ri, ci;
            const int nblk = this->m_nblocks;
            const int blkw = this->m_blockw;
            if (sati > 0){
                ri = blocki*blkw + li;
                ci = (nblk+blocki - sati)*blkw + li;
            } else{
                ri = (nblk+blocki + sati)*blkw + li;
                ci = blocki*blkw + li;
            }
            return m_data[col_maj ? ri + ci*m_ld : ri*m_ld + ci];
        }
    };
}
