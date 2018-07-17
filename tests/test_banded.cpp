// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#if !defined(USE_LAPACK)
#define USE_LAPACK 1
#endif
#include <anyode/anyode_decomposition_lapack.hpp>

#include <array>
#include <cmath>


TEST_CASE( "BandedLU(view)", "[BandedLU]" ) {

    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    const int nsat = 0;
    const int ld = blockw;
// >>> scipy.linalg.lu_factor(numpy.array([[5, 3, 2, 0, 0, 0],
//                                         [5, 8, 0, 3, 0, 0],
//                                         [1, 0, 8, 4, 4, 0],
//                                         [0, 2, 4, 4, 0, 5],
//                                         [0, 0, 3, 0, 6, 9],
//                                         [0, 0, 0, 4, 2, 7]]))

    std::array<double, blockw*blockw*nblocks + 2*blockw*(nblocks-1)> data {{
            5, 5, 3, 8,
                8, 4, 4, 4,
                6, 2, 9, 7,
                1, 2, 3, 4,
                2, 3, 4, 5}};
    block_diag_ilu::BlockDiagMatrix<double> cmbdv {data.data(), nblocks, blockw, ndiag, nsat, ld};
    REQUIRE( cmbdv.m_ld == ld );
    std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
//    std::array<double, blockw*nblocks> x;
    std::array<double, blockw*nblocks> b;
    cmbdv.dot_vec(&xref[0], &b[0]);
    const int nouter = ndiag*blockw;
    auto bndv = AnyODE::BandedMatrix<double>(cmbdv, nouter, nouter);
    REQUIRE( bndv.m_ld == 3*blockw + 1 );
    REQUIRE( bndv.m_kl == 2 );
    REQUIRE( bndv.m_ku == 2 );
    REQUIRE(std::abs(bndv.m_data[4] - 5) < 1e-15);
    REQUIRE(std::abs(bndv.m_data[5] - 5) < 1e-15);
    REQUIRE(std::abs(bndv.m_data[10] - 3) < 1e-15);
    REQUIRE(std::abs(bndv.m_data[11] - 8) < 1e-15);
    REQUIRE(std::abs(bndv.m_data[12] - 0) < 1e-15);
    REQUIRE(std::abs(bndv.m_data[13] - 2) < 1e-15);

    auto lu = AnyODE::BandedLU<double>(&bndv);
    int flag = lu.factorize();
    REQUIRE( flag == 0 );
// array([[  5.00000000e+00,   3.00000000e+00,   2.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
//        [  1.00000000e+00,   5.00000000e+00,  -2.00000000e+00,   3.00000000e+00,   0.00000000e+00,   0.00000000e+00],
//        [  2.00000000e-01,  -1.20000000e-01,   7.36000000e+00,   4.36000000e+00,   4.00000000e+00,   0.00000000e+00],
//        [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   4.00000000e+00,   2.00000000e+00,   7.00000000e+00],
//        [  0.00000000e+00,   0.00000000e+00,   4.07608696e-01,  -4.44293478e-01,   5.25815217e+00,   1.21100543e+01],
//        [  0.00000000e+00,   4.00000000e-01,   6.52173913e-01,  -1.08695652e-02,  -4.91989664e-01,   1.10341085e+01]]

//    0    7    14           21           28          35
//    -    -     -            -            -           -
//
//0   X    X     X            X            X           X
//1   X    X     X            X            X           X
//2   X    X     2            3            4           7
//3   X    3    -2            4.36         2           12.1100543
//4   5    5     7.36         4.0          5.25815217  11.0341085
//5   1   -0.12  0           -0.444293478 -0.491989664 X
//6   0.2  0     0.407608696 -.0108695652  X           X

    REQUIRE( lu.m_ipiv[0] == 1 );
    REQUIRE( lu.m_ipiv[1] == 2 );
    REQUIRE( lu.m_ipiv[2] == 3 );
    // only check first three rows (row swapping)

    REQUIRE( std::abs((lu.m_view->m_data[4] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[5] - 1)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[6] - 0.2)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_view->m_data[10] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[11] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[12] + 0.12)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_view->m_data[16] - 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[17] + 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[18] - 7.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_view->m_data[23] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_view->m_data[24] - 4.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_view->m_data[30] - 4)/1e-15) < 1 );

}

TEST_CASE( "solve", "[BandedLU]" ) {

    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    const int nsat = 0;
    const int ld = blockw;
    std::array<double, blockw*blockw*nblocks + 2*blockw*nblocks> data {{
            5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
                1, 2, 3, 4,
                2, 3, 4, 5}};
    block_diag_ilu::BlockDiagMatrix<double> cmbdv {&data[0], nblocks, blockw, ndiag, nsat, ld};
    std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
    std::array<double, blockw*nblocks> x;
    std::array<double, blockw*nblocks> b;
    cmbdv.dot_vec(&xref[0], &b[0]);
    const int nouter = ndiag*blockw + blockw - 1;
    auto bndv = AnyODE::BandedMatrix<double>(cmbdv, nouter, nouter);
    auto lu = AnyODE::BandedLU<double>(&bndv);
    int flag = lu.factorize();
    REQUIRE( flag == 0 );
    flag = lu.solve(&b[0], &x[0]);
    REQUIRE( flag == 0 );
    for (int idx=0; idx<blockw*nblocks; ++idx){
        REQUIRE( std::abs((x[idx] - xref[idx])/1e-14) < 1 );
    }

}

std::array<double, 7*4> get_arr(){
    return std::array<double, 7*4> {{ 0, 0, 0, 0, 6, 1, 4,
                0, 0, 0, 3, 7, 2, 5,
                0, 0, 1, 4, 8, 3, 0,
                0, 0, 2, 5, 9, 0, 0
                }};
}

block_diag_ilu::BlockBandedMatrix<double> get_cmbv(std::array<double, 28>& arr){
    // 0 0 0 0
    // 0 0 0 0
    // 0 0 1 2
    // 0 3 4 5
    // 6 7 8 9
    // 1 2 3 0
    // 4 5 0 0

    // 6 3 1 0
    // 1 7 4 2
    // 4 2 8 5
    // 0 5 3 9

    const int blockw = 2;
    const int nblocks = 2;
    const int ndiag = 1;
    return block_diag_ilu::BlockBandedMatrix<double>((double*)arr.data(), nblocks, blockw, ndiag);
}

TEST_CASE( "global_block", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv(0, 0) == 6 );
    REQUIRE( cmbv(0, 1) == 3 );
    REQUIRE( cmbv(1, 0) == 1 );
    REQUIRE( cmbv(1, 1) == 7 );

    REQUIRE( cmbv(2, 2) == 8 );
    REQUIRE( cmbv(2, 3) == 5 );
    REQUIRE( cmbv(3, 2) == 3 );
    REQUIRE( cmbv(3, 3) == 9 );
}

TEST_CASE( "global_sub", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv(2, 0) == 4 );
    REQUIRE( cmbv(3, 1) == 5 );

}

TEST_CASE( "global_sup", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv(0, 2) == 1 );
    REQUIRE( cmbv(1, 3) == 2 );
}

TEST_CASE( "global_non_accessible", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv(2, 1) == 2 );
    REQUIRE( cmbv(1, 2) == 4 );
}

TEST_CASE( "block", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv.m_ld == 7 );
    REQUIRE( cmbv.block(0, 0, 0) == 6 );
    REQUIRE( cmbv.block(0, 0, 1) == 3 );
    REQUIRE( cmbv.block(0, 1, 0) == 1 );
    REQUIRE( cmbv.block(0, 1, 1) == 7 );

    REQUIRE( cmbv.block(1, 0, 0) == 8 );
    REQUIRE( cmbv.block(1, 0, 1) == 5 );
    REQUIRE( cmbv.block(1, 1, 0) == 3 );
    REQUIRE( cmbv.block(1, 1, 1) == 9 );
}

TEST_CASE( "sub", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv.sub(0, 0, 0) == 4 );
    REQUIRE( cmbv.sub(0, 0, 1) == 5 );
}

TEST_CASE( "sup", "[BlockBandedMatrix]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv.sup(0, 0, 0) == 1 );
    REQUIRE( cmbv.sup(0, 0, 1) == 2 );
}


std::array<double, 13*6> get_arr2(){
    return std::array<double, 13*6> {{
            0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 9, 0, 8, 0, 0, 0, 0, 0, 0, 0, 2, 6,
                0, 4, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 1, 5, 9, 0, 0, 0, 0, 0, 0, 0,
                0, 7, 2, 6, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 3, 0, 1, 5, 0, 0, 0, 0,
                0, 0, 0, 8, 0, 7, 2, 6, 0, 0, 0, 0}};
}

block_diag_ilu::BlockBandedMatrix<double> get_cmbv2(std::array<double, 13*6>& arr){
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 4 8
    // 0 0 0 0 0 0
    // 0 0 3 7 3 7
    // 0 2 0 2 0 2
    // 1 6 1 6 1 6
    // 5 0 5 0 5 0
    // 9 4 9 4 0 0
    // 0 0 0 0 0 0
    // 8 3 0 0 0 0

    // 1 2 3 0 4 0
    // 5 6 0 7 0 8
    // 9 0 1 2 3 0
    // 0 4 5 6 0 7
    // 8 0 9 0 1 2
    // 0 3 0 4 5 6
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    return block_diag_ilu::BlockBandedMatrix<double>((double*)arr.data(), nblocks, blockw, ndiag);
}


TEST_CASE( "block2", "[BlockBandedMatrix]" ) {
    auto arr = get_arr2();
    auto cmbv = get_cmbv2(arr);
    REQUIRE( cmbv.m_ld == 13 );
    REQUIRE( cmbv.block(0, 0, 0) == 1 );
    REQUIRE( cmbv.block(0, 0, 1) == 2 );
    REQUIRE( cmbv.block(0, 1, 0) == 5 );
    REQUIRE( cmbv.block(0, 1, 1) == 6 );

    REQUIRE( cmbv.block(1, 0, 0) == 1 );
    REQUIRE( cmbv.block(1, 0, 1) == 2 );
    REQUIRE( cmbv.block(1, 1, 0) == 5 );
    REQUIRE( cmbv.block(1, 1, 1) == 6 );

    REQUIRE( cmbv.block(2, 0, 0) == 1 );
    REQUIRE( cmbv.block(2, 0, 1) == 2 );
    REQUIRE( cmbv.block(2, 1, 0) == 5 );
    REQUIRE( cmbv.block(2, 1, 1) == 6 );
}

TEST_CASE( "sub2", "[BlockBandedMatrix]" ) {
    auto arr = get_arr2();
    auto cmbv = get_cmbv2(arr);
    REQUIRE( cmbv.sub(0, 0, 0) == 9 );
    REQUIRE( cmbv.sub(0, 0, 1) == 4 );
    REQUIRE( cmbv.sub(0, 1, 0) == 9 );
    REQUIRE( cmbv.sub(0, 1, 1) == 4 );

    REQUIRE( cmbv.sub(1, 0, 0) == 8 );
    REQUIRE( cmbv.sub(1, 0, 1) == 3 );

}

TEST_CASE( "sup2", "[BlockBandedMatrix]" ) {
    auto arr = get_arr2();
    auto cmbv = get_cmbv2(arr);
    REQUIRE( cmbv.sup(0, 0, 0) == 3 );
    REQUIRE( cmbv.sup(0, 0, 1) == 7 );
    REQUIRE( cmbv.sup(0, 1, 0) == 3 );
    REQUIRE( cmbv.sup(0, 1, 1) == 7 );

    REQUIRE( cmbv.sup(1, 0, 0) == 4 );
    REQUIRE( cmbv.sup(1, 0, 1) == 8 );
}

TEST_CASE( "block_sub_sup__offset", "[BlockBandedMatrix]" ) {
    // 1 2 3 0 4 0
    // 5 6 0 7 0 8
    // 9 0 1 2 3 0
    // 0 4 5 6 0 7
    // 8 0 9 0 1 2
    // 0 3 0 4 5 6

    // 0 0 0 0 4 8
    // 0 0 0 0 0 0
    // 0 0 3 7 3 7
    // 0 2 0 2 0 2
    // 1 6 1 6 1 6
    // 5 0 5 0 5 0
    // 9 4 9 4 0 0
    // 0 0 0 0 0 0
    // 8 3 0 0 0 0


    std::array<double, 13*6> arr {{
                0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 9, 0, 8,
                0, 0, 0, 0, 0, 0, 0, 2, 6, 0, 4, 0, 3,
                0, 0, 0, 0, 0, 0, 3, 0, 1, 5, 9, 0, 0,
                0, 0, 0, 0, 0, 0, 7, 2, 6, 0, 4, 0, 0,
                0, 0, 0, 0, 4, 0, 3, 0, 1, 5, 0, 0, 0,
                0, 0, 0, 0, 8, 0, 7, 2, 6, 0, 0, 0, 0}};
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    auto cmbv = block_diag_ilu::BlockBandedMatrix<double>((double*)arr.data(), nblocks, blockw, ndiag);
    REQUIRE( cmbv.m_ld == 13 );
    for (auto i=0; i < nblocks; ++i){
        REQUIRE( cmbv.block(i, 0, 0) == 1 );
        REQUIRE( cmbv.block(i, 0, 1) == 2 );
        REQUIRE( cmbv.block(i, 1, 0) == 5 );
        REQUIRE( cmbv.block(i, 1, 1) == 6 );
    }
    REQUIRE( cmbv.sup(0, 0, 0) == 3 );
    REQUIRE( cmbv.sup(0, 0, 1) == 7 );
    REQUIRE( cmbv.sup(0, 1, 0) == 3 );
    REQUIRE( cmbv.sup(0, 1, 1) == 7 );

    REQUIRE( cmbv.sup(1, 0, 0) == 4 );
    REQUIRE( cmbv.sup(1, 0, 1) == 8 );

    REQUIRE( cmbv.sub(0, 0, 0) == 9 );
    REQUIRE( cmbv.sub(0, 0, 1) == 4 );
    REQUIRE( cmbv.sub(0, 1, 0) == 9 );
    REQUIRE( cmbv.sub(0, 1, 1) == 4 );

    REQUIRE( cmbv.sub(1, 0, 0) == 8 );
    REQUIRE( cmbv.sub(1, 0, 1) == 3 );
}

TEST_CASE( "as_banded_padded", "[BlockDiagMatrix]" ) {
    constexpr int blockw = 2;
    constexpr int nblocks = 3;
    constexpr int ndiag = 1;
    constexpr int nsat = 0;
    constexpr int ld = blockw;
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks + 2*blockw*(nblocks-1)> data {{5, 5, 3, 8,
                8, 4, 4, 4,
                6, 2, 9, 7,
                1, 2, 3, 4,
                2, 3, 4, 5}};
    block_diag_ilu::BlockDiagMatrix<double> cmbdv {&data[0], nblocks, blockw, ndiag, nsat, ld};
    auto banded = cmbdv.as_banded_padded();

//   0   7  14  21  28  35
//   -   -   -   -   -   -
//0  X   X   X   X   X   X
//1  X   X   X   X   X   X
//2  X   X   2   3   4   5
//3  X   3   #   4   #   9
//4  5   8   8   4   6   7
//5  5   #   4   #   2   X
//6  1   2   3   4   X   X
    REQUIRE( std::abs((banded->m_data[4] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[5] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[6] - 1)/1e-15) < 1 );

    REQUIRE( std::abs((banded->m_data[10] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[11] - 8)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[13] - 2)/1e-15) < 1 );

    REQUIRE( std::abs((banded->m_data[16] - 2)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[18] - 8)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[19] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[20] - 3)/1e-15) < 1 );

    REQUIRE( std::abs((banded->m_data[23] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[24] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[25] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[27] - 4)/1e-15) < 1 );

    REQUIRE( std::abs((banded->m_data[30] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[32] - 6)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[33] - 2)/1e-15) < 1 );

    REQUIRE( std::abs((banded->m_data[37] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[38] - 9)/1e-15) < 1 );
    REQUIRE( std::abs((banded->m_data[39] - 7)/1e-15) < 1 );
}
