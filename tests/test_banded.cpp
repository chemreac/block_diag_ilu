// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include "block_diag_ilu/banded.hpp"
#include <array>
#include <cmath>


TEST_CASE( "LU(view)", "[LU]" ) {

    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    std::array<double, blockw*blockw*nblocks> block_d {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
    std::array<double, blockw*nblocks> sub_d {{1, 2, 3, 4}};
    std::array<double, blockw*nblocks> sup_d {{2, 3, 4, 5}};
    block_diag_ilu::ColMajBlockDiagView<double> cmbdv {&block_d[0], &sub_d[0], &sup_d[0], nblocks, blockw, ndiag};
    std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
    std::array<double, blockw*nblocks> x;
    std::array<double, blockw*nblocks> b;
    cmbdv.dot_vec(&xref[0], &b[0]);
    auto lu = block_diag_ilu::LU<double>(cmbdv);
// >>> scipy.linalg.lu_factor(numpy.array([[5, 3, 2, 0, 0, 0],
//                                         [5, 8, 0, 3, 0, 0],
//                                         [1, 0, 8, 4, 4, 0],
//                                         [0, 2, 4, 4, 0, 5],
//                                         [0, 0, 3, 0, 6, 9],
//                                         [0, 0, 0, 4, 2, 7]]))

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

    // only check first three rows (row swapping)

    REQUIRE( std::abs((lu.m_data[4] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[5] - 1)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[6] - 0.2)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_data[10] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[11] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[12] + 0.12)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_data[16] - 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[17] + 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[18] - 7.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_data[23] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.m_data[24] - 4.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.m_data[30] - 4)/1e-15) < 1 );

}

TEST_CASE( "solve", "[LU]" ) {

    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    std::array<double, blockw*blockw*nblocks> block_d {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
    std::array<double, blockw*nblocks> sub_d {{1, 2, 3, 4}};
    std::array<double, blockw*nblocks> sup_d {{2, 3, 4, 5}};
    block_diag_ilu::ColMajBlockDiagView<double> cmbdv {&block_d[0], &sub_d[0], &sup_d[0], nblocks, blockw, ndiag};
    std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
    std::array<double, blockw*nblocks> x;
    std::array<double, blockw*nblocks> b;
    cmbdv.dot_vec(&xref[0], &b[0]);
    auto lu = block_diag_ilu::LU<double>(cmbdv);
    int flag = lu.solve(&b[0], &x[0]);
    REQUIRE( flag == 0 );
    for (int idx=0; idx<blockw*nblocks; ++idx){
        REQUIRE( std::abs((x[idx] - xref[idx])/1e-14) < 1 );
    }

}

std::array<double, 28> get_arr(){
    return std::array<double, 7*4> {{ 0, 0, 0, 0, 6, 1, 4, 0, 0, 0, 3, 7, 2, 5, 0,
                0, 1, 4, 8, 3, 0, 0, 0, 2, 5, 9, 0, 0 }};
}

block_diag_ilu::ColMajBandedView<double> get_cmbv(std::array<double, 28>& arr){
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
    const int nouter = 2;
    return block_diag_ilu::ColMajBandedView<double>((double*)arr.data(), nblocks, blockw, ndiag);
}


TEST_CASE( "block", "[ColMajBandedView]" ) {
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

TEST_CASE( "sub", "[ColMajBandedView]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv.sub(0, 0, 0) == 4 );
    REQUIRE( cmbv.sub(0, 0, 1) == 5 );
}

TEST_CASE( "sup", "[ColMajBandedView]" ) {
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

block_diag_ilu::ColMajBandedView<double> get_cmbv2(std::array<double, 13*6>& arr){
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
    return block_diag_ilu::ColMajBandedView<double>((double*)arr.data(), nblocks, blockw, ndiag);
}


TEST_CASE( "block2", "[ColMajBandedView]" ) {
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

TEST_CASE( "sub2", "[ColMajBandedView]" ) {
    auto arr = get_arr2();
    auto cmbv = get_cmbv2(arr);
    REQUIRE( cmbv.sub(0, 0, 0) == 9 );
    REQUIRE( cmbv.sub(0, 0, 1) == 4 );
    REQUIRE( cmbv.sub(0, 1, 0) == 9 );
    REQUIRE( cmbv.sub(0, 1, 1) == 4 );

    REQUIRE( cmbv.sub(1, 0, 0) == 8 );
    REQUIRE( cmbv.sub(1, 0, 1) == 3 );

}

TEST_CASE( "sup2", "[ColMajBandedView]" ) {
    auto arr = get_arr2();
    auto cmbv = get_cmbv2(arr);
    REQUIRE( cmbv.sup(0, 0, 0) == 3 );
    REQUIRE( cmbv.sup(0, 0, 1) == 7 );
    REQUIRE( cmbv.sup(0, 1, 0) == 3 );
    REQUIRE( cmbv.sup(0, 1, 1) == 7 );

    REQUIRE( cmbv.sup(1, 0, 0) == 4 );
    REQUIRE( cmbv.sup(1, 0, 1) == 8 );
}

TEST_CASE( "block_sub_sup__offset", "[ColMajBandedView]" ) {
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


    std::array<double, 54> arr {{
                0, 0, 0, 0, 1, 5, 9, 0, 8,
                0, 0, 0, 2, 6, 0, 4, 0, 3,
                0, 0, 3, 0, 1, 5, 9, 0, 0,
                0, 0, 7, 2, 6, 0, 4, 0, 0,
                4, 0, 3, 0, 1, 5, 0, 0, 0,
                8, 0, 7, 2, 6, 0, 0, 0, 0}};
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    auto cmbv = block_diag_ilu::ColMajBandedView<double>((double*)arr.data(), nblocks, blockw, ndiag, 0, 0);
    REQUIRE( cmbv.m_ld == 9 );
    for (auto i=0; i<2; ++i){
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

TEST_CASE( "to_banded", "[ColMajBlockDiagView]" ) {
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks> block_d {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
    std::array<double, blockw*nblocks> sub_d {{1, 2, 3, 4}};
    std::array<double, blockw*nblocks> sup_d {{2, 3, 4, 5}};
    block_diag_ilu::ColMajBlockDiagView<double> cmbdv {&block_d[0], &sub_d[0], &sup_d[0], nblocks, blockw, ndiag};
    auto banded = cmbdv.to_banded();

//   0   7  14  21  28  35
//   -   -   -   -   -   -
//0  X   X   X   X   X   X
//1  X   X   X   X   X   X
//2  X   X   2   3   4   5
//3  X   3   #   4   #   9
//4  5   8   8   4   6   7
//5  5   #   4   #   2   X
//6  1   2   3   4   X   X
    REQUIRE( std::abs((banded[4] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded[5] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded[6] - 1)/1e-15) < 1 );

    REQUIRE( std::abs((banded[10] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((banded[11] - 8)/1e-15) < 1 );
    REQUIRE( std::abs((banded[13] - 2)/1e-15) < 1 );

    REQUIRE( std::abs((banded[16] - 2)/1e-15) < 1 );
    REQUIRE( std::abs((banded[18] - 8)/1e-15) < 1 );
    REQUIRE( std::abs((banded[19] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded[20] - 3)/1e-15) < 1 );

    REQUIRE( std::abs((banded[23] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((banded[24] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded[25] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded[27] - 4)/1e-15) < 1 );

    REQUIRE( std::abs((banded[30] - 4)/1e-15) < 1 );
    REQUIRE( std::abs((banded[32] - 6)/1e-15) < 1 );
    REQUIRE( std::abs((banded[33] - 2)/1e-15) < 1 );

    REQUIRE( std::abs((banded[37] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((banded[38] - 9)/1e-15) < 1 );
    REQUIRE( std::abs((banded[39] - 7)/1e-15) < 1 );
}
