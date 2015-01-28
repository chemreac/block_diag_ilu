// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>
#include <cmath>


TEST_CASE( "rowpiv2rowbycol" "[ILU]" ) {
    const std::array<int, 5> piv {{3, 4, 5, 4, 5}};
    std::array<int, 5> rowbycol;
    block_diag_ilu::rowpiv2rowbycol(5, &piv[0], &rowbycol[0]);
    REQUIRE( rowbycol[0] == 2 );
    REQUIRE( rowbycol[1] == 3 );
    REQUIRE( rowbycol[2] == 4 );
    REQUIRE( rowbycol[3] == 1 );
    REQUIRE( rowbycol[4] == 0 );
}

TEST_CASE( "rowbycol2colbyrow" "[ILU]" ) {
    const std::array<int, 5> rowbycol {{2, 3, 4, 1, 0}};
    std::array<int, 5> colbyrow;
    block_diag_ilu::rowbycol2colbyrow(5, &rowbycol[0], &colbyrow[0]);
    REQUIRE( colbyrow[0] == 4 );
    REQUIRE( colbyrow[1] == 3 );
    REQUIRE( colbyrow[2] == 0 );
    REQUIRE( colbyrow[3] == 1 );
    REQUIRE( colbyrow[4] == 2 );
}

TEST_CASE( "_get_test_m2 in test_fakelu.py", "[ILU]" ) {

    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    std::array<double, blockw*blockw*nblocks> block {{
            5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
    std::array<double, blockw*(nblocks-1)> sub {{1, 2, 3, 4}};
    const std::array<double, blockw*(nblocks-1)> sup {{2, 3, 4, 5}};
    block_diag_ilu::ILU ilu(block.data(), sub.data(), sup.data(),
                            nblocks, blockw, ndiag);

    REQUIRE( ilu.nblocks == nblocks );
    REQUIRE( ilu.blockw == blockw );
    REQUIRE( ilu.ndiag == ndiag );

    SECTION( "check lower correctly computed" ) {
        REQUIRE( ilu.sub_get(0, 0, 0) == 1/5. );
        REQUIRE( ilu.sub_get(0, 0, 1) == 2/5. );
        REQUIRE( ilu.sub_get(0, 1, 0) == 3/8. );
        REQUIRE( ilu.sub_get(0, 1, 1) == 4/2. );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.sup_get(0, 0, 0) == 2 );
        REQUIRE( ilu.sup_get(0, 0, 1) == 3 );
        REQUIRE( ilu.sup_get(0, 1, 0) == 4 );
        REQUIRE( ilu.sup_get(0, 1, 1) == 5 );
    }
    SECTION( "solve performs adequately" ) {
        std::array<double, 6> b {{65, 202, 11, 65, 60, 121}};
        std::array<double, 6> xref {{-31.47775, 53.42125, 31.0625,
                    -43.36875, -19.25625, 19.5875}};
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( std::abs((x[0] - xref[0])/1e-14) < 1.0 );
        REQUIRE( std::abs((x[1] - xref[1])/1e-14) < 1.0 );
        REQUIRE( std::abs((x[2] - xref[2])/1e-14) < 1.0 );
        REQUIRE( std::abs((x[3] - xref[3])/1e-14) < 1.0 );
        REQUIRE( std::abs((x[4] - xref[4])/1e-14) < 1.0 );
        REQUIRE( std::abs((x[5] - xref[5])/1e-14) < 1.0 );
    }
}

TEST_CASE( "_get_test_m4 in test_fakelu.py", "[ILU]" ) {

    // this is _get_test_m4 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    std::array<double, blockw*blockw*nblocks> block {{
            -17, 37, 63, 13, 11, -42, 72, 24, 72, 14, -13, -57}};
    std::array<double, 6> sub {{.1, .2, -.1, .08, .03, -.1}};
    const std::array<double, 6> sup {{.2, .3, -.1, .2, .02, .03}};
    block_diag_ilu::ILU ilu(block.data(), sub.data(), sup.data(),
                            nblocks, blockw, ndiag);

    REQUIRE( ilu.nblocks == nblocks );
    REQUIRE( ilu.blockw == blockw );
    REQUIRE( ilu.ndiag == ndiag );

    SECTION( "check lower correctly computed" ) {
        REQUIRE( std::abs(ilu.sub_get(0, 0, 0) - .1/37 ) < 1e-15 );
        REQUIRE( std::abs(ilu.sub_get(0, 0, 1) - .2/(63+17/37.*13) ) < 1e-15 );
        REQUIRE( std::abs(ilu.sub_get(0, 1, 0) - .1/42 ) < 1e-15 );
        REQUIRE( std::abs(ilu.sub_get(0, 1, 1) - .08/(72+11/42.*24) ) < 1e-15 );
        REQUIRE( std::abs(ilu.sub_get(1, 0, 0) - .03/37 ) < 1e-15 );
        REQUIRE( std::abs(ilu.sub_get(1, 0, 1) - -.1/(63+17/37.*13) ) < 1e-15 );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.sup_get(0, 0, 0) == 0.2 );
        REQUIRE( ilu.sup_get(0, 0, 1) == 0.3 );
        REQUIRE( ilu.sup_get(0, 1, 0) == -.1 );
        REQUIRE( ilu.sup_get(0, 1, 1) == 0.2 );
        REQUIRE( ilu.sup_get(1, 0, 0) == 0.02 );
        REQUIRE( ilu.sup_get(1, 0, 1) == 0.03 );
    }
    SECTION( "solve performs adequately" ) {
        std::array<double, 6> b {{-62, 207, 11, -14, 25, -167}};
        std::array<double, 6> xref {{5.42317616680147374e+00,
                4.78588898186963929e-01,
                4.00565700557765081e-01,
                8.73749367816923223e-02,
                9.14791409109598774e-01,
                3.15378902934640371e+00
                    }};
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( std::abs(x[0] - xref[0]) < 1e-15 );
        REQUIRE( std::abs(x[1] - xref[1]) < 1e-15 );
        REQUIRE( std::abs(x[2] - xref[2]) < 1e-15 );
        REQUIRE( std::abs(x[3] - xref[3]) < 1e-15 );
        REQUIRE( std::abs(x[4] - xref[4]) < 1e-15 );
        REQUIRE( std::abs(x[5] - xref[5]) < 1e-15 );
    }
}

TEST_CASE( "addressing", "[BlockDiagMat]" ) {
    block_diag_ilu::BlockDiagMat m {3, 2, 1};
    std::array<double, 2*2*3+2*2+2*2> d {{
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20
            }};
    for (size_t i=0; i<d.size(); ++i)
        m.data[i] = d[i];

    SECTION( "block" ) {
        REQUIRE( m.block(0, 0, 0) == d[0] );
        REQUIRE( m.block(0, 1, 0) == d[1] );
        REQUIRE( m.block(0, 0, 1) == d[2] );
        REQUIRE( m.block(0, 1, 1) == d[3] );

        REQUIRE( m.block(1, 0, 0) == d[4] );
        REQUIRE( m.block(1, 1, 0) == d[5] );
        REQUIRE( m.block(1, 0, 1) == d[6] );
        REQUIRE( m.block(1, 1, 1) == d[7] );

        REQUIRE( m.block(2, 0, 0) == d[8] );
        REQUIRE( m.block(2, 1, 0) == d[9] );
        REQUIRE( m.block(2, 0, 1) == d[10] );
        REQUIRE( m.block(2, 1, 1) == d[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( m.sub(0, 0, 0) == d[12] );
        REQUIRE( m.sub(0, 0, 1) == d[13] );
        REQUIRE( m.sub(0, 1, 0) == d[14] );
        REQUIRE( m.sub(0, 1, 1) == d[15] );
    }
    SECTION( "sup" ) {
        REQUIRE( m.sup(0, 0, 0) == d[16] );
        REQUIRE( m.sup(0, 0, 1) == d[17] );
        REQUIRE( m.sup(0, 1, 0) == d[18] );
        REQUIRE( m.sup(0, 1, 1) == d[19] );
    }
}

TEST_CASE( "addressing multi diag", "[BlockDiagMat]" ) {
    block_diag_ilu::BlockDiagMat m {3, 2, 2};
    std::array<double, 2*2*3+2*2*(2+1)> d {{
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            91, 92,
            17, 18, 19, 20,
            81, 82
            }};
    for (size_t i=0; i<d.size(); ++i)
        m.data[i] = d[i];

    SECTION( "block" ) {
        REQUIRE( m.block(0, 0, 0) == d[0] );
        REQUIRE( m.block(0, 1, 0) == d[1] );
        REQUIRE( m.block(0, 0, 1) == d[2] );
        REQUIRE( m.block(0, 1, 1) == d[3] );

        REQUIRE( m.block(1, 0, 0) == d[4] );
        REQUIRE( m.block(1, 1, 0) == d[5] );
        REQUIRE( m.block(1, 0, 1) == d[6] );
        REQUIRE( m.block(1, 1, 1) == d[7] );

        REQUIRE( m.block(2, 0, 0) == d[8] );
        REQUIRE( m.block(2, 1, 0) == d[9] );
        REQUIRE( m.block(2, 0, 1) == d[10] );
        REQUIRE( m.block(2, 1, 1) == d[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( m.sub(0, 0, 0) == d[12] );
        REQUIRE( m.sub(0, 0, 1) == d[13] );
        REQUIRE( m.sub(0, 1, 0) == d[14] );
        REQUIRE( m.sub(0, 1, 1) == d[15] );

        REQUIRE( m.sub(1, 0, 0) == d[16] );
        REQUIRE( m.sub(1, 0, 1) == d[17] );
    }
    SECTION( "sup" ) {
        REQUIRE( m.sup(0, 0, 0) == d[18] );
        REQUIRE( m.sup(0, 0, 1) == d[19] );
        REQUIRE( m.sup(0, 1, 0) == d[20] );
        REQUIRE( m.sup(0, 1, 1) == d[21] );

        REQUIRE( m.sup(1, 0, 0) == d[22] );
        REQUIRE( m.sup(1, 0, 1) == d[23] );
    }
}

TEST_CASE( "set_to_1_minus_gamma_times_other", "[BlockDiagMat]" ) {
    block_diag_ilu::BlockDiagMat m {3, 2, 1};
    std::array<double, 2*2*3+2*2+2*2> d {{
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20
            }};
    for (size_t i=0; i<d.size(); ++i)
        m.data[i] = d[i];

    block_diag_ilu::BlockDiagMat n {3, 2, 1};
    double gamma = 0.7;
    n.set_to_1_minus_gamma_times_other(gamma, m);

    SECTION( "block" ) {
        REQUIRE( n.block(0, 0, 0) == 1-gamma*d[0] );
        REQUIRE( n.block(0, 1, 0) == -gamma*d[1] );
        REQUIRE( n.block(0, 0, 1) == -gamma*d[2] );
        REQUIRE( n.block(0, 1, 1) == 1-gamma*d[3] );

        REQUIRE( n.block(1, 0, 0) == 1-gamma*d[4] );
        REQUIRE( n.block(1, 1, 0) == -gamma*d[5] );
        REQUIRE( n.block(1, 0, 1) == -gamma*d[6] );
        REQUIRE( n.block(1, 1, 1) == 1-gamma*d[7] );

        REQUIRE( n.block(2, 0, 0) == 1-gamma*d[8] );
        REQUIRE( n.block(2, 1, 0) == -gamma*d[9] );
        REQUIRE( n.block(2, 0, 1) == -gamma*d[10] );
        REQUIRE( n.block(2, 1, 1) == 1-gamma*d[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( n.sub(0, 0, 0) == -gamma*d[12] );
        REQUIRE( n.sub(0, 0, 1) == -gamma*d[13] );
        REQUIRE( n.sub(0, 1, 0) == -gamma*d[14] );
        REQUIRE( n.sub(0, 1, 1) == -gamma*d[15] );
    }
    SECTION( "sup" ) {
        REQUIRE( n.sup(0, 0, 0) == -gamma*d[16] );
        REQUIRE( n.sup(0, 0, 1) == -gamma*d[17] );
        REQUIRE( n.sup(0, 1, 0) == -gamma*d[18] );
        REQUIRE( n.sup(0, 1, 1) == -gamma*d[19] );
    }
}

TEST_CASE( "ilu_inplace", "[BlockDiagMat]" ) {

    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    block_diag_ilu::BlockDiagMat bdm {nblocks, blockw, ndiag};
    std::array<double, blockw*blockw*nblocks+2*blockw*(nblocks-1)> d {{
        5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
            1, 2, 3, 4,
            2, 3, 4, 5
            }};
    for (size_t i=0; i<d.size(); ++i)
        bdm.data[i] = d[i];

    block_diag_ilu::ILU ilu = bdm.ilu_inplace();

    REQUIRE( ilu.nblocks == nblocks );
    REQUIRE( ilu.blockw == blockw );
    REQUIRE( ilu.ndiag == ndiag );

    SECTION( "check lower correctly computed" ) {
        REQUIRE( ilu.sub_get(0, 0, 0) == 1/5. );
        REQUIRE( ilu.sub_get(0, 0, 1) == 2/5. );
        REQUIRE( ilu.sub_get(0, 1, 0) == 3/8. );
        REQUIRE( ilu.sub_get(0, 1, 1) == 4/2. );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.sup_get(0, 0, 0) == 2 );
        REQUIRE( ilu.sup_get(0, 0, 1) == 3 );
        REQUIRE( ilu.sup_get(0, 1, 0) == 4 );
        REQUIRE( ilu.sup_get(0, 1, 1) == 5 );
    }
    SECTION( "solve performs adequately" ) {
        std::array<double, 6> b {{65, 202, 11, 65, 60, 121}};
        std::array<double, 6> xref {{-31.47775, 53.42125, 31.0625,
                    -43.36875, -19.25625, 19.5875}};
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( std::abs((x[0] - xref[0])/1e-14) < 1 );
        REQUIRE( std::abs((x[1] - xref[1])/1e-14) < 1 );
        REQUIRE( std::abs((x[2] - xref[2])/1e-14) < 1 );
        REQUIRE( std::abs((x[3] - xref[3])/1e-14) < 1 );
        REQUIRE( std::abs((x[4] - xref[4])/1e-14) < 1 );
        REQUIRE( std::abs((x[5] - xref[5])/1e-14) < 1 );
    }
}

TEST_CASE( "dot_vec", "[BlockDiagMat]" ) {
    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    block_diag_ilu::BlockDiagMat bdm {nblocks, blockw, ndiag};
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks+2*blockw*(nblocks-1)> d {{
        5, 5, 3, 8,
            8, 4, 4, 4,
            6, 2, 9, 7,
            1, 2, 3, 4,
            2, 3, 4, 5
            }};
    for (size_t i=0; i<d.size(); ++i)
        bdm.data[i] = d[i];
    const std::array<double, nblocks*blockw> x {{3, 2, 6, 2, 5, 4}};
    const std::array<double, nblocks*blockw> bref {{
        5*3 + 3*2 + 2*6,
            5*3 + 8*2 + 3*2,
            8*6 + 4*2 + 4*5 + 1*3,
            4*6 + 4*2 + 5*4 + 2*2,
            6*5 + 9*4 + 3*6,
            2*5 + 7*4 + 4*2}};
    std::array<double, nblocks*blockw> b;
    bdm.dot_vec(x.data(), b.data());
    REQUIRE( std::abs(b[0] - bref[0]) < 1e-15 );
    REQUIRE( std::abs(b[1] - bref[1]) < 1e-15 );
    REQUIRE( std::abs(b[2] - bref[2]) < 1e-15 );
    REQUIRE( std::abs(b[3] - bref[3]) < 1e-15 );
    REQUIRE( std::abs(b[4] - bref[4]) < 1e-15 );
    REQUIRE( std::abs(b[5] - bref[5]) < 1e-15 );
}

TEST_CASE( "dot_vec2", "[BlockDiagMat]" ) {
    const int blockw = 4;
    const int nblocks = 3;
    const int nx = blockw*nblocks;
    const int ndiag = 2;
    const int ndata = blockw*blockw*nblocks + \
        2*block_diag_ilu::diag_store_len(nblocks, blockw, ndiag);

    block_diag_ilu::BlockDiagMat bdm {nblocks, blockw, ndiag};
    for (int i=0; i<ndata; ++i)
        bdm.data[i] = i + 2.5;

    std::array<double, nx> x;
    for (int i=0; i<nx; ++i)
        x[i] = 1.5*nblocks*blockw - 2.5*i;

    std::array<double, nblocks*blockw> bref;
    for (int i=0; i<nx; ++i){
        double val = 0.0;
        for (int j=0; j<nx; ++j){
            val += x[j] * bdm.get(i, j);
        }
        bref[i] = val;
    }
    std::array<double, nblocks*blockw> b;
    bdm.dot_vec(x.data(), b.data());
    for (int i=0; i<nx; ++i)
        REQUIRE( std::abs(b[i] - bref[i]) < 1e-15 );
}

#include <iostream> //DEBUG
// TEST_CASE( "factor_lu", "[ColMajBlockDiagMat]" ) {

//     const int blockw = 2;
//     const int nblocks = 3;
//     const int ndiag = 1;
//     std::cout << "A" << std::endl; std::cout.flush(); //DEBUG
//     block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag};
//     std::array<double, blockw*blockw*nblocks> block_d {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
//     std::array<double, blockw*nblocks> sub_d {{1, 2, 3, 4}};
//     std::array<double, blockw*nblocks> sup_d {{2, 3, 4, 5}};
//     std::cout << "B" << std::endl;std::cout.flush(); //DEBUG
//     memcpy(cmbdm.block_data.get(), &block_d[0], sizeof(double)*block_d.size());
//     memcpy(cmbdm.sub_data.get(), &sub_d[0], sizeof(double)*sub_d.size());
//     memcpy(cmbdm.sup_data.get(), &sup_d[0], sizeof(double)*sup_d.size());
//     std::cout << "C" << std::endl;std::cout.flush(); //DEBUG
//     std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
//     std::array<double, blockw*nblocks> x;
//     std::array<double, blockw*nblocks> b;
//     cmbdm.dot_vec(&xref[0], &b[0]);
//     std::cout << "D" << std::endl;std::cout.flush(); //DEBUG
//     auto lu = cmbdm.factor_lu();
//     std::cout << "E" << std::endl;std::cout.flush(); //DEBUG
//     lu.solve(&b[0], &x[0]);
//     std::cout << "F" << std::endl;std::cout.flush(); //DEBUG

//     for (size_t i=0; i<blockw*nblocks; ++i)
//         REQUIRE( std::abs((x[i] - xref[i])/1e-14) < 1 );
// }

TEST_CASE( "get_global", "[ColMajBlockDiagView]" ) {
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    auto X = -99;
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

    // sub
    REQUIRE( std::abs((cmbdv.sub(0, 0, 0) - 1)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 0, 1) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 1, 0) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 1, 1) - 4)/1e-15) < 1 );
    // sup
    REQUIRE( std::abs((cmbdv.sup(0, 0, 0) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 0, 1) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 1, 0) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 1, 1) - 5)/1e-15) < 1 );

    // block
    REQUIRE( std::abs((cmbdv.block(0, 0, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(0, 1, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(0, 0, 1) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(0, 1, 1) - 8)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.block(1, 0, 0) - 8)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(1, 1, 0) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(1, 0, 1) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(1, 1, 1) - 4)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.block(2, 0, 0) - 6)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(2, 1, 0) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(2, 0, 1) - 9)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.block(2, 1, 1) - 7)/1e-15) < 1 );

    // get global
    REQUIRE( std::abs((cmbdv.get_global(0, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(1, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(2, 0) - 1)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.get_global(0, 1) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(1, 1) - 8)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(3, 1) - 2)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.get_global(0, 2) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(2, 2) - 8)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(3, 2) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(4, 2) - 3)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.get_global(1, 3) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(2, 3) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(3, 3) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(5, 3) - 4)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.get_global(2, 4) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(4, 4) - 6)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(5, 4) - 2)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.get_global(3, 5) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(4, 5) - 9)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.get_global(5, 5) - 7)/1e-15) < 1 );
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

TEST_CASE( "dot_vec_ColMajBlockDiagView", "[ColMajBlockDiagView]" ) {
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
    std::array<double, blockw*nblocks> xref {{-7, 13, 9, -4, -0.7, 42}};
    std::array<double, blockw*nblocks> b;
    cmbdv.dot_vec(&xref[0], &b[0]);
    std::array<double, blockw*nblocks> bref {{22, 57, 46.2, 256, 400.8, 276.6}};
    for (int i=0; i<6; ++i)
        REQUIRE( std::abs((b[i] - bref[i])/1e-15) < 1 );
}


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
    auto lu = block_diag_ilu::LU(cmbdv);
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

    REQUIRE( std::abs((lu.data[4] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[5] - 1)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[6] - 0.2)/1e-15) < 1 );

    REQUIRE( std::abs((lu.data[10] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[11] - 5)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[12] + 0.12)/1e-15) < 1 );

    REQUIRE( std::abs((lu.data[16] - 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[17] + 2)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[18] - 7.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.data[23] - 3)/1e-15) < 1 );
    REQUIRE( std::abs((lu.data[24] - 4.36)/1e-15) < 1 );

    REQUIRE( std::abs((lu.data[30] - 4)/1e-15) < 1 );

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
    auto lu = block_diag_ilu::LU(cmbdv);
    lu.solve(&b[0], &x[0]);

    for (int idx=0; idx<blockw*nblocks; ++idx){
        REQUIRE( std::abs((x[idx] - xref[idx])/1e-14) < 1 );
    }

}
