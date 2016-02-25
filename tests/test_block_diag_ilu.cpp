// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>
#include <cmath>

block_diag_ilu::ColMajBlockDiagMat<double> get_test_case_colmajblockdiagmat(){
    constexpr int nblocks = 3;
    constexpr int blockw = 2;
    constexpr int ndiag = 1;
    block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag};
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks> blocks {{
            5, 5, 3, 8,
            8, 4, 4, 4,
            6, 2, 9, 7}};
    std::array<double,blockw*(nblocks-1)> sub {{
            1, 2, 3, 4 }};
    std::array<double,blockw*(nblocks-1)> sup {{
            2, 3, 4, 5 }};
    for (size_t bi=0; bi<3; ++bi)
        for (size_t ci=0; ci<2; ++ci){
            if (bi<2){
                cmbdm.view.sub(0, bi, ci) = sub[bi*2+ci];
                cmbdm.view.sup(0, bi, ci) = sup[bi*2+ci];
            }
            for (size_t ri=0; ri<2; ++ri)
                cmbdm.view.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
        }
    return cmbdm;
}


TEST_CASE( "rowpiv2rowbycol" "[ILU_inplace]" ) {
    const std::array<int, 5> piv {{3, 4, 5, 4, 5}};
    std::array<int, 5> rowbycol;
    block_diag_ilu::rowpiv2rowbycol(5, &piv[0], &rowbycol[0]);
    REQUIRE( rowbycol[0] == 2 );
    REQUIRE( rowbycol[1] == 3 );
    REQUIRE( rowbycol[2] == 4 );
    REQUIRE( rowbycol[3] == 1 );
    REQUIRE( rowbycol[4] == 0 );
}

TEST_CASE( "rowbycol2colbyrow" "[ILU_inplace]" ) {
    const std::array<int, 5> rowbycol {{2, 3, 4, 1, 0}};
    std::array<int, 5> colbyrow;
    block_diag_ilu::rowbycol2colbyrow(5, &rowbycol[0], &colbyrow[0]);
    REQUIRE( colbyrow[0] == 4 );
    REQUIRE( colbyrow[1] == 3 );
    REQUIRE( colbyrow[2] == 0 );
    REQUIRE( colbyrow[3] == 1 );
    REQUIRE( colbyrow[4] == 2 );
}

TEST_CASE( "_get_test_m2 in test_fakelu.py", "[ILU_inplace]" ) {

    // this is _get_test_m2 in test_fakelu.py
    auto cmbdm = get_test_case_colmajblockdiagmat();
    auto cmbdv = cmbdm.view;
    block_diag_ilu::ILU_inplace ilu(cmbdv);

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
    block_diag_ilu::ColMajBlockDiagView<double> cmbdv {
        (double*)block.data(), (double*)sub.data(), (double*)sup.data(), nblocks, blockw, ndiag};
    block_diag_ilu::ILU_inplace ilu(cmbdv);

    REQUIRE( ilu.nblocks() == nblocks );
    REQUIRE( ilu.blockw() == blockw );
    REQUIRE( ilu.ndiag() == ndiag );

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

TEST_CASE( "addressing", "[ColMajBlockDiagView]" ) {
    std::array<double, 2*2*3> blocks {{1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12 }};
    std::array<double, 2*2> sub {{
            13, 14, 15, 16 }};
    std::array<double, 2*2> sup {{
            17, 18, 19, 20 }};
    block_diag_ilu::ColMajBlockDiagView<double> v {
        blocks.data(), sub.data(), sup.data(), 3, 2, 1};

    SECTION( "block" ) {
        REQUIRE( v.block(0, 0, 0) == blocks[0] );
        REQUIRE( v.block(0, 1, 0) == blocks[1] );
        REQUIRE( v.block(0, 0, 1) == blocks[2] );
        REQUIRE( v.block(0, 1, 1) == blocks[3] );

        REQUIRE( v.block(1, 0, 0) == blocks[4] );
        REQUIRE( v.block(1, 1, 0) == blocks[5] );
        REQUIRE( v.block(1, 0, 1) == blocks[6] );
        REQUIRE( v.block(1, 1, 1) == blocks[7] );

        REQUIRE( v.block(2, 0, 0) == blocks[8] );
        REQUIRE( v.block(2, 1, 0) == blocks[9] );
        REQUIRE( v.block(2, 0, 1) == blocks[10] );
        REQUIRE( v.block(2, 1, 1) == blocks[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( v.sub(0, 0, 0) == sub[0] );
        REQUIRE( v.sub(0, 0, 1) == sub[1] );
        REQUIRE( v.sub(0, 1, 0) == sub[2] );
        REQUIRE( v.sub(0, 1, 1) == sub[3] );
    }
    SECTION( "sup" ) {
        REQUIRE( v.sup(0, 0, 0) == sup[0] );
        REQUIRE( v.sup(0, 0, 1) == sup[1] );
        REQUIRE( v.sup(0, 1, 0) == sup[2] );
        REQUIRE( v.sup(0, 1, 1) == sup[3] );
    }
}

TEST_CASE( "addressing multi diag", "[ColMajBlockDiagView]" ) {
    auto X = -99.0;
    std::array<double, 2*3*3+2> blocks {{
            1, 2, X,
                3, 4, X,
                X,
                5, 6, X,
                7, 8, X,
                X,
                9, 10, X,
                11, 12, X,
            }};
    std::array<double, 3*(2+1)> sub {{
            13, 14, X, 15, 16, X,
                91, 92, X}};
    std::array<double, 3*(2+1)> sup {{
            17, 18, X, 19, 20, X,
                81, 82, X}};
    block_diag_ilu::ColMajBlockDiagView<double> v {blocks.data(), sub.data(), sup.data(), 3, 2, 2, 3, 7, 3};

    SECTION( "block" ) {
        REQUIRE( v.block(0, 0, 0) == 1 );
        REQUIRE( v.block(0, 1, 0) == 2 );
        REQUIRE( v.block(0, 0, 1) == 3 );
        REQUIRE( v.block(0, 1, 1) == 4 );

        REQUIRE( v.block(1, 0, 0) == 5 );
        REQUIRE( v.block(1, 1, 0) == 6 );
        REQUIRE( v.block(1, 0, 1) == 7 );
        REQUIRE( v.block(1, 1, 1) == 8 );

        REQUIRE( v.block(2, 0, 0) == 9 );
        REQUIRE( v.block(2, 1, 0) == 10 );
        REQUIRE( v.block(2, 0, 1) == 11 );
        REQUIRE( v.block(2, 1, 1) == 12 );
    }
    SECTION( "sub" ) {
        REQUIRE( v.sub(0, 0, 0) == 13 );
        REQUIRE( v.sub(0, 0, 1) == 14 );
        REQUIRE( v.sub(0, 1, 0) == 15 );
        REQUIRE( v.sub(0, 1, 1) == 16 );

        REQUIRE( v.sub(1, 0, 0) == 91 );
        REQUIRE( v.sub(1, 0, 1) == 92 );
    }
    SECTION( "sup" ) {
        REQUIRE( v.sup(0, 0, 0) == 17 );
        REQUIRE( v.sup(0, 0, 1) == 18 );
        REQUIRE( v.sup(0, 1, 0) == 19 );
        REQUIRE( v.sup(0, 1, 1) == 20 );

        REQUIRE( v.sup(1, 0, 0) == 81 );
        REQUIRE( v.sup(1, 0, 1) == 82 );
    }
}

TEST_CASE( "average_diag_weight", "[ColMajBlockDiagView]" ) {
    auto X = -97.0;
    std::array<double, 2*3*3 + 2> blocks {{
                100, 5, X,
                5, 100, X,
                X,
                100, 11, X,
                11, 100, X,
                X,
                100, 13, X,
                13, 100, X,
            }};
    std::array<double, 3*(2+1)> sub {{
            10, 10, X, 10, 10, X,
                1, 1, X}};
    std::array<double, 3*(2+1)> sup {{
            10, 10, X, 10, 10, X,
                1, 1, X}};
    block_diag_ilu::ColMajBlockDiagView<double> v {blocks.data(), sub.data(), sup.data(), 3, 2, 2, 3, 7, 3};
    auto res_0 = v.average_diag_weight(0);
    auto res_1 = v.average_diag_weight(1);
    REQUIRE( std::abs(res_0 - 10) < 1e-15 );
    REQUIRE( std::abs(res_1 - 100) < 1e-15 );
}


TEST_CASE( "set_to_1_minus_gamma_times_other", "[ColMajBlockDiagView]" ) {
    std::array<double, 2*2*3> blocks {{1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12 }};
    std::array<double, 2*2> sub {{
            13, 14, 15, 16 }};
    std::array<double, 2*2> sup {{
            17, 18, 19, 20 }};
    block_diag_ilu::ColMajBlockDiagView<double> v {
        blocks.data(), sub.data(), sup.data(), 3, 2, 1};

    block_diag_ilu::ColMajBlockDiagMat<double> m {3, 2, 1};
    double gamma = 0.7;
    m.view.set_to_1_minus_gamma_times_view(gamma, v);

    SECTION( "block" ) {
        REQUIRE( m.view.block(0, 0, 0) == 1-gamma*blocks[0] );
        REQUIRE( m.view.block(0, 1, 0) == -gamma*blocks[1] );
        REQUIRE( m.view.block(0, 0, 1) == -gamma*blocks[2] );
        REQUIRE( m.view.block(0, 1, 1) == 1-gamma*blocks[3] );

        REQUIRE( m.view.block(1, 0, 0) == 1-gamma*blocks[4] );
        REQUIRE( m.view.block(1, 1, 0) == -gamma*blocks[5] );
        REQUIRE( m.view.block(1, 0, 1) == -gamma*blocks[6] );
        REQUIRE( m.view.block(1, 1, 1) == 1-gamma*blocks[7] );

        REQUIRE( m.view.block(2, 0, 0) == 1-gamma*blocks[8] );
        REQUIRE( m.view.block(2, 1, 0) == -gamma*blocks[9] );
        REQUIRE( m.view.block(2, 0, 1) == -gamma*blocks[10] );
        REQUIRE( m.view.block(2, 1, 1) == 1-gamma*blocks[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( m.view.sub(0, 0, 0) == -gamma*sub[0] );
        REQUIRE( m.view.sub(0, 0, 1) == -gamma*sub[1] );
        REQUIRE( m.view.sub(0, 1, 0) == -gamma*sub[2] );
        REQUIRE( m.view.sub(0, 1, 1) == -gamma*sub[3] );
    }
    SECTION( "sup" ) {
        REQUIRE( m.view.sup(0, 0, 0) == -gamma*sup[0] );
        REQUIRE( m.view.sup(0, 0, 1) == -gamma*sup[1] );
        REQUIRE( m.view.sup(0, 1, 0) == -gamma*sup[2] );
        REQUIRE( m.view.sup(0, 1, 1) == -gamma*sup[3] );
    }
}


TEST_CASE( "dot_vec", "[ColMajBlockDiagMat]" ) {
    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    auto cmbdm = get_test_case_colmajblockdiagmat();
    const std::array<double, nblocks*blockw> x {{3, 2, 6, 2, 5, 4}};
    const std::array<double, nblocks*blockw> bref {{
        5*3 + 3*2 + 2*6,
            5*3 + 8*2 + 3*2,
            8*6 + 4*2 + 4*5 + 1*3,
            4*6 + 4*2 + 5*4 + 2*2,
            6*5 + 9*4 + 3*6,
            2*5 + 7*4 + 4*2}};
    std::array<double, nblocks*blockw> b;
    cmbdm.view.dot_vec(x.data(), b.data());
    REQUIRE( std::abs(b[0] - bref[0]) < 1e-15 );
    REQUIRE( std::abs(b[1] - bref[1]) < 1e-15 );
    REQUIRE( std::abs(b[2] - bref[2]) < 1e-15 );
    REQUIRE( std::abs(b[3] - bref[3]) < 1e-15 );
    REQUIRE( std::abs(b[4] - bref[4]) < 1e-15 );
    REQUIRE( std::abs(b[5] - bref[5]) < 1e-15 );
}

TEST_CASE( "get_global", "[ColMajBlockDiagView]" ) {
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    auto X = -99;
    // 5 3 2 # 6 #
    // 5 8 # 3 # 7
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // 5 # 3 # 6 9
    // # 6 # 4 2 7
    std::array<double, blockw*blockw*nblocks> block_d {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7}};
    std::array<double, blockw*nblocks> sub_d {{1, 2, 3, 4, 5, 6}};
    std::array<double, blockw*nblocks> sup_d {{2, 3, 4, 5, 6, 7}};
    block_diag_ilu::ColMajBlockDiagView<double> cmbdv {&block_d[0], &sub_d[0], &sup_d[0], nblocks, blockw, ndiag};

    // sub
    REQUIRE( std::abs((cmbdv.sub(0, 0, 0) - 1)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 0, 1) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 1, 0) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(0, 1, 1) - 4)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.sub(1, 0, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sub(1, 0, 1) - 6)/1e-15) < 1 );
    // sup
    REQUIRE( std::abs((cmbdv.sup(0, 0, 0) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 0, 1) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 1, 0) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(0, 1, 1) - 5)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv.sup(1, 0, 0) - 6)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv.sup(1, 0, 1) - 7)/1e-15) < 1 );
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

TEST_CASE( "dot_vec2", "[ColMajBlockDiagMat]" ) {
    const int blockw = 4;
    const int nblocks = 3;
    const int nx = blockw*nblocks;
    const int ndiag = 2;

    block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag};
    for (size_t bi=0; bi<nblocks; ++bi)
        for (size_t ci=0; ci<blockw; ++ci){
            for (size_t ri=0; ri<blockw; ++ri)
                cmbdm.view.block(bi, ri, ci) = 1.2*bi + 4.1*ci - 2.7*ri;
        }

    for (int di=0; di<ndiag; ++di){
        for (int bi=0; bi<nblocks-di-1; ++bi){
            for (int ci=0; ci<blockw; ++ci){
                cmbdm.view.sub(di, bi, ci) = 1.5*bi + 0.6*ci - 0.1*di;
                cmbdm.view.sup(di, bi, ci) = 3.7*bi - 0.3*ci + 0.2*di;
            }
        }
    }

    std::array<double, nx> x;
    for (int i=0; i<nx; ++i)
        x[i] = 1.5*nblocks*blockw - 2.7*i;

    std::array<double, nblocks*blockw> bref;
    for (int ri=0; ri<nx; ++ri){
        double val = 0.0;
        for (int ci=0; ci<nx; ++ci){
            val += x[ci] * cmbdm.view.get_global(ri, ci);
        }
        bref[ri] = val;
    }
    std::array<double, nblocks*blockw> b;
    cmbdm.view.dot_vec(x.data(), b.data());
    for (int i=0; i<nx; ++i)
        REQUIRE( std::abs((b[i] - bref[i])/5e-14) < 1 );
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

TEST_CASE( "copy_to_matrix", "[ColMajBlockDiagView]" ) {
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
    auto mat = cmbdv.copy_to_matrix();
    for (int bi=0; bi<3; ++bi){
        for (int ri=0; ri<2; ++ri){
            for (int ci=0; ci<2; ++ci){
                const double diff = mat.view.block(bi, ri, ci) - cmbdv.block(bi, ri, ci);
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mat.view.sub(0, bi, li) - cmbdv.sub(0, bi, li);
                const double supdiff = mat.view.sup(0, bi, li) - cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
    mat.view.zero_out_blocks();
    mat.view.zero_out_diags();
    mat.view.scale_diag_add(cmbdv, 2, 1);
    for (int bi=0; bi<3; ++bi){
        for (int ri=0; ri<2; ++ri){
            for (int ci=0; ci<2; ++ci){
                double diff = mat.view.block(bi, ri, ci) - 2*cmbdv.block(bi, ri, ci);
                if (ri == ci){
                    diff -= 1;
                }
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mat.view.sub(0, bi, li) - 2*cmbdv.sub(0, bi, li);
                const double supdiff = mat.view.sup(0, bi, li) - 2*cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
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

TEST_CASE( "rms_diag", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    auto rms_subd = cmbdm.view.rms_diag(-1);
    auto rms_main = cmbdm.view.rms_diag(0);
    auto rms_supd = cmbdm.view.rms_diag(1);
    auto ref_subd = std::sqrt((1+4+9+16)/4.0);
    auto ref_main = std::sqrt((25+64+64+16+36+49)/6.0);
    auto ref_supd = std::sqrt((4+9+16+25)/4.0);
    REQUIRE( std::abs((rms_subd - ref_subd)/1e-14) < 1 );
    REQUIRE( std::abs((rms_main - ref_main)/1e-14) < 1 );
    REQUIRE( std::abs((rms_supd - ref_supd)/1e-14) < 1 );
}

TEST_CASE( "zero_out_blocks", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    cmbdm.view.zero_out_blocks();
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci)
            for (int ri=0; ri<2; ++ri)
                REQUIRE( cmbdm.view.block(bi, ri, ci) == 0.0 );
}

TEST_CASE( "zero_out_diags", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    cmbdm.view.zero_out_diags();
    for (int bi=0; bi<2; ++bi)
        for (int ci=0; ci<2; ++ci){
            REQUIRE( cmbdm.view.sub(0, bi, ci) == 0.0 );
            REQUIRE( cmbdm.view.sup(0, bi, ci) == 0.0 );
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
    const uint nouter = 2;
    return block_diag_ilu::ColMajBandedView<double>((double*)arr.data(), nblocks, blockw, ndiag);
}


TEST_CASE( "block", "[ColMajBandedView]" ) {
    auto arr = get_arr();
    auto cmbv = get_cmbv(arr);
    REQUIRE( cmbv.ld == 7 );
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
    REQUIRE( cmbv.ld == 13 );
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

TEST_CASE( "block_sub_sup", "[DenseView]" ) {
    // 1 2 3 0 4 0
    // 5 6 0 7 0 8
    // 9 0 1 2 3 0
    // 0 4 5 6 0 7
    // 8 0 9 0 1 2
    // 0 3 0 4 5 6
    std::array<double, 36> arr {{1, 5, 9, 0, 8, 0, 2, 6, 0, 4, 0, 3, 3, 0, 1, 5, 9, 0, 0, 7, 2, 6, 0, 4, 4, 0, 3, 0, 1, 5,
                0, 8, 0, 7, 2, 6}};
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    auto cmdv = block_diag_ilu::DenseView<double>((double*)arr.data(), nblocks, blockw, ndiag);  // col maj
    REQUIRE( cmdv.ld == 6 );
    for (auto i=0; i<2; ++i){
        REQUIRE( cmdv.block(i, 0, 0) == 1 );
        REQUIRE( cmdv.block(i, 0, 1) == 2 );
        REQUIRE( cmdv.block(i, 1, 0) == 5 );
        REQUIRE( cmdv.block(i, 1, 1) == 6 );
    }
    REQUIRE( cmdv.sup(0, 0, 0) == 3 );
    REQUIRE( cmdv.sup(0, 0, 1) == 7 );
    REQUIRE( cmdv.sup(0, 1, 0) == 3 );
    REQUIRE( cmdv.sup(0, 1, 1) == 7 );
    REQUIRE( cmdv.sup(1, 0, 0) == 4 );
    REQUIRE( cmdv.sup(1, 0, 1) == 8 );

    REQUIRE( cmdv.sub(0, 0, 0) == 9 );
    REQUIRE( cmdv.sub(0, 0, 1) == 4 );
    REQUIRE( cmdv.sub(0, 1, 0) == 9 );
    REQUIRE( cmdv.sub(0, 1, 1) == 4 );
    REQUIRE( cmdv.sub(1, 0, 0) == 8 );
    REQUIRE( cmdv.sub(1, 0, 1) == 3 );

    std::array<double, 36> rarr {{
            1, 2, 3, 0, 4, 0,
                5, 6, 0, 7, 0, 8,
                9, 0, 1, 2, 3, 0,
                0, 4, 5, 6, 0, 7,
                8, 0, 9, 0, 1, 2,
                0, 3, 0, 4, 5, 6
                }};
    auto rmdv = block_diag_ilu::DenseView<double, false>((double*)rarr.data(), nblocks, blockw, ndiag);  // row maj
    REQUIRE( rmdv.ld == 6 );
    for (auto i=0; i<2; ++i){
        REQUIRE( rmdv.block(i, 0, 0) == 1 );
        REQUIRE( rmdv.block(i, 0, 1) == 2 );
        REQUIRE( rmdv.block(i, 1, 0) == 5 );
        REQUIRE( rmdv.block(i, 1, 1) == 6 );
    }
    REQUIRE( rmdv.sup(0, 0, 0) == 3 );
    REQUIRE( rmdv.sup(0, 0, 1) == 7 );
    REQUIRE( rmdv.sup(0, 1, 0) == 3 );
    REQUIRE( rmdv.sup(0, 1, 1) == 7 );
    REQUIRE( rmdv.sup(1, 0, 0) == 4 );
    REQUIRE( rmdv.sup(1, 0, 1) == 8 );

    REQUIRE( rmdv.sub(0, 0, 0) == 9 );
    REQUIRE( rmdv.sub(0, 0, 1) == 4 );
    REQUIRE( rmdv.sub(0, 1, 0) == 9 );
    REQUIRE( rmdv.sub(0, 1, 1) == 4 );
    REQUIRE( rmdv.sub(1, 0, 0) == 8 );
    REQUIRE( rmdv.sub(1, 0, 1) == 3 );
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
    REQUIRE( cmbv.ld == 9 );
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
