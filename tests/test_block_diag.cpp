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
    block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag, 0};
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
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci){
            if (bi<2){
                cmbdm.m_view.sub(0, bi, ci) = sub[bi*2+ci];
                cmbdm.m_view.sup(0, bi, ci) = sup[bi*2+ci];
            }
            for (int ri=0; ri<2; ++ri)
                cmbdm.m_view.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
        }
    return cmbdm;
}

block_diag_ilu::ColMajBlockDiagMat<double> get_test_case_sat(){
    // >>> A = np.array([[7, 2, 3, 0, 4, 0],
    // ...               [6, 6, 0, 7, 0, 8],
    // ...               [1, 0, 1, 3, 9, 0],
    // ...               [0, 6, 5, 4, 0, 1],
    // ...               [8, 0, 2, 0, 9, 7],
    // ...               [0, 3, 0, 5, 2, 3]])
    // ...
    // >>> A.dot([-7, 4, 2, 1, 5, 9])
    // array([ 27,  68,  42,  49, -29,  96])
    constexpr int nblocks = 3;
    constexpr int blockw = 2;
    constexpr int ndiag = 0;
    constexpr int nsat = 2;
    block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag, nsat};
    std::array<double, blockw*blockw*nblocks> blocks {{
            7, 6, 2, 6,
            1, 5, 3, 4,
            9, 2, 7, 3}};
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci)
            for (int ri=0; ri<2; ++ri)
                cmbdm.m_view.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
    cmbdm.m_view.sat(1, 0, 0) = 4;
    cmbdm.m_view.sat(1, 0, 1) = 8;
    cmbdm.m_view.sat(2, 0, 0) = 3;
    cmbdm.m_view.sat(2, 0, 1) = 7;
    cmbdm.m_view.sat(2, 1, 0) = 9;
    cmbdm.m_view.sat(2, 1, 1) = 1;
    cmbdm.m_view.sat(-1, 0, 0) = 8;
    cmbdm.m_view.sat(-1, 0, 1) = 3;
    cmbdm.m_view.sat(-2, 0, 0) = 1;
    cmbdm.m_view.sat(-2, 0, 1) = 6;
    cmbdm.m_view.sat(-2, 1, 0) = 2;
    cmbdm.m_view.sat(-2, 1, 1) = 5;
    return cmbdm;
}

TEST_CASE( "get_global sattelites", "[ViewBase]" ) {
    auto cmbdm = get_test_case_sat();
    std::array<double, 36> ref {{7, 2, 3, 0, 4, 0, 6, 6, 0, 7, 0, 8, 1, 0, 1, 3, 9, 0, 0, 6, 5, 4, 0, 1
                , 8, 0, 2, 0, 9, 7, 0, 3, 0, 5, 2, 3}};
    for (int i=0; i<36; ++i){
        REQUIRE( std::abs((cmbdm.m_view.get_global(i/6, i%6) - ref[i])/1e-15) < 1 );
    }
}

TEST_CASE( "sattelites", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_sat();
    std::array<double, 6> vec {{-7, 4, 2, 1, 5, 9}};
    std::array<double, 6> res;
    cmbdm.m_view.dot_vec(&vec[0], &res[0]);
    REQUIRE( std::abs((res[0] + 15)/1e-14) < 1.0 );
    REQUIRE( std::abs((res[1] - 61)/1e-14) < 1.0 );
    REQUIRE( std::abs((res[2] - 43)/1e-14) < 1.0 );
    REQUIRE( std::abs((res[3] - 47)/1e-14) < 1.0 );
    REQUIRE( std::abs((res[4] - 56)/1e-14) < 1.0 );
    REQUIRE( std::abs((res[5] - 54)/1e-14) < 1.0 );
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
    auto cmbdv = cmbdm.m_view;
    block_diag_ilu::ILU_inplace<double> ilu(cmbdv);

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
        int flag = ilu.solve(b.data(), x.data());
        REQUIRE( flag == 0 );
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
    block_diag_ilu::ILU_inplace<double> ilu(cmbdv);

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
        int flag = ilu.solve(b.data(), x.data());
        REQUIRE( flag == 0 );
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
    block_diag_ilu::ColMajBlockDiagView<double> v {blocks.data(), sub.data(), sup.data(), 3, 2, 2, nullptr, 0, 3, 7, 3};

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
    block_diag_ilu::ColMajBlockDiagView<double> v {blocks.data(), sub.data(), sup.data(), 3, 2, 2, nullptr, 0, 3, 7, 3};
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

    block_diag_ilu::ColMajBlockDiagMat<double> m {3, 2, 1, 0};
    double gamma = 0.7;
    m.m_view.set_to_1_minus_gamma_times_view(gamma, v);

    SECTION( "block" ) {
        REQUIRE( m.m_view.block(0, 0, 0) == 1-gamma*blocks[0] );
        REQUIRE( m.m_view.block(0, 1, 0) == -gamma*blocks[1] );
        REQUIRE( m.m_view.block(0, 0, 1) == -gamma*blocks[2] );
        REQUIRE( m.m_view.block(0, 1, 1) == 1-gamma*blocks[3] );

        REQUIRE( m.m_view.block(1, 0, 0) == 1-gamma*blocks[4] );
        REQUIRE( m.m_view.block(1, 1, 0) == -gamma*blocks[5] );
        REQUIRE( m.m_view.block(1, 0, 1) == -gamma*blocks[6] );
        REQUIRE( m.m_view.block(1, 1, 1) == 1-gamma*blocks[7] );

        REQUIRE( m.m_view.block(2, 0, 0) == 1-gamma*blocks[8] );
        REQUIRE( m.m_view.block(2, 1, 0) == -gamma*blocks[9] );
        REQUIRE( m.m_view.block(2, 0, 1) == -gamma*blocks[10] );
        REQUIRE( m.m_view.block(2, 1, 1) == 1-gamma*blocks[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( m.m_view.sub(0, 0, 0) == -gamma*sub[0] );
        REQUIRE( m.m_view.sub(0, 0, 1) == -gamma*sub[1] );
        REQUIRE( m.m_view.sub(0, 1, 0) == -gamma*sub[2] );
        REQUIRE( m.m_view.sub(0, 1, 1) == -gamma*sub[3] );
    }
    SECTION( "sup" ) {
        REQUIRE( m.m_view.sup(0, 0, 0) == -gamma*sup[0] );
        REQUIRE( m.m_view.sup(0, 0, 1) == -gamma*sup[1] );
        REQUIRE( m.m_view.sup(0, 1, 0) == -gamma*sup[2] );
        REQUIRE( m.m_view.sup(0, 1, 1) == -gamma*sup[3] );
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
    cmbdm.m_view.dot_vec(x.data(), b.data());
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
    const int nsat = 0;

    block_diag_ilu::ColMajBlockDiagMat<double> cmbdm {nblocks, blockw, ndiag, nsat};
    for (int bi=0; bi<nblocks; ++bi)
        for (int ci=0; ci<blockw; ++ci){
            for (int ri=0; ri<blockw; ++ri)
                cmbdm.m_view.block(bi, ri, ci) = 1.2*bi + 4.1*ci - 2.7*ri;
        }

    for (int di=0; di<ndiag; ++di){
        for (int bi=0; bi<nblocks-di-1; ++bi){
            for (int ci=0; ci<blockw; ++ci){
                cmbdm.m_view.sub(di, bi, ci) = 1.5*bi + 0.6*ci - 0.1*di;
                cmbdm.m_view.sup(di, bi, ci) = 3.7*bi - 0.3*ci + 0.2*di;
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
            val += x[ci] * cmbdm.m_view.get_global(ri, ci);
        }
        bref[ri] = val;
    }
    std::array<double, nblocks*blockw> b;
    cmbdm.m_view.dot_vec(x.data(), b.data());
    for (int i=0; i<nx; ++i)
        REQUIRE( std::abs((b[i] - bref[i])/5e-14) < 1 );
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
                const double diff = mat.m_view.block(bi, ri, ci) - cmbdv.block(bi, ri, ci);
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mat.m_view.sub(0, bi, li) - cmbdv.sub(0, bi, li);
                const double supdiff = mat.m_view.sup(0, bi, li) - cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
    mat.m_view.zero_out_blocks();
    mat.m_view.zero_out_diags();
    mat.m_view.scale_diag_add(cmbdv, 2, 1);
    for (int bi=0; bi<3; ++bi){
        for (int ri=0; ri<2; ++ri){
            for (int ci=0; ci<2; ++ci){
                double diff = mat.m_view.block(bi, ri, ci) - 2*cmbdv.block(bi, ri, ci);
                if (ri == ci){
                    diff -= 1;
                }
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mat.m_view.sub(0, bi, li) - 2*cmbdv.sub(0, bi, li);
                const double supdiff = mat.m_view.sup(0, bi, li) - 2*cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
}


TEST_CASE( "rms_diag", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    auto rms_subd = cmbdm.m_view.rms_diag(-1);
    auto rms_main = cmbdm.m_view.rms_diag(0);
    auto rms_supd = cmbdm.m_view.rms_diag(1);
    auto ref_subd = std::sqrt((1+4+9+16)/4.0);
    auto ref_main = std::sqrt((25+64+64+16+36+49)/6.0);
    auto ref_supd = std::sqrt((4+9+16+25)/4.0);
    REQUIRE( std::abs((rms_subd - ref_subd)/1e-14) < 1 );
    REQUIRE( std::abs((rms_main - ref_main)/1e-14) < 1 );
    REQUIRE( std::abs((rms_supd - ref_supd)/1e-14) < 1 );
}

TEST_CASE( "zero_out_blocks", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    cmbdm.m_view.zero_out_blocks();
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci)
            for (int ri=0; ri<2; ++ri)
                REQUIRE( cmbdm.m_view.block(bi, ri, ci) == 0.0 );
}

TEST_CASE( "zero_out_diags", "[ColMajBlockDiagMat]" ) {
    auto cmbdm = get_test_case_colmajblockdiagmat();
    cmbdm.m_view.zero_out_diags();
    for (int bi=0; bi<2; ++bi)
        for (int ci=0; ci<2; ++ci){
            REQUIRE( cmbdm.m_view.sub(0, bi, ci) == 0.0 );
            REQUIRE( cmbdm.m_view.sup(0, bi, ci) == 0.0 );
        }
}

#if defined(BLOCK_DIAG_ILU_WITH_DGETRF)
TEST_CASE( "long double ilu inplace", "[ILU_inplace]" ) {
    constexpr int nblocks = 3;
    constexpr int blockw = 2;
    constexpr int ndiag = 1;
    constexpr int nsat = 1;
    block_diag_ilu::ColMajBlockDiagMat<long double> cmbdm {nblocks, blockw, ndiag, nsat};
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<long double, blockw*blockw*nblocks> blocks {{
            5, 5, 3, 8,
            8, 4, 4, 4,
            6, 2, 9, 7}};
    std::array<long double,blockw*(nblocks-1)> sub {{
            1, 2, 3, 4 }};
    std::array<long double,blockw*(nblocks-1)> sup {{
            2, 3, 4, 5 }};
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci){
            if (bi<2){
                cmbdm.m_view.sub(0, bi, ci) = sub[bi*2+ci];
                cmbdm.m_view.sup(0, bi, ci) = sup[bi*2+ci];
            }
            for (int ri=0; ri<2; ++ri)
                cmbdm.m_view.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
        }

    auto cmbdv = cmbdm.m_view;
    block_diag_ilu::ILU_inplace<long double> ilu(cmbdv);

    SECTION( "check lower correctly computed" ) {
        REQUIRE( std::abs(ilu.sub_get(0, 0, 0) - 1/5.L) < 1e-18L );
        REQUIRE( std::abs(ilu.sub_get(0, 0, 1) - 2/5.L) < 1e-18L );
        REQUIRE( std::abs(ilu.sub_get(0, 1, 0) - 3/8.L) < 1e-18L );
        REQUIRE( std::abs(ilu.sub_get(0, 1, 1) - 4/2.L) < 1e-18L );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.sup_get(0, 0, 0) == 2 );
        REQUIRE( ilu.sup_get(0, 0, 1) == 3 );
        REQUIRE( ilu.sup_get(0, 1, 0) == 4 );
        REQUIRE( ilu.sup_get(0, 1, 1) == 5 );
    }
    SECTION( "solve performs adequately" ) {
        std::array<long double, 6> b {{65L, 202L, 11L, 65L, 60L, 121L}};
        std::array<long double, 6> xref {{-31.47775L, 53.42125L, 31.0625L,
                    -43.36875L, -19.25625L, 19.5875L}};
        std::array<long double, 6> x;
        int flag = ilu.solve(b.data(), x.data());
        REQUIRE( flag == 0 );
        REQUIRE( std::abs(x[0] - xref[0]) < 5e-18L );
        REQUIRE( std::abs(x[1] - xref[1]) < 5e-18L );
        REQUIRE( std::abs(x[2] - xref[2]) < 5e-18L );
        REQUIRE( std::abs(x[3] - xref[3]) < 5e-18L );
        REQUIRE( std::abs(x[4] - xref[4]) < 5e-18L );
        REQUIRE( std::abs(x[5] - xref[5]) < 5e-18L );
    }
}
#endif