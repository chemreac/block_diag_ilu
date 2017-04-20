// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>
#include <cmath>

block_diag_ilu::ColMajBlockDiagMatrixView<double> get_test_case_colmajblockdiagmat(){
    constexpr int nblocks = 3;
    constexpr int blockw = 2;
    constexpr int ndiag = 1;
    constexpr int nsat = 0;
    constexpr int ld = 6;
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdmv {nullptr, nblocks, blockw, ndiag, nsat, ld};
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
                cmbdmv.sub(0, bi, ci) = sub[bi*2+ci];
                cmbdmv.sup(0, bi, ci) = sup[bi*2+ci];
            }
            for (int ri=0; ri<2; ++ri)
                cmbdmv.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
        }
    return cmbdmv;
}

block_diag_ilu::ColMajBlockDiagMatrixView<double> get_test_case_sat(){
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
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdmv {nullptr, nblocks, blockw, ndiag, nsat};
    std::array<double, blockw*blockw*nblocks> blocks {{
            7, 6, 2, 6,
            1, 5, 3, 4,
            9, 2, 7, 3}};
    for (int bi=0; bi<3; ++bi)
        for (int ci=0; ci<2; ++ci)
            for (int ri=0; ri<2; ++ri)
                cmbdmv.block(bi, ri, ci) = blocks[bi*4 + ci*2 + ri];
    cmbdmv.top(0, 0, 0) = 4;
    cmbdmv.top(0, 0, 1) = 8;
    cmbdmv.top(1, 0, 0) = 3;
    cmbdmv.top(1, 0, 1) = 7;
    cmbdmv.top(1, 1, 0) = 9;
    cmbdmv.top(1, 1, 1) = 1;
    cmbdmv.bot(0, 0, 0) = 8;
    cmbdmv.bot(0, 0, 1) = 3;
    cmbdmv.bot(1, 0, 0) = 1;
    cmbdmv.bot(1, 0, 1) = 6;
    cmbdmv.bot(1, 1, 0) = 2;
    cmbdmv.bot(1, 1, 1) = 5;
    return cmbdmv;
}

TEST_CASE( "get_global sattelites", "[ViewBase]" ) {
    auto cmbdmv = get_test_case_sat();
    std::array<double, 36> ref {{7, 2, 3, 0, 4, 0,
                6, 6, 0, 7, 0, 8,
                1, 0, 1, 3, 9, 0,
                0, 6, 5, 4, 0, 1,
                8, 0, 2, 0, 9, 7,
                0, 3, 0, 5, 2, 3}};

    for (int i=0; i<36; ++i){
        const int ri = i/6;
        const int ci = i%6;
        if (cmbdmv.valid_index(ri, ci)){
            REQUIRE( std::abs((cmbdmv(ri, ci) - ref[i])/1e-15) < 1 );
        }
    }
}

TEST_CASE( "sattelites", "[ColMajBlockDiagMat]" ) {
    auto cmbdmv = get_test_case_sat();
    std::array<double, 6> vec {{-7, 4, 2, 1, 5, 9}};
    std::array<double, 6> res;
    cmbdmv.dot_vec(&vec[0], &res[0]);
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
    auto cmbdv = get_test_case_colmajblockdiagmat();
    block_diag_ilu::ILU_inplace<double> ilu(cmbdv);

    SECTION( "check lower correctly computed" ) {
        REQUIRE( ilu.m_view.sub(0, 0, 0) == 1/5. );
        REQUIRE( ilu.m_view.sub(0, 0, 1) == 2/5. );
        REQUIRE( ilu.m_view.sub(0, 1, 0) == 3/8. );
        REQUIRE( ilu.m_view.sub(0, 1, 1) == 4/2. );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.m_view.sup(0, 0, 0) == 2 );
        REQUIRE( ilu.m_view.sup(0, 0, 1) == 3 );
        REQUIRE( ilu.m_view.sup(0, 1, 0) == 4 );
        REQUIRE( ilu.m_view.sup(0, 1, 1) == 5 );
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
    const int nsat = 0;
    const int ld = 2;
    std::array<double, blockw*blockw*nblocks + 12> data {{
            -17, 37, 63, 13, 11, -42, 72, 24, 72, 14, -13, -57, .1, .2,
                -.1, .08, .03, -.1, .2, .3, -.1, .2, .02, .03}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdv {
        (double*)data.data(), nblocks, blockw, ndiag, nsat, ld};
    block_diag_ilu::ILU_inplace<double> ilu(cmbdv);

    REQUIRE( ilu.m_view.m_nblocks == nblocks );
    REQUIRE( ilu.m_view.m_blockw == blockw );
    REQUIRE( ilu.m_view.m_ndiag == ndiag );

    SECTION( "check lower correctly computed" ) {
        REQUIRE( std::abs(ilu.m_view.sub(0, 0, 0) - .1/37 ) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(0, 0, 1) - .2/(63+17/37.*13) ) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(0, 1, 0) - .1/42 ) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(0, 1, 1) - .08/(72+11/42.*24) ) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(1, 0, 0) - .03/37 ) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(1, 0, 1) - -.1/(63+17/37.*13) ) < 1e-15 );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.m_view.sup(0, 0, 0) == 0.2 );
        REQUIRE( ilu.m_view.sup(0, 0, 1) == 0.3 );
        REQUIRE( ilu.m_view.sup(0, 1, 0) == -.1 );
        REQUIRE( ilu.m_view.sup(0, 1, 1) == 0.2 );
        REQUIRE( ilu.m_view.sup(1, 0, 0) == 0.02 );
        REQUIRE( ilu.m_view.sup(1, 0, 1) == 0.03 );
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
    std::array<double, 2*2*3 + 8> data {{1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20 }};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> v {
        data.data(), 3, 2, 1, 0, 2};
    double * blocks = data.data();
    double * sub = data.data() + 2*2*3;
    double * sup = data.data() + 2*2*3 + 2*(3-1);

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
    std::array<double, 2*3*3 + 3*(2+1) + 3*(2+1)> data {{
            1, 2, X,
                3, 4, X,
                5, 6, X,
                7, 8, X,
                9, 10, X,
                11, 12, X,

            13, 14, X, 15, 16, X,
                91, 92, X,

            17, 18, X, 19, 20, X,
                81, 82, X}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> v {data.data(), 3, 2, 2, 0, 3};

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
    std::array<double, 2*3*3 + 3*(2+1) + 3*(2+1)> data {{
                100, 5, X,
                5, 100, X,
                100, 11, X,
                11, 100, X,
                100, 13, X,
                13, 100, X,

            10, 10, X, 10, 10, X,
                    1, 1, X,

            10, 10, X, 10, 10, X,
                1, 1, X}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> v {data.data(), 3, 2, 2, 0, 3};
    auto res_0 = v.average_diag_weight(0);
    auto res_1 = v.average_diag_weight(1);
    REQUIRE( std::abs(res_0 - 10) < 1e-15 );
    REQUIRE( std::abs(res_1 - 100) < 1e-15 );
}


TEST_CASE( "set_to_eye_plus_scaled_mtx", "[ColMajBlockDiagView]" ) {
    constexpr int blockw = 2;
    constexpr int nblocks = 3;
    constexpr int ndiag = 1;
    constexpr int nsat = 0;
    constexpr int ld = blockw;
    std::array<double, 2*2*3 + 2*2 + 2*2> data {{1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,

                13, 14, 15, 16,

            17, 18, 19, 20 }};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> v {
        data.data(), nblocks, blockw, ndiag, nsat, ld};

    block_diag_ilu::ColMajBlockDiagMatrixView<double> m {nullptr, nblocks, blockw, ndiag, nsat, ld};
    double gamma = 0.7;
    m.set_to_eye_plus_scaled_mtx(-gamma, v);

    double * blocks = data.data();
    double * sub = data.data() + 2*2*3;
    double * sup = data.data() + 2*2*3 + 2*(3-1);

    SECTION( "block" ) {
        REQUIRE( m.block(0, 0, 0) == 1-gamma*blocks[0] );
        REQUIRE( m.block(0, 1, 0) == -gamma*blocks[1] );
        REQUIRE( m.block(0, 0, 1) == -gamma*blocks[2] );
        REQUIRE( m.block(0, 1, 1) == 1-gamma*blocks[3] );

        REQUIRE( m.block(1, 0, 0) == 1-gamma*blocks[4] );
        REQUIRE( m.block(1, 1, 0) == -gamma*blocks[5] );
        REQUIRE( m.block(1, 0, 1) == -gamma*blocks[6] );
        REQUIRE( m.block(1, 1, 1) == 1-gamma*blocks[7] );

        REQUIRE( m.block(2, 0, 0) == 1-gamma*blocks[8] );
        REQUIRE( m.block(2, 1, 0) == -gamma*blocks[9] );
        REQUIRE( m.block(2, 0, 1) == -gamma*blocks[10] );
        REQUIRE( m.block(2, 1, 1) == 1-gamma*blocks[11] );
    }
    SECTION( "sub" ) {
        REQUIRE( m.sub(0, 0, 0) == -gamma*sub[0] );
        REQUIRE( m.sub(0, 0, 1) == -gamma*sub[1] );
        REQUIRE( m.sub(0, 1, 0) == -gamma*sub[2] );
        REQUIRE( m.sub(0, 1, 1) == -gamma*sub[3] );
    }
    SECTION( "sup" ) {
        REQUIRE( m.sup(0, 0, 0) == -gamma*sup[0] );
        REQUIRE( m.sup(0, 0, 1) == -gamma*sup[1] );
        REQUIRE( m.sup(0, 1, 0) == -gamma*sup[2] );
        REQUIRE( m.sup(0, 1, 1) == -gamma*sup[3] );
    }
}


TEST_CASE( "dot_vec", "[ColMajBlockDiagMat]" ) {
    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    auto cmbdmv = get_test_case_colmajblockdiagmat();
    const std::array<double, nblocks*blockw> x {{3, 2, 6, 2, 5, 4}};
    const std::array<double, nblocks*blockw> bref {{
        5*3 + 3*2 + 2*6,
            5*3 + 8*2 + 3*2,
            8*6 + 4*2 + 4*5 + 1*3,
            4*6 + 4*2 + 5*4 + 2*2,
            6*5 + 9*4 + 3*6,
            2*5 + 7*4 + 4*2}};
    std::array<double, nblocks*blockw> b;
    cmbdmv.dot_vec(x.data(), b.data());
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
    const int nsat = 0;
    const int ld = 2;
    auto X = -99;
    // 5 3 2 # 6 #
    // 5 8 # 3 # 7
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // 5 # 3 # 6 9
    // # 6 # 4 2 7
    std::array<double, blockw*blockw*nblocks + 2*blockw*nblocks> data {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
                1, 2, 3, 4, 5, 6,
                2, 3, 4, 5, 6, 7}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdv {&data[0], nblocks, blockw, ndiag, nsat, ld};

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
    REQUIRE( std::abs((cmbdv(0, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(1, 0) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(2, 0) - 1)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv(0, 1) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(1, 1) - 8)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(3, 1) - 2)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv(0, 2) - 2)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(2, 2) - 8)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(3, 2) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(4, 2) - 3)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv(1, 3) - 3)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(2, 3) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(3, 3) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(5, 3) - 4)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv(2, 4) - 4)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(4, 4) - 6)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(5, 4) - 2)/1e-15) < 1 );

    REQUIRE( std::abs((cmbdv(3, 5) - 5)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(4, 5) - 9)/1e-15) < 1 );
    REQUIRE( std::abs((cmbdv(5, 5) - 7)/1e-15) < 1 );
}

TEST_CASE( "dot_vec2", "[ColMajBlockDiagMat]" ) {
    const int blockw = 4;
    const int nblocks = 3;
    const int nx = blockw*nblocks;
    const int ndiag = 2;
    const int nsat = 0;
    const int ld = 4;

    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdmv {nullptr, nblocks, blockw, ndiag, nsat, ld};
    for (int bi=0; bi<nblocks; ++bi)
        for (int ci=0; ci<blockw; ++ci)
            for (int ri=0; ri<blockw; ++ri)
                cmbdmv.block(bi, ri, ci) = 1.2*bi + 4.1*ci - 2.7*ri;

    for (int di=0; di<ndiag; ++di){
        for (int bi=0; bi<nblocks-di-1; ++bi){
            for (int ci=0; ci<blockw; ++ci){
                cmbdmv.sub(di, bi, ci) = 1.5*bi + 0.6*ci - 0.1*di;
                cmbdmv.sup(di, bi, ci) = 3.7*bi - 0.3*ci + 0.2*di;
            }
        }
    }

    std::array<double, nx> x;
    for (int i=0; i<nx; ++i)
        x[i] = 1.5*nblocks*blockw - 2.7*i;

    std::array<double, nx> bref;
    for (int ri=0; ri<nx; ++ri){
        double val = 0.0;
        for (int ci=0; ci<nx; ++ci){
            try{
                val += x[ci] * cmbdmv(ri, ci);
            } catch (...) {
                ; // invalid indices
            }
        }
        bref[ri] = val;
    }
    std::array<double, nblocks*blockw> b;
    cmbdmv.dot_vec(x.data(), b.data());
    for (int i=0; i<nx; ++i)
        REQUIRE( std::abs((b[i] - bref[i])/5e-14) < 1 );
}

TEST_CASE( "dot_vec_ColMajBlockDiagView", "[ColMajBlockDiagView]" ) {
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    const int nsat=0;
    const int ld = blockw;
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks + 2*blockw*(nblocks-1)> data {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
                1, 2, 3, 4, 2, 3, 4, 5}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdv {&data[0], nblocks, blockw, ndiag, nsat, ld};
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
    const int ld = blockw;
    const int nsat = 0;
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, blockw*blockw*nblocks + 2*blockw*nblocks> data {{5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
                1, 2, 3, 4, 2, 3, 4, 5}};
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdv {data.data(), nblocks, blockw, ndiag, nsat, ld};
    auto mv = cmbdv;
    for (int bi=0; bi<3; ++bi){
        for (int ri=0; ri<2; ++ri){
            for (int ci=0; ci<2; ++ci){
                const double diff = mv.block(bi, ri, ci) - cmbdv.block(bi, ri, ci);
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mv.sub(0, bi, li) - cmbdv.sub(0, bi, li);
                const double supdiff = mv.sup(0, bi, li) - cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
    mv.scale_diag_add(cmbdv, 2, 1);
    for (int bi=0; bi<3; ++bi){
        for (int ri=0; ri<2; ++ri){
            for (int ci=0; ci<2; ++ci){
                double diff = mv.block(bi, ri, ci) - 2*cmbdv.block(bi, ri, ci);
                if (ri == ci){
                    diff -= 1;
                }
                REQUIRE( std::abs(diff) < 1e-15 );
            }
        }
        if (bi < 2){
            for (int li=0; li<2; ++li){
                const double subdiff = mv.sub(0, bi, li) - 2*cmbdv.sub(0, bi, li);
                const double supdiff = mv.sup(0, bi, li) - 2*cmbdv.sup(0, bi, li);
                REQUIRE( std::abs(subdiff) < 1e-15 );
                REQUIRE( std::abs(supdiff) < 1e-15 );
            }
        }
    }
}


TEST_CASE( "rms_diag", "[ColMajBlockDiagMat]" ) {
    auto cmbdmv = get_test_case_colmajblockdiagmat();
    auto rms_subd = cmbdmv.rms_diag(-1);
    auto rms_main = cmbdmv.rms_diag(0);
    auto rms_supd = cmbdmv.rms_diag(1);
    auto ref_subd = std::sqrt((1+4+9+16)/4.0);
    auto ref_main = std::sqrt((25+64+64+16+36+49)/6.0);
    auto ref_supd = std::sqrt((4+9+16+25)/4.0);
    REQUIRE( std::abs((rms_subd - ref_subd)/1e-14) < 1 );
    REQUIRE( std::abs((rms_main - ref_main)/1e-14) < 1 );
    REQUIRE( std::abs((rms_supd - ref_supd)/1e-14) < 1 );
}

// TEST_CASE( "zero_out_blocks", "[ColMajBlockDiagMat]" ) {
//     auto cmbdmv = get_test_case_colmajblockdiagmat();
//     cmbdmv.zero_out_blocks();
//     for (int bi=0; bi<3; ++bi)
//         for (int ci=0; ci<2; ++ci)
//             for (int ri=0; ri<2; ++ri)
//                 REQUIRE( cmbdmv.block(bi, ri, ci) == 0.0 );
// }

// TEST_CASE( "zero_out_diags", "[ColMajBlockDiagMat]" ) {
//     auto cmbdmv = get_test_case_colmajblockdiagmat();
//     cmbdmv.zero_out_diags();
//     for (int bi=0; bi<2; ++bi)
//         for (int ci=0; ci<2; ++ci){
//             REQUIRE( cmbdm.m_view.sub(0, bi, ci) == 0.0 );
//             REQUIRE( cmbdm.m_view.sup(0, bi, ci) == 0.0 );
//         }
// }

#if defined(BLOCK_DIAG_ILU_WITH_GETRF)
TEST_CASE( "long double ilu inplace", "[ILU_inplace]" ) {
    constexpr int nblocks = 3;
    constexpr int blockw = 2;
    constexpr int ndiag = 1;
    constexpr int nsat = 1;
    constexpr int ld = 2;
    block_diag_ilu::ColMajBlockDiagMatrixView<long double> cmbdmv {nullptr, nblocks, blockw, ndiag, nsat, ld};
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<long double, blockw*blockw*nblocks + 2*blockw*(nblocks-1)> data {{
            5, 5, 3, 8,
            8, 4, 4, 4,
                6, 2, 9, 7,

                1, 2, 3, 4,
                2, 3, 4, 5 }};
    long double * blocks = data.data()
    long double * sub = data.data() + blockw*blockw*nblocks;
    long double * sup = data.data() + blockw*blockw*nblocks + blockw*(nblocks-1);
    for (int bi=0; bi<nblocks; ++bi)
        for (int ci=0; ci<blockw; ++ci){
            if (bi<nblocks-1){
                cmbdmv.sub(0, bi, ci) = sub[bi*blockw+ci];
                cmbdmv.sup(0, bi, ci) = sup[bi*blockw+ci];
            }
            for (int ri=0; ri<blockw; ++ri)
                cmbdmv.block(bi, ri, ci) = blocks[bi*blockw*blockw + ci*blockw + ri];
        }

    block_diag_ilu::ILU_inplace<long double> ilu(cmbdv);

    SECTION( "check lower correctly computed" ) {
        REQUIRE( std::abs(ilu.m_view.sub(0, 0, 0) - 1/5.L) < 1e-18L );
        REQUIRE( std::abs(ilu.m_view.sub(0, 0, 1) - 2/5.L) < 1e-18L );
        REQUIRE( std::abs(ilu.m_view.sub(0, 1, 0) - 3/8.L) < 1e-18L );
        REQUIRE( std::abs(ilu.m_view.sub(0, 1, 1) - 4/2.L) < 1e-18L );
    }
    SECTION( "check upper still perserved" ) {
        REQUIRE( ilu.m_view.sup(0, 0, 0) == 2 );
        REQUIRE( ilu.m_view.sup(0, 0, 1) == 3 );
        REQUIRE( ilu.m_view.sup(0, 1, 0) == 4 );
        REQUIRE( ilu.m_view.sup(0, 1, 1) == 5 );
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

TEST_CASE( "ilu inplace with blockw1", "[ILU_inplace]" ) {
    constexpr int nblocks = 6;
    constexpr int blockw = 1;
    constexpr int ndiag = 2;
    constexpr int nsat = 0;
    constexpr int ld = blockw;
    block_diag_ilu::ColMajBlockDiagMatrixView<double> cmbdmv {nullptr, nblocks, blockw, ndiag, nsat, ld};
    // 5 3 2 # # #
    // 5 8 # 3 # #
    // 1 # 8 4 4 #
    // # 2 4 4 # 5
    // # # 3 # 6 9
    // # # # 4 2 7
    std::array<double, ld*blockw*nblocks> blocks {{
            5, 8,
            8, 4,
            6, 7}};
    std::array<double, ld*(2*nblocks-3)> sub {{
            5, 0, 4, 0, 2,
            1, 2, 3, 4 }};
    std::array<double, ld*(2*nblocks-3)> sup {{
            3, 0, 4, 0, 9,
            2, 3, 4, 5 }};
    for (int bi=0; bi<nblocks; ++bi)
        for (int ci=0; ci<blockw; ++ci){
            for (int di=0; di<ndiag; ++di){
                const int skip = di*nblocks - (di*di + di)/2;
                if (bi<nblocks-di-1){
                    cmbdmv.sub(di, bi, ci) = sub[skip + bi*blockw + ci];
                    cmbdmv.sup(di, bi, ci) = sup[skip + bi*blockw + ci];
                }
                for (int ri=0; ri<blockw; ++ri)
                    cmbdmv.block(bi, ri, ci) = blocks[(bi*blockw + ci)*ld + ri];
            }
        }

    block_diag_ilu::ILU_inplace<double> ilu(cmbdmv);
    //  5   3   2   #   #   #
    // 1/1  8   #   3   #   #
    // 1/5  #   8   4   4   #
    //  #  1/4 1/2  4   #   5
    //  #   #  3/8  #   6   9
    //  #   #   #  1/1 1/3  7

    SECTION( "check lower correctly computed" ) {
        REQUIRE( std::abs(ilu.m_view.sub(1, 0, 0) - 1/5.) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(1, 1, 0) - 2/8.) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(1, 2, 0) - 3/8.) < 1e-15 );
        REQUIRE( std::abs(ilu.m_view.sub(1, 3, 0) - 4/4.) < 1e-15 );
    }
    SECTION( "check upper still preserved" ) {
        REQUIRE( ilu.m_view.sup(1, 0, 0) == 2 );
        REQUIRE( ilu.m_view.sup(1, 1, 0) == 3 );
        REQUIRE( ilu.m_view.sup(1, 2, 0) == 4 );
        REQUIRE( ilu.m_view.sup(1, 3, 0) == 5 );
    }

    // LUx = b
    // Ly = b
    // Ux = y
    //
    // y[0] = 65
    // y[1] = 202 - 65 = 137
    // y[2] = 11 - 65/5 = -2
    // y[3] = 65 - 137?/4 - 11/2 = 9 // <--- from here it's most likely wrong
    // y[4] = 60 - 11*3/8 = 55.875
    // y[5] = 121 - 65 - 60/3 = 36
    //
    // x[5] = 36 / 7
    // x[4] = (55.875 - 9*36) / 6
    // x[3] = (9 - 5*36) / 4
    // x[2] = (-2 - 4*9 - 4*55.875) / 8
    // x[1] = (137 - 3*9) / 8
    // x[0] = (65 - 3*137 + 2*2) / 5

    SECTION( "solve performs adequately" ) {
        std::array<double, 6> b {{65, 202, 11, 65, 60, 121}};
        std::array<double, 6> xref {{5.142857142857143,
                    }};
        std::array<double, 6> x;
        int flag = ilu.solve(b.data(), x.data());
        REQUIRE( flag == 0 );
        REQUIRE( std::abs(x[0] - xref[0]) < 5e-15 );
        REQUIRE( std::abs(x[1] - xref[1]) < 5e-15 );
        REQUIRE( std::abs(x[2] - xref[2]) < 5e-15 );
        REQUIRE( std::abs(x[3] - xref[3]) < 5e-15 );
        REQUIRE( std::abs(x[4] - xref[4]) < 5e-15 );
        REQUIRE( std::abs(x[5] - xref[5]) < 5e-15 );
    }
}
