// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>

TEST_CASE( "rowpiv2rowbycol" "[ILU]" ) {
    const std::array<int, 5> piv {3, 4, 5, 4, 5};
    std::array<int, 5> rowbycol;
    block_diag_ilu::rowpiv2rowbycol(5, &piv[0], &rowbycol[0]);
    REQUIRE( rowbycol[0] == 2 );
    REQUIRE( rowbycol[1] == 3 );
    REQUIRE( rowbycol[2] == 4 );
    REQUIRE( rowbycol[3] == 1 );
    REQUIRE( rowbycol[4] == 0 );
}

TEST_CASE( "rowbycol2colbyrow" "[ILU]" ) {
    const std::array<int, 5> rowbycol {2, 3, 4, 1, 0};
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
    std::array<double, blockw*blockw*nblocks> block {
        5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7};
    std::array<double, blockw*(nblocks-1)> sub {1, 2, 3, 4};
    const std::array<double, blockw*(nblocks-1)> sup {2, 3, 4, 5};
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
        std::array<double, 6> b {65, 202, 11, 65, 60, 121};
        std::array<double, 6> xref {-31.47775, 53.42125, 31.0625,
                -43.36875, -19.25625, 19.5875};
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( abs((x[0] - xref[0])/1e-14) < 1.0 );
        REQUIRE( abs((x[1] - xref[1])/1e-14) < 1.0 );
        REQUIRE( abs((x[2] - xref[2])/1e-14) < 1.0 );
        REQUIRE( abs((x[3] - xref[3])/1e-14) < 1.0 );
        REQUIRE( abs((x[4] - xref[4])/1e-14) < 1.0 );
        REQUIRE( abs((x[5] - xref[5])/1e-14) < 1.0 );
    }
}

TEST_CASE( "_get_test_m4 in test_fakelu.py", "[ILU]" ) {

    // this is _get_test_m4 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 2;
    std::array<double, blockw*blockw*nblocks> block {
        -17, 37, 63, 13, 11, -42, 72, 24, 72, 14, -13, -57};
    std::array<double, 6> sub {.1, .2, -.1, .08, .03, -.1};
    const std::array<double, 6> sup {.2, .3, -.1, .2, .02, .03};
    block_diag_ilu::ILU ilu(block.data(), sub.data(), sup.data(),
                            nblocks, blockw, ndiag);

    REQUIRE( ilu.nblocks == nblocks );
    REQUIRE( ilu.blockw == blockw );
    REQUIRE( ilu.ndiag == ndiag );

    SECTION( "check lower correctly computed" ) {
        REQUIRE( abs(ilu.sub_get(0, 0, 0) - .1/37 ) < 1e-15 );
        REQUIRE( abs(ilu.sub_get(0, 0, 1) - .2/(63+17/37.*13) ) < 1e-15 );
        REQUIRE( abs(ilu.sub_get(0, 1, 0) - .1/42 ) < 1e-15 );
        REQUIRE( abs(ilu.sub_get(0, 1, 1) - .08/(72+11/42.*24) ) < 1e-15 );
        REQUIRE( abs(ilu.sub_get(1, 0, 0) - .03/37 ) < 1e-15 );
        REQUIRE( abs(ilu.sub_get(1, 0, 1) - -.1/(63+17/37.*13) ) < 1e-15 );
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
        std::array<double, 6> b {-62, 207, 11, -14, 25, -167};
        std::array<double, 6> xref {5.42317616680147374e+00,
                4.78588898186963929e-01,
                4.00565700557765081e-01,
                8.73749367816923223e-02,
                9.14791409109598774e-01,
                3.15378902934640371e+00
                };
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( abs(x[0] - xref[0]) < 1e-15 );
        REQUIRE( abs(x[1] - xref[1]) < 1e-15 );
        REQUIRE( abs(x[2] - xref[2]) < 1e-15 );
        REQUIRE( abs(x[3] - xref[3]) < 1e-15 );
        REQUIRE( abs(x[4] - xref[4]) < 1e-15 );
        REQUIRE( abs(x[5] - xref[5]) < 1e-15 );
    }
}

TEST_CASE( "addressing", "[BlockDiagMat]" ) {
    block_diag_ilu::BlockDiagMat m {3, 2, 1};
    std::array<double, 2*2*3+2*2+2*2> d {
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20
    };
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
    std::array<double, 2*2*3+2*2*(2+1)> d {
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            91, 92,
            17, 18, 19, 20,
            81, 82
    };
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
    std::array<double, 2*2*3+2*2+2*2> d {
        1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
            17, 18, 19, 20
    };
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
    std::array<double, blockw*blockw*nblocks+2*blockw*(nblocks-1)> d {
        5, 5, 3, 8, 8, 4, 4, 4, 6, 2, 9, 7,
            1, 2, 3, 4,
            2, 3, 4, 5
    };
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
        std::array<double, 6> b {65, 202, 11, 65, 60, 121};
        std::array<double, 6> xref {-31.47775, 53.42125, 31.0625,
                -43.36875, -19.25625, 19.5875};
        std::array<double, 6> x;
        ilu.solve(b.data(), x.data());
        REQUIRE( abs((x[0] - xref[0])/1e-14) < 1 );
        REQUIRE( abs((x[1] - xref[1])/1e-14) < 1 );
        REQUIRE( abs((x[2] - xref[2])/1e-14) < 1 );
        REQUIRE( abs((x[3] - xref[3])/1e-14) < 1 );
        REQUIRE( abs((x[4] - xref[4])/1e-14) < 1 );
        REQUIRE( abs((x[5] - xref[5])/1e-14) < 1 );
    }
}

TEST_CASE( "dot_vec", "[BlockDiagMat]" ) {
    // this is _get_test_m2 in test_fakelu.py
    const int blockw = 2;
    const int nblocks = 3;
    const int ndiag = 1;
    block_diag_ilu::BlockDiagMat bdm {nblocks, blockw, ndiag};
    std::array<double, blockw*blockw*nblocks+2*blockw*(nblocks-1)> d {
        5, 5, 3, 8,
            8, 4, 4, 4,
            6, 2, 9, 7,
            1, 2, 3, 4,
            2, 3, 4, 5
    };
    for (size_t i=0; i<d.size(); ++i)
        bdm.data[i] = d[i];
    const std::array<double, nblocks*blockw> x {3, 2, 6, 2, 5, 4};
    const std::array<double, nblocks*blockw> bref {
        5*3 + 3*2 + 2*6,
            5*3 + 8*2 + 3*2,
            8*6 + 4*2 + 4*5 + 1*3,
            4*6 + 4*2 + 5*4 + 2*2,
            6*5 + 9*4 + 3*6,
            2*5 + 7*4 + 4*2};
    std::array<double, nblocks*blockw> b;
    bdm.dot_vec(x.data(), b.data());
    REQUIRE( abs(b[0] - bref[0]) < 1e-15 );
    REQUIRE( abs(b[1] - bref[1]) < 1e-15 );
    REQUIRE( abs(b[2] - bref[2]) < 1e-15 );
    REQUIRE( abs(b[3] - bref[3]) < 1e-15 );
    REQUIRE( abs(b[4] - bref[4]) < 1e-15 );
    REQUIRE( abs(b[5] - bref[5]) < 1e-15 );
}
