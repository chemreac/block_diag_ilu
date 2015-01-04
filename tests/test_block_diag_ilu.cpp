// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>

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
        REQUIRE( abs(x[0] - xref[0]) < 1e-15 );
        REQUIRE( abs(x[1] - xref[1]) < 1e-15 );
        REQUIRE( abs(x[2] - xref[2]) < 1e-15 );
        REQUIRE( abs(x[3] - xref[3]) < 1e-15 );
        REQUIRE( abs(x[4] - xref[4]) < 1e-15 );
        REQUIRE( abs(x[5] - xref[5]) < 1e-15 );
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
