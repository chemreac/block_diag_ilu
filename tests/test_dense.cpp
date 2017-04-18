// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include "block_diag_ilu/dense.hpp"
#include <array>
#include <cmath>

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
    REQUIRE( cmdv.m_ld == 6 );
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

    REQUIRE( cmdv.sat(-1, 0, 0) == 8 );
    REQUIRE( cmdv.sat(-1, 0, 1) == 3 );
    REQUIRE( cmdv.sat(-2, 0, 0) == 9 );
    REQUIRE( cmdv.sat(-2, 0, 1) == 4 );
    REQUIRE( cmdv.sat(-2, 1, 0) == 9 );
    REQUIRE( cmdv.sat(-2, 1, 1) == 4 );

    REQUIRE( cmdv.sat(1, 0, 0) == 4 );
    REQUIRE( cmdv.sat(1, 0, 1) == 8 );
    REQUIRE( cmdv.sat(2, 0, 0) == 3 );
    REQUIRE( cmdv.sat(2, 0, 1) == 7 );
    REQUIRE( cmdv.sat(2, 1, 0) == 3 );
    REQUIRE( cmdv.sat(2, 1, 1) == 7 );

    std::array<double, 36> rarr {{
            1, 2, 3, 0, 4, 0,
                5, 6, 0, 7, 0, 8,
                9, 0, 1, 2, 3, 0,
                0, 4, 5, 6, 0, 7,
                8, 0, 9, 0, 1, 2,
                0, 3, 0, 4, 5, 6
                }};
    auto rmdv = block_diag_ilu::DenseView<double, false>((double*)rarr.data(), nblocks, blockw, ndiag);  // row maj
    REQUIRE( rmdv.m_ld == 6 );
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