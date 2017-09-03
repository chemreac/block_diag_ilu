// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "block_diag_ilu.hpp"
#include <array>
#include <cmath>

TEST_CASE( "block_sub_sup", "[BlockDenseMatrix]" ) {
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
    const int ld = 6;
    auto cmdv = block_diag_ilu::BlockDenseMatrix<double>((double*)arr.data(), nblocks, blockw, ndiag, ld, true);  // col maj
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

    REQUIRE( cmdv.bot(0, 0, 0) == 8 );
    REQUIRE( cmdv.bot(0, 0, 1) == 3 );
    REQUIRE( cmdv.bot(1, 0, 0) == 9 );
    REQUIRE( cmdv.bot(1, 0, 1) == 4 );
    REQUIRE( cmdv.bot(1, 1, 0) == 9 );
    REQUIRE( cmdv.bot(1, 1, 1) == 4 );

    REQUIRE( cmdv.top(0, 0, 0) == 4 );
    REQUIRE( cmdv.top(0, 0, 1) == 8 );
    REQUIRE( cmdv.top(1, 0, 0) == 3 );
    REQUIRE( cmdv.top(1, 0, 1) == 7 );
    REQUIRE( cmdv.top(1, 1, 0) == 3 );
    REQUIRE( cmdv.top(1, 1, 1) == 7 );
}
