CXX ?= g++
LIBS ?= -llapack
CXXFLAGS ?= -std=c++14 -Wall -Wextra -Werror -pedantic -DBLOCK_DIAG_ILU_UNIT_TEST -g -ggdb -O0 -D_GLIBCXX_DEBUG
CXXFLAGS += $(EXTRA_FLAGS)
INCLUDE ?= -I../include
DEFINES ?=

#-D__restrict__=""

.PHONY: test clean

test: test_block_diag_ilu test_block_diag_ilu_omp
	./test_block_diag_ilu --abortx 1
	BLOCK_DIAG_ILU_NUM_THREADS=2 ./test_block_diag_ilu_omp --abortx 1

clean:
	rm catch.hpp
	rm test_block_diag_ilu

test_block_diag_ilu: test_block_diag_ilu.cpp ../include/block_diag_ilu.hpp catch.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(DEFINES) -o $@ $< $(LIBS)

test_block_diag_ilu_omp: test_block_diag_ilu.cpp ../include/block_diag_ilu.hpp catch.hpp
	$(CXX) $(CXXFLAGS) -fopenmp $(INCLUDE) $(DEFINES) -DBLOCK_DIAG_ILU_WITH_OPENMP=1 -o $@ $< $(LIBS)


catch.hpp: catch.hpp.bz2
	bunzip2 -k -f $<
