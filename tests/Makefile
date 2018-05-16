CXX ?= g++
LIBS ?= -llapack -lblas
CXXFLAGS ?= -std=c++11 -Wall -Wextra -Werror -pedantic -DBLOCK_DIAG_ILU_UNIT_TEST -g -ggdb -O0
CXXFLAGS += $(EXTRA_FLAGS)
INCLUDE ?= -I../include -I../external/anyode/include
DEFINES ?=

COMMON_DEP := ../include/block_diag_ilu.hpp catch.hpp
#-D__restrict__=""

.PHONY: test clean

test: test_block_diag test_banded test_dense
	./test_block_diag --abortx 1
	./test_banded --abortx 1
	./test_dense --abortx 1

clean:
	rm -f catch.hpp
	rm -f test_block_diag

test_block_diag_omp: test_block_diag.cpp $(COMMON_DEP)
	$(CXX) $(CXXFLAGS) -fopenmp $(INCLUDE) $(DEFINES) -DBLOCK_DIAG_ILU_WITH_OPENMP=1 -o $@ $< $(LIBS)

catch.hpp: catch.hpp.bz2
	bunzip2 -k -f $<

test_%: test_%.cpp $(COMMON_DEP)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(DEFINES) -o $@ $< $(LIBS)
