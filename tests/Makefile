CXX ?= g++
LIBS ?= -llapack
CXXFLAGS ?= -std=c++11 -Wall -Wextra -Werror -pedantic -DBLOCK_DIAG_ILU_UNIT_TEST -g -ggdb -O0
CXXFLAGS += $(EXTRA_FLAGS)
INCLUDE ?= -I../include
DEFINES ?=

#-D__restrict__=""

.PHONY: test clean

test: test_block_diag_ilu
	./$< --abortx 1

clean:
	rm catch.hpp
	rm test_block_diag_ilu

test_block_diag_ilu: test_block_diag_ilu.cpp ../include/block_diag_ilu.hpp catch.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(DEFINES) -o $@ $< $(LIBS)

catch.hpp: catch.hpp.bz2
	bunzip2 -k -f $<
