#!/bin/bash
export BLOCK_DIAG_ILU_LAPACK=openblas
export BLOCK_DIAG_ILU_WITH_OPENMP=1
export CPLUS_INCLUDE_PATH=${PREFIX}/include
export LIBRARY_PATH=${PREFIX}/lib
export CC=clang-6.0
export CXX=clang++
export CFLAGS="-stdlib=libc++ -std=c++17"
${PYTHON} -m pip install --no-deps --ignore-installed .
