#!/bin/bash
export BLOCK_DIAG_ILU_LAPACK=openblas
export BLOCK_DIAG_ILU_WITH_OPENMP=1
export CPLUS_INCLUDE_PATH=${PREFIX}/include
python setup.py build
python setup.py install --single-version-externally-managed --record record.txt
