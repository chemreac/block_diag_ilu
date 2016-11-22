#!/bin/bash
BLOCK_DIAG_ILU_LAPACK=openblas CPLUS_INCLUDE_PATH=${PREFIX}/include python setup.py build
python setup.py install
