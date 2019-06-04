#!/bin/bash
export BLOCK_DIAG_ILU_LAPACK=lapack,blas
export BLOCK_DIAG_ILU_WITH_OPENMP=0
python -m pip install --no-deps --ignore-installed . -vv
