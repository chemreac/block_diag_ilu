#!/bin/bash -xeu
cd tests
make
make clean
make DEFINES=-DWITH_BLOCK_DIAG_ILU_DGETRF EXTRA_FLAGS="-Wfatal-errors -fopenmp"
cd ../python_prototype
PYTHONPATH=$(pwd) py.test
export NP_INC=$(python -c "import numpy; print(numpy.get_include())")
echo $NP_INC
python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 py.test
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_DGETRF=1 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 py.test
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_OPENMP=1 WITH_BLOCK_DIAG_ILU_DGETRF=0 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 py.test
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_OPENMP=1 WITH_BLOCK_DIAG_ILU_DGETRF=1 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 py.test
PYTHONPATH=$(pwd) python demo.py