#!/bin/bash -xeu
cd tests
make
make clean; make DEFINES=-DWITH_BLOCK_DIAG_ILU_DGETRF EXTRA_FLAGS="-fopenmp"
make clean; make DEFINES=-DNDEBUG
make clean; make CXX=clang++-3.7

cd ..
python2.7 setup.py sdist
for PYTHON in python2.7 python3.4; do
    (cd dist/; $PYTHON -m pip install $1-$($PYTHON ../setup.py --version).tar.gz)
    (cd /; $PYTHON -m pytest --pyargs $1)
done

cd python_prototype
PYTHONPATH=$(pwd) python -m pytest
export NP_INC=$(python -c "import numpy; print(numpy.get_include())")
echo $NP_INC
python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_DGETRF=1 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_OPENMP=1 WITH_BLOCK_DIAG_ILU_DGETRF=0 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
PYTHONPATH=$(pwd) python demo.py
rm _block_diag_ilu.so
WITH_BLOCK_DIAG_ILU_OPENMP=1 WITH_BLOCK_DIAG_ILU_DGETRF=1 python setup.py build_ext -i
PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
PYTHONPATH=$(pwd) python demo.py

if [[ "$CI_BRANCH" == "master" ]]; then
    ./run_demo.sh
    mkdir -p ../deploy/public_html/branches/"${CI_BRANCH}"/
    cp run_demo.out demo_out.png  ../deploy/public_html/branches/"${CI_BRANCH}"/
fi

! grep "DO-NOT-MERGE!" -R . --exclude ci.sh
