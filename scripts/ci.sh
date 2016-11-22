#!/bin/bash -xeu
if ! [[ $(python setup.py --version) =~ ^[0-9]+.* ]]; then
    exit 1
fi

cd tests
make
make clean; make DEFINES=-DBLOCK_DIAG_ILU_WITH_DGETRF EXTRA_FLAGS="-fopenmp"
make clean; make DEFINES=-DNDEBUG
make clean; make CXX=clang++-3.8

cd ..
python2.7 setup.py sdist
for PYTHON in python2.7 python3; do
    (cd dist/; $PYTHON -m pip install $1-$($PYTHON ../setup.py --version).tar.gz)
    (cd /; $PYTHON -m pytest --pyargs $1)
done

(
    cd python_prototype
    PYTHONPATH=$(pwd) python -m pytest
    export NP_INC=$(python -c "import numpy; print(numpy.get_include())")
    echo $NP_INC
    python setup.py build_ext -i
    PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
    PYTHONPATH=$(pwd) python demo.py
    rm _block_diag_ilu.so
    BLOCK_DIAG_ILU_WITH_DGETRF=1 python setup.py build_ext -i
    PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
    PYTHONPATH=$(pwd) python demo.py
    rm _block_diag_ilu.so
    BLOCK_DIAG_ILU_WITH_OPENMP=1 BLOCK_DIAG_ILU_WITH_DGETRF=0 python setup.py build_ext -i
    PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
    PYTHONPATH=$(pwd) python demo.py
    rm _block_diag_ilu.so
    BLOCK_DIAG_ILU_WITH_OPENMP=1 BLOCK_DIAG_ILU_WITH_DGETRF=1 python setup.py build_ext -i
    PYTHONPATH=$(pwd) USE_FAST_FAKELU=1 python -m pytest
    PYTHONPATH=$(pwd) python demo.py

    if [[ "$CI_BRANCH" == "master" ]]; then
        ./run_demo.sh
        mkdir -p ../deploy/public_html/branches/"${CI_BRANCH}"/
        cp run_demo.out demo_out.png  ../deploy/public_html/branches/"${CI_BRANCH}"/
    fi
)

# Make sure repo is pip installable from git-archive zip
git archive -o /tmp/archive.zip HEAD
(
    cd /
    python3 -m pip install --force-reinstall /tmp/archive.zip
    python3 -c '
from block_diag_ilu import get_include as gi
import os
assert "block_diag_ilu.pxd" in os.listdir(gi())
'
)

if grep "DO-NOT-MERGE!" -R . --exclude ci.sh; then exit 1; fi
