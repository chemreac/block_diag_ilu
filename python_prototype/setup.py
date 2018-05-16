import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np


if __name__ == '__main__':
    kwargs = {
        'include_dirs': ['../include', np.get_include(), '../external/anyode/include'],
        'extra_compile_args': ['-std=c++11', '-DBLOCK_DIAG_ILU_UNIT_TEST'],
        'extra_link_args': [],
        'language': 'c++'
    }
    if os.environ.get('BLOCK_DIAG_ILU_WITH_GETRF', '0') == '1':
        kwargs['extra_compile_args'] += ['-DBLOCK_DIAG_ILU_WITH_GETRF']
    else:
        kwargs['libraries'] = ['lapack']

    if os.environ.get('BLOCK_DIAG_ILU_WITH_OPENMP', '0') == '1':
        kwargs['extra_compile_args'] += ['-fopenmp', '-DBLOCK_DIAG_ILU_WITH_OPENMP']
        kwargs['extra_link_args'] += ['-fopenmp']

    ext_modules = [
        Extension('_block_diag_ilu', ['_block_diag_ilu.pyx'], **kwargs)
    ]

    setup(ext_modules = cythonize(ext_modules))
