from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([Extension(
        '_block_diag_ilu', ['_block_diag_ilu.pyx'],
#        define_macros=[('DEBUG_VERBOSE', None)]
    )])
)
