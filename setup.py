#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
from setuptools import setup, Extension


pkg_name = 'block_diag_ilu'

BLOCK_DIAG_ILU_RELEASE_VERSION = os.environ.get(
    'BLOCK_DIAG_ILU_RELEASE_VERSION', '')  # v*
WITH_BLOCK_DIAG_ILU_DGETRF = os.environ.get('WITH_BLOCK_DIAG_ILU_DGETRF', '0') == '1'
WITH_BLOCK_DIAG_ILU_OPENMP = os.environ.get('WITH_BLOCK_DIAG_ILU_OPENMP', '0') == '1'

# Cythonize .pyx file if it exists (not in source distribution)
ext_modules = []


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)

if len(sys.argv) > 1 and '--help' not in sys.argv[1:] and sys.argv[1] not in (
        '--help-commands', 'egg_info', 'clean', '--version'):
    USE_CYTHON = os.path.exists(_path_under_setup('block_diag_ilu',
                                                  '_block_diag_ilu.pyx'))
    ext = '.pyx' if USE_CYTHON else '.cpp'
    sources = ['block_diag_ilu/_block_diag_ilu'+ext]
    ext_modules = [
        Extension('block_diag_ilu._block_diag_ilu', sources)
    ]
    if USE_CYTHON:
        from Cython.Build import cythonize
        ext_modules = cythonize(ext_modules, include_path=['./include'], gdb_debug=True)
    macros = [('BLOCK_DIAG_ILU_PY', None)]
    if WITH_BLOCK_DIAG_ILU_DGETRF:
        macros.append(('WITH_BLOCK_DIAG_ILU_DGETRF', None))
    if WITH_BLOCK_DIAG_ILU_OPENMP:
        macros.append(('WITH_BLOCK_DIAG_ILU_OPENMP', None))
    ext_modules[0].language = 'c++'
    ext_modules[0].extra_compile_args = ['-std=c++11']
    ext_modules[0].include_dirs = [_path_under_setup('include')]
    ext_modules[0].define_macros += macros
    ext_modules[0].libraries += [os.environ.get('LLAPACK', 'lapack')]

# http://conda.pydata.org/docs/build.html#environment-variables-set-during-the-build-process
if os.environ.get('CONDA_BUILD', '0') == '1':
    try:
        BLOCK_DIAG_ILU_RELEASE_VERSION = 'v' + open(
            '__conda_version__.txt', 'rt').readline().rstrip()
    except IOError:
        pass

release_py_path = _path_under_setup(pkg_name, '_release.py')

if len(BLOCK_DIAG_ILU_RELEASE_VERSION) > 0:
    if BLOCK_DIAG_ILU_RELEASE_VERSION[0] == 'v':
        TAGGED_RELEASE = True
        __version__ = BLOCK_DIAG_ILU_RELEASE_VERSION[1:]
    else:
        raise ValueError("Ill formated version")
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(open(release_py_path).read())


classifiers = [
    "Development Status :: 3 - Alpha",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

tests = [
    'block_diag_ilu.tests',
]

with open(_path_under_setup(pkg_name, '__init__.py'), 'rt') as f:
    short_description = f.read().split('"""')[1].split('\n')[1]
assert 10 < len(short_description) < 255
long_description = open(_path_under_setup('README.rst')).read()
assert len(long_description) > 100

setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=short_description,
    long_description=long_description,
    classifiers=classifiers,
    author='Björn Dahlgren',
    author_email='bjodah@DELETEMEgmail.com',
    url='https://github.com/bjodah/' + pkg_name,
    license='BSD',
    packages=[pkg_name] + tests,
    package_data={pkg_name: ['include/*.hpp']},
    setup_requires=['numpy'],
    install_requires=['numpy'],
    ext_modules=ext_modules,
)

# Same commit should generate different sdist
# depending on tagged version (set $BLOCK_DIAG_ILU_RELEASE_VERSION)
# e.g.:  $ BLOCK_DIAG_ILU_RELEASE_VERSION=v1.2.3 python setup.py sdist
# this will ensure source distributions contain the correct version
if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__', release_py_path)
