#!/bin/bash -e
# Usage
#   $ ./scripts/run_tests.sh
# or
#   $ ./scripts/run_tests.sh --cov block_diag_ilu --cov-report html
${PYTHON:-python3} setup.py build_ext -i
${PYTHON:-python3} -m pytest --doctest-modules "$@"  # --pep8 --flakes
${PYTHON:-python3} -m doctest README.rst
