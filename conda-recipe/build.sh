#!/bin/bash
LLAPACK=openblas ${PYTHON} setup.py build
${PYTHON} setup.py install
