#!/usr/bin/env bash
#
# Usage:
#
#  $ ./scripts/build-conda-recipe-using-docker.sh conda/recipe
# or
#  $ ./scripts/build-conda-recipe-using-docker.sh dist/conda-recipe-0.11.4
#
anfilte-build . $1 dist/
