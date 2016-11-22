#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from block_diag_ilu import get_include
from block_diag_ilu.tests.test_block_diag_ilu import test_ilu_solve__1

test_ilu_solve__1()  # rest of test suite run as CI

assert 'block_diag_ilu.hpp' in os.listdir(get_include())
path = os.path.join(get_include(), 'block_diag_ilu.hpp')
assert open(path, 'rt').readline().startswith('#pragma once')
