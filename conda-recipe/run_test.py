#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from block_diag_ilu import get_include
assert os.listdir(get_include())[0] == 'block_diag_ilu.hpp'
path = os.path.join(get_include(), 'block_diag_ilu.hpp')
assert open(path, 'rt').readline().startswith('#pragma once')
