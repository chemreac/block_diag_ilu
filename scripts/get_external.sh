#!/bin/bash -xeu
cd external/
if [[ ! -d anyode ]]; then
   git clone -b template-Real_t --depth 1 git://github.com/bjodah/anyode.git
fi
