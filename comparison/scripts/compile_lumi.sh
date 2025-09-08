#!/bin/bash

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3

CC -x hip -shared -O3 -std=c++17 \
    -fPIC \
    -DNDEBUG \
    -o cpp/liblbm.so cpp/lbm.cpp
