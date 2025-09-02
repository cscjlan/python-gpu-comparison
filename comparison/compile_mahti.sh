#!/bin/bash

ml gcc/10.4.0
ml cuda/12.6.1

export HOP_ROOT=${PWD}/include/hop
export HOP_FLAGS="-I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA"

nvcc -x cu $HOP_FLAGS -shared -O3 -std=c++20 \
    --expt-relaxed-constexpr \
    --extended-lambda \
    --forward-unknown-to-host-compiler \
    -fPIC \
    -o lbm.so lbm.cpp
