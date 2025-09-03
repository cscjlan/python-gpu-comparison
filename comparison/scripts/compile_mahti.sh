#!/bin/bash

ml gcc/11.2.0
ml cuda/11.5.0

export HOP_ROOT=${PWD}/cpp/include/hop
export HOP_FLAGS="-I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA"

nvcc -x cu $HOP_FLAGS -shared -O3 -std=c++17 \
    --expt-relaxed-constexpr \
    --extended-lambda \
    --forward-unknown-to-host-compiler \
    -fPIC \
    -o cpp/lbm.so cpp/lbm.cpp
