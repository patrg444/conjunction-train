#!/bin/bash
# CPU-specific optimizations for c5.24xlarge

# Configure TensorFlow for optimal CPU performance
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=96
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1

# For Intel CPUs (c5 instances use Intel CPUs)
export TF_ENABLE_ONEDNN_OPTS=1

# Disable CUDA to ensure CPU usage
export CUDA_VISIBLE_DEVICES=""
