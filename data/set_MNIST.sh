#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python3.6 ${SCRIPT_DIR}/set_MNIST.py \
    --output1_dir "MNIST_org" \
    --output2_dir "MNIST"
