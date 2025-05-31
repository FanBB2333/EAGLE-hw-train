#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
CUDA_VISIBLE_DEVICES='0' python fzy/process_ds.py \
