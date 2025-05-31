#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
CUDA_VISIBLE_DEVICES='0' python eval/process_ds.py \
