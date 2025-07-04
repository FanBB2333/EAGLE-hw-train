#!/bin/bash
export PYTHONPATH=$(pwd):/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind
CUDA_VISIBLE_DEVICES='0' python fzy/compose_data.py \
