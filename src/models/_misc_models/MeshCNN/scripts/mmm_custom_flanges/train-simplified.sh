#!/usr/bin/env bash

## run the training
python train.py \
--dataroot "../datasets/eh_mmm_custom_flanges/data-obj-folder-structured-simplified" \
--name mmm_custom_flanges_simplified \
--ncf 64 128 256 256 \
--pool_res 3500 2200 1400 800 \
--ninput_edges 4500 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--gpu_ids 0 \
## --gpu_ids 0 because the workload manage will take care about which gpu device is available which is automatically
## available under device id 0