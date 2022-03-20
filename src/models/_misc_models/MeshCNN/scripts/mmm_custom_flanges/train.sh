#!/usr/bin/env bash

## run the training
python train.py \
--dataroot "../datasets/eh_mmm_custom_flanges/data-obj-folder-structured" \
--name mmm_custom_flanges \
--ncf 64 128 256 256 \
--pool_res 15000 9000 6000 3000 \
--ninput_edges 20000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \