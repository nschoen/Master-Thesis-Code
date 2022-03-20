#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot "../datasets/eh_mmm_custom_flanges/data-obj-folder-structured" \
--name mmm_custom_flanges \
--ncf 64 128 256 256 \
--pool_res 15000 9000 6000 3000 \
--ninput_edges 20000 \
--norm group \
--resblocks 1 \
--export_folder meshes \