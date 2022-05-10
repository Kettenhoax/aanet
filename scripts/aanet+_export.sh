#!/usr/bin/env bash

# Inference on KITTI 2015 test set for submission
CUDA_VISIBLE_DEVICES=0 python3 export.py \
--pretrained_aanet pretrained/aanet+_kitti15-2075aea1.pth \
--img_height 768 \
--img_width 1024 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--no_intermediate_supervision \
