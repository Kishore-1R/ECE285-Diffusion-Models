#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --batch_size 1 --num_samples 1 
--diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear 
--num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True 
--use_fp16 True --use_scale_shift_norm True"
python src/image_sample.py $MODEL_FLAGS --model_path ../codes/models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS