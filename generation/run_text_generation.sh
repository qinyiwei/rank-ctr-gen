#!/bin/sh
n=200
OUTPUT_DIR=data/toxicity/generation_candidates/
MODEL=gpt2-large
MODELTYPE=gpt2

python generation/run_text_generation.py \
    --dataset-file /projects/tir4/users/yiweiq/toxicity/dataset/DExperts/prompts/nontoxic_prompts-10k.jsonl \
    --n $n \
    --model-type $MODELTYPE \
    --model $MODEL \
    --p 0.9 \
    --batch-size 128 \
    --number 1000 \
    $OUTPUT_DIR