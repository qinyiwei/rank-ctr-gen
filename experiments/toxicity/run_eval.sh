#!/bin/sh
OUTPUT_DIR=data/toxicity/generations/
PROMPT_FILE="/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/prompts/nontoxic_prompts-10k.jsonl"
'''
python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl

python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_ppl_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl
'''
python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_ppl_gpt2-large_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl
'''
python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_dist_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl

python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_ppl_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl

python evaluation/evaluate_generations.py \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_ppl_dist_rerank_generations.jsonl \
    --num-generations 25 \
    --output-dir $OUTPUT_DIR \
    --src $PROMPT_FILE \
    --ppl-model gpt2-xl
'''