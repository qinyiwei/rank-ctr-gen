#!/bin/sh
OUTPUT_DIR=data/toxicity/generations/
PROMPT_FILE="/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/prompts/nontoxic_prompts-10k.jsonl"

'''
python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR

python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_ppl_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR
'''
python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_ppl_gpt2-large_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR
'''
python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_dist_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR

python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_ppl_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR

python evaluation/run_toxicity_evaluation.py \
    --dataset-file $PROMPT_FILE \
    --perspective-rate-limit 1 \
    --number 1000 \
    --generation-file-name gpt2_gpt2-large_n_200_generations_classifier_toxicity_ppl_dist_rerank_generations.jsonl \
    --output-dir $OUTPUT_DIR
'''