#!/bin/sh
OUTPUT_DIR=data/toxicity/generations/
PROMPT_FILE="/projects/tir4/users/yiweiq/toxicity/dataset/DExperts/prompts/nontoxic_prompts-10k.jsonl"
'''
python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics classifier_toxicity\
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25 \
    --src $PROMPT_FILE

python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics ppl \
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25\
    --ppl-model gpt2-xl \
    --src $PROMPT_FILE
'''
python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics ppl \
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25\
    --ppl-model gpt2-large \
    --src $PROMPT_FILE
'''
python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics dist \
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25\
    --src $PROMPT_FILE

python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics classifier_toxicity ppl \
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25\
    --ppl-model gpt2-xl \
    --src $PROMPT_FILE

python qaware_decode/rerank.py \
    data/toxicity/generation_candidates/gpt2_gpt2-large_n_200_generations.jsonl \
    -n 200 \
    --qe-metrics classifier_toxicity ppl dist \
    --weights-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_epoch_10.pt \
    --meta-path models/pplm_classifiers/toxicity_classifierhead_1280/toxic_classifier_head_meta.json \
    --batch-size 32 \
    --output-dir $OUTPUT_DIR \
    --num-generations 25\
    --ppl-model gpt2-xl \
    --src $PROMPT_FILE
'''