"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import click
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qaware_decode.utils import load_generations
import json

def get_loss(model, tokenizer, device, texts):
    loss_fct = torch.nn.NLLLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    lsm = torch.nn.LogSoftmax(dim=1)

    def query_model_batch_gen(model, tokenizer, device, texts):
        inputs = tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        output = model(**inputs,labels=inputs['input_ids'])
        #print("lm_loss:{}".format(output['loss']))
        return inputs, output['logits']

    inputs, logits = query_model_batch_gen(model, tokenizer, device, texts)
    tgt_tokens = inputs['input_ids']
    logits = logits[..., :-1, :].contiguous()
    tgt_tokens = tgt_tokens[..., 1:].contiguous()
    tgt_len = inputs['attention_mask'].sum(dim=1)-1

    logits = logits.view(-1, model.config.vocab_size)
    loss = loss_fct(lsm(logits), tgt_tokens.view(-1))
    loss = loss.view(tgt_tokens.shape[0], -1)
    #prob = -loss.sum(dim=1) 
    #prob = prob/tgt_len
    loss = loss.sum(dim=1) 

    return loss,tgt_len

def conditional_perplexity(generations, srcs, model, tokenizer, device='cuda'):
    perplexities = []
    ct = 0
    # for every prompt
    for i in tqdm(range(len(generations)), total=len(generations), desc='Evaluating fluency'):
        prompt = srcs[i]
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
        # for every generation conditioned on the prompt
        batch_gen = [prompt+gen for gen in generations[i]]

        full_loss, tgt_len = get_loss(model, tokenizer, device, batch_gen) 
        losses = (full_loss - prompt_loss) / (tgt_len - (prompt_input_ids.shape[1]-1))
        
        ppl = [math.exp(loss.item()) for loss in losses]
        #if ppl < 1e4:   # for sanity
        perplexities.extend([p for p in ppl if (p < 1e4 and p != 0)])
    return np.nanmean(perplexities)

def distinctness(generations):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for one_prompt in tqdm(generations, total=len(generations), desc='Evaluating diversity'):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in one_prompt:
            o = gen.split(' ')
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


@click.command()
@click.option('--output-dir', required=False)
@click.option('--src', required=False)
@click.option('--generation-file-name', required=False, type=str, help='a jsonl file with generations')
@click.option('--ppl-model', default="gpt2-xl", type=str, help="model used to caculate ppl")
@click.option('--num-generations', default=25, type=int, help="number of generations for one prompt")

def main(output_dir: str, src: str, generation_file_name:str, ppl_model:str, num_generations:int):
    
    # Create output files
    generation_file = os.path.join(output_dir, generation_file_name)
    assert os.path.exists(generation_file)
    hyps = load_generations(generation_file, num_generations)

    # calculate diversity
    dist1, dist2, dist3 = distinctness(hyps)
    print("dist1-{},dist2-{},dist3-{}".format(dist1,dist2,dist3))

    # write output results
    output_dir = Path(output_dir)
    with open(output_dir / generation_file_name.replace('_generations.jsonl','_eval_results.txt'), 'w') as fo:
        for i, dist_n in enumerate([dist1, dist2, dist3]):
            fo.write(f'dist-{i+1} = {dist_n}\n')

    #load src file
    srcs = None
    if src is not None:
        srcs = []
        with open(src, 'r', encoding='utf8') as fh:
            for line in fh:
                line_json = json.loads(line)
                srcs.append(line_json['prompt']['text'])
        srcs = srcs[:len(hyps)]

    # calculate fluency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = AutoModelForCausalLM.from_pretrained(ppl_model).to(device)
    eval_tokenizer = AutoTokenizer.from_pretrained(ppl_model)
    if "gpt2" in ppl_model:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    torch.cuda.empty_cache()
    with torch.no_grad():
        ppl = conditional_perplexity(hyps, srcs, eval_model, eval_tokenizer, device=device)
    print("ppl:{}".format(ppl))

    # write output results
    with open(output_dir / generation_file_name.replace('_generations.jsonl','_eval_results.txt'), 'a') as fo:
        fo.write(f'perplexity = {ppl}')
    print("write output to {}/eval_results.txt".format(output_dir))

if __name__ == '__main__':
    main()