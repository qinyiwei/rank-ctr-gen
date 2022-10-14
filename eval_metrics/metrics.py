"""
evaluate generated output for diversity (dist-n) and fluency (perplexity according to GPT2-XL)
"""

import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    loss = loss.sum(dim=1) 

    return loss,tgt_len

class Perplexity:
    def __init__(self, ppl_model):
        print("model name of ppl:{}".format(ppl_model))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(ppl_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ppl_model)
        self.batch_size = 16
        if "gpt2" in ppl_model:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    '''
    def __call__(self, generations, scrs):
        perplexities = []
        n = int(len(generations)/len(scrs))
        for i in tqdm(range(len(scrs))):
            prompt = scrs[i]
            prompt_input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            prompt_loss = self.model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            for j in range(n):
                gen = generations[i*n + j]
                full_input_ids = self.tokenizer.encode(prompt+gen, return_tensors='pt').to(self.device)
                full_loss = self.model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
                loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
                ppl = math.exp(loss.item())
                ppl = 1e4 if math.isnan(ppl) else min(ppl,1e4)
                perplexities.append(ppl)
        return np.array(perplexities)
    '''
    
    def __call__(self, generations, scrs):
        perplexities = []
        n = int(len(generations)/len(scrs))
        # for every prompt
        for i in tqdm(range(len(scrs))):
            prompt = scrs[i]
            prompt_input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            prompt_loss = self.model(prompt_input_ids, labels=prompt_input_ids)[0] * (prompt_input_ids.shape[1]-1)
            
            for j in np.arange(0, n/self.batch_size, 1):
                j = int(j)
                gen_batch = generations[i*n + j*self.batch_size: min((i+1)*n, i*n + (j+1)*self.batch_size)]
                # for every generation conditioned on the prompt
                batch_gen = [prompt+gen for gen in gen_batch]

                full_loss, tgt_len = get_loss(self.model, self.tokenizer, self.device, batch_gen) 
                losses = (full_loss - prompt_loss) / (tgt_len - (prompt_input_ids.shape[1]-1))
                
                ppl = [math.exp(loss.item()) for loss in losses]
                #if ppl < 1e4:   # for sanity
                perplexities.extend([min(p,1e4) if p != 0 else 1e4 for p in ppl ])

        return np.array(perplexities)

def distinctness(generations, srcs):
    dist = {
        'dist1': [],
        'dist2': [],
        'dist3': [],
    }
    # calculate dist1, dist2, dist3 across generations for every prompt
    for gen in tqdm(generations):
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0

        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i+1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist['dist1'].append(1 - len(unigrams) / total_words)
        dist['dist2'].append(1 - len(bigrams) / total_words)
        dist['dist3'].append(1 - len(trigrams) / total_words)

    dist['dist1'] = np.array(dist['dist1'])
    dist['dist2'] = np.array(dist['dist2'])
    dist['dist3'] = np.array(dist['dist3'])
    return dist

