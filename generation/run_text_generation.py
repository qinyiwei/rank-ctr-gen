import pickle
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import os

from generation import gpt2
from utils.utils import ensure_dir

ALLOWED_MODELS = ['gpt2']


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--n', default=100, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--k', default=None, type=int, help='Hyperparameter for truncation of p_base')
@click.option('--p', default=0.9, type=float, help='Hyperparameter for nucleus sampling')
@click.option('--number', type=int, default=None)

def main(output_dir: str, dataset_file: Optional[str], model: str, model_type: str, n: int, max_tokens: int, batch_size: int, 
         k: int, p: float, number: int):
    # Load prompts from dataset file
    assert dataset_file.endswith('.jsonl')
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])['text']

    print('Prompts:', '\n', prompts)

    if number is not None:
        dataset = dataset[:number]
        prompts = prompts[:number]

    # Create output files
    output_dir = Path(output_dir)
    model_name = model + "_n_{}".format(n)
    generations_file = output_dir / f'{model_type}_{model_name}_generations.jsonl'
    print("generation file:{}".format(generations_file))
    assert not os.path.exists(generations_file)   # don't overwrite generations!
    ensure_dir(output_dir)

    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            p=p,
            batch_size=batch_size,
            model_name_or_path=model,
            out_file=generations_file
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)

    print("save file to {}".format(generations_file))
if __name__ == '__main__':
    main()
