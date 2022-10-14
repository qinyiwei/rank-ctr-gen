from joblib import Parallel
from tqdm import tqdm

from functools import partial, wraps
from pathlib import Path
import json

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def wrapped_partial(f, *args, **kwargs):
    return wraps(f)(partial(f, *args, **kwargs))

def save_generations(generations, out_file):
    f = open(out_file, 'w')
    for generation in generations:
        json.dump(generation, f)
        f.write("\n")
    f.close()
    print("save generations to:{}".format(out_file))

def load_generations(file, num_samples):
    if file.endswith("txt"):
        hyp_f = open(file, encoding="utf-8")
        flat_hyps = [line.strip() for line in hyp_f.readlines()]
    elif file.endswith("jsonl"):
        hyp_f = open(file)
        flat_hyps = [json.loads(line) for line in hyp_f]
    else:
        raise NotImplementedError

    print(f'Loading generations from {file}')
    assert len(flat_hyps) % num_samples == 0

    # unflatten the hypotheses
    hyps = []
    for i in range(0, len(flat_hyps) // num_samples):
        hyps.append([])
        for j in range(num_samples):
            hyps[i].append(flat_hyps[i * num_samples + j])

    return hyps
