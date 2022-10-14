
import torch
from tqdm import tqdm

from eval_metrics.run_pplm_discrim_train import load_discriminator, collate_fn
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, X, discriminator, device, max_length_seq = 128, add_eos_token = True):
        """Reads source and target sequences from txt files."""
        self.X = []
        self.y = []

        for text in X:
            seq = discriminator.tokenizer.encode(text)
            if add_eos_token:
                seq = [discriminator.tokenizer.eos_token_id] + seq
            seq = seq[:max_length_seq]
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            self.X.append(seq)
            self.y.append(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data

class ToxicityScorer:

    def __init__(self, weights_path, meta_path, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.meta_path = meta_path
        self.batch_size = batch_size

    def __call__(self, generations, srcs):
        
        discriminator, meta_param = load_discriminator(self.weights_path, self.meta_path, device=self.device)
        discriminator = discriminator.to(self.device)
        discriminator.eval()
        eval_dataset = Dataset(generations, discriminator, self.device, max_length_seq = 128, add_eos_token = True)
        data_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                    batch_size=self.batch_size,
                                                    collate_fn=collate_fn)
        SCORES = []
        with torch.no_grad():
            for input_t,_ in tqdm(data_loader):
                output_t = discriminator(input_t)
                scores = torch.softmax(output_t,1)[:,1]
                SCORES.extend(scores.cpu().detach().numpy().tolist())

        return np.array(SCORES)