# This script exemplify a selective fine-tuning method based on the condition number using freely available data and LLM
# Author: Oswaldo Ludwig (now with AI support)
# Date: 03/07/2025
# In case of publication using this script or ideas in this script, cite:
# Ludwig, Oswaldo. "The Condition Number as a Scale-Invariant Proxy for Information Encoding in Neural Units." arXiv preprint arXiv:2506.16289 (2025).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from selective_fine_tuning import SelectiveFineTuningOptimizer

# Dataset using AG News
class RealTextDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=200, seq_len=64):
        dataset = load_dataset("ag_news", split=split)
        self.samples = dataset.select(range(max_samples))
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.seq_len, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        return input_ids, input_ids.clone()

# Training loop
def train():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Tokenizer and LLM loaded", flush=True)

    dataset = RealTextDataset(tokenizer=tokenizer, max_samples=200, seq_len=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Data loader and dataset loaded", flush=True)

    criterion = nn.CrossEntropyLoss()

    optimizer_wrapper = SelectiveFineTuningOptimizer(
        model=model,
        base_optimizer_cls=optim.AdamW,
        optimizer_args={'lr': 5e-5},
        condition_file='condition_numbers.json',
        num_tensors_to_finetune=20,
        recompute=True
    )
    print("Optimizer instantiated", flush=True)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer_wrapper.zero_grad()
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer_wrapper.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    train()
