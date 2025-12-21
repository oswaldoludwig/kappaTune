# This script implements experiments for applying KappaTune to the OLMoE MoE-based LLM for sarcasm detection,
# comparing with LoRA baseline. It uses selective fine-tuning based on condition numbers.
# Author: Oswaldo Ludwig (including vibe coding)
# Date: December 15, 2025
# Citation: Ludwig, Oswaldo. "The Condition Number as a Scale-Invariant Proxy for Information Encoding in Neural Units." arXiv preprint arXiv:2506.16289 (2025).

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import math
import evaluate  # For perplexity, but we'll implement manually
import json
import os
import time

# Assuming selective_fine_tuning.py is in the same directory or installed.
# Download from https://github.com/oswaldoludwig/kappaTune
from selective_fine_tuning import SelectiveFineTuningOptimizer

# Dataset for sarcasm detection using tweet_eval/irony as proxy (irony ~ sarcasm)
class SarcasmDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=200, max_length=128):
        # Use tweet_eval/irony: labels 0 (non-ironic), 1 (ironic/sarcastic)
        dataset = load_dataset("tweet_eval", "irony", split=split)
        self.samples = dataset.select(range(min(max_samples, len(dataset))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {0: "no", 1: "yes"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        label = self.samples[idx]['label']
        label_str = self.label_map[label]
        prompt = f"Tweet: {text}\nSarcasm: {label_str}"
        encoding = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label  # Return label for evaluation

# Function to compute perplexity on a general dataset (e.g., wikitext) to measure forgetting
def compute_perplexity(model, tokenizer, split='test', max_samples=100, max_length=128, device='cuda'):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    samples = dataset.select(range(min(max_samples, len(dataset))))
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    with torch.no_grad():
        for sample in samples:
            text = sample['text'].strip()
            if not text:
                continue
            encoding = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]  # Shift for causal LM
            targets = input_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()
    if total_tokens == 0:
        return float('inf')
    ppl = math.exp(total_loss / total_tokens)
    return ppl

# Function to evaluate sarcasm detection accuracy
def evaluate_sarcasm(model, tokenizer, dataloader, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # Prompt up to "Sarcasm:", generate next token
            prompt_ids = tokenizer("Tweet: ", add_special_tokens=False)['input_ids']  # Simplified; adjust based on actual formatting
            # Actually, to generate, find position of "\nSarcasm:"
            for i in range(input_ids.size(0)):
                # Find the position where generation starts
                seq = input_ids[i]
                sarcasm_pos = (seq == tokenizer.encode("\nSarcasm:")[0]).nonzero(as_tuple=True)[0]
                if len(sarcasm_pos) == 0:
                    continue
                input_prefix = seq[:sarcasm_pos[-1] + len(tokenizer.encode("\nSarcasm:"))]
                outputs = model.generate(input_prefix.unsqueeze(0), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
                generated_token = outputs[0, -1].item()
                generated_str = tokenizer.decode(generated_token).strip()
                true_label = "yes" if labels[i] == 1 else "no"
                if generated_str.lower() in true_label:
                    correct += 1
                total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Main function for KappaTune experiment
def run_kappatune_experiment(epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "allenai/OLMoE-1B-7B-0924"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    print("Tokenizer and OLMoE model loaded.")

    # Datasets
    train_dataset = SarcasmDataset(tokenizer, split='train', max_samples=200)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = SarcasmDataset(tokenizer, split='test', max_samples=100)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # Pre-fine-tune evaluations
    pre_ppl = compute_perplexity(model, tokenizer, device=device)
    pre_acc = evaluate_sarcasm(model, tokenizer, test_dataloader, device=device)
    print(f"Pre-fine-tune: Perplexity = {pre_ppl:.4f}, Accuracy = {pre_acc:.4f}")

    # KappaTune optimizer
    optimizer_wrapper = SelectiveFineTuningOptimizer(
        model=model,
        base_optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': 5e-5},
        condition_file='olmoe_condition_numbers.json',
        num_tensors_to_finetune=75,  # Finer granularity in MoE
        recompute=True,  # Compute for OLMoE
        max_dim_size_to_analyze=4096  # Skip very large tensors if needed
    )
    print("KappaTune optimizer instantiated.")

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in KappaTune: {trainable_params}")

    start = time.time()
    model.train()
    for epoch in range(epochs):  # Small epochs for demo
        total_loss = 0
        for input_ids, attention_mask, _ in train_dataloader:  # Ignore label for training loop
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer_wrapper.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer_wrapper.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")
    end = time.time()
    print("Elapsed time for KappaTune training: " + str(end - start))

    # Post-fine-tune evaluations
    post_ppl = compute_perplexity(model, tokenizer, device=device)
    post_acc = evaluate_sarcasm(model, tokenizer, test_dataloader, device=device)
    print(f"Post-KappaTune: Perplexity = {post_ppl:.4f} (delta (forgetting): {post_ppl - pre_ppl:.4f}), Accuracy = {post_acc:.4f}")

    # Save model
    model.save_pretrained("olmoe_kappatune_sarcasm")

# Main function for LoRA baseline
def run_lora_experiment(target_trainable_params, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "allenai/OLMoE-1B-7B-0924"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Tokenizer and OLMoE model loaded for LoRA.")

    # Datasets same as above
    train_dataset = SarcasmDataset(tokenizer, split='train', max_samples=200)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = SarcasmDataset(tokenizer, split='test', max_samples=100)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # Pre-fine-tune same
    pre_ppl = compute_perplexity(model, tokenizer, device=device)
    pre_acc = evaluate_sarcasm(model, tokenizer, test_dataloader, device=device)
    print(f"Pre-fine-tune (LoRA): Perplexity = {pre_ppl:.4f}, Accuracy = {pre_acc:.4f}")

    # Bypassing the endless problems of PEFT: generating explicit target_modules list for OLMoE (16 layers, 64 experts/layer)
    target_modules = []
    for layer in range(16):
        target_modules.append(f"model.layers.{layer}.self_attn.q_proj")
        target_modules.append(f"model.layers.{layer}.self_attn.v_proj")
        for expert in range(64):
            target_modules.append(f"model.layers.{layer}.mlp.experts.{expert}.gate_proj")
            target_modules.append(f"model.layers.{layer}.mlp.experts.{expert}.up_proj")
            target_modules.append(f"model.layers.{layer}.mlp.experts.{expert}.down_proj")
    print(f"Generated {len(target_modules)} target modules.")

    # Then lora_config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=target_trainable_params,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules  # Use the explicit list
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # To verify

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    start = time.time()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, _ in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}")
    end = time.time()
    print("Elapsed time for LoRA training: " + str(end - start))

    # Post evaluations
    post_ppl = compute_perplexity(model, tokenizer, device=device)
    post_acc = evaluate_sarcasm(model, tokenizer, test_dataloader, device=device)
    print(f"Post-LoRA: Perplexity = {post_ppl:.4f} (delta (forgetting): {post_ppl - pre_ppl:.4f}), Accuracy = {post_acc:.4f}")

    model.save_pretrained("olmoe_lora_sarcasm")

if __name__ == '__main__':
    # Run KappaTune first
    Start = time.time()
    run_kappatune_experiment(epochs=18)
    End = time.time()
    print("Total elapsed time for Kappatune training, including condition number calculation and moodel instantiation: " + str(End - Start))
    # Then run LoRA
    run_lora_experiment(target_trainable_params=16, epochs=12)
