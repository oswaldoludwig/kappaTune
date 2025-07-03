# kappaTune

A PyTorch-based optimizer wrapper for continual learning via selective fine-tuning, guided by the condition number ($\kappa$) of model tensors. KappaTune identifies and updates only the least anisotropic parameters to preserve pre-trained knowledge and mitigate catastrophic forgetting.

**Please cite the following paper if you use this code or ideas derived from it in your publications:**
[https://arxiv.org/html/2506.16289v1](https://arxiv.org/html/2506.16289v1)

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
`kappaTune` is designed to address the challenge of catastrophic forgetting in continual learning scenarios. By analyzing the condition numbers of a neural network's weight matrices, it selects a subset of parameters to fine-tune. This approach updates only tensors with the smallest condition numbers due to a synergy of factors: they inherent numerical stability makes them less susceptible to training noise, and their less specialized nature allows for robust adaptation without overwriting critical, highly specific pre-training knowledge, thereby effectively mitigating catastrophic forgetting of foundational capabilities, as shown in the [paper](https://arxiv.org/html/2506.16289v1).

## Features
* **Condition Number Guided Selection:** Ranks model parameters based on their condition numbers, prioritizing those that are less anisotropic (more "round" in their singular value distribution).
* **Selective Fine-Tuning:** Integrates with any standard PyTorch optimizer, ensuring only the selected parameters are updated.
* **Efficient Analysis:** Caches condition numbers to avoid redundant computations across multiple runs or experiments.
* **Flexible Filtering:** Allows skipping parameters based on number of dimensions, or maximum dimension size, providing fine-grained control over which tensors are considered for analysis.
* **Catastrophic Forgetting Mitigation:** By selectively updating parameters, `kappaTune` helps preserve pre-trained knowledge, making it suitable for continual learning and domain adaptation tasks.

## Installation

### Prerequisites
* Python 3.8+
* `pip` package manager

### Dependencies
You can install the required libraries using `pip`:

```bash
pip install torch transformers datasets numpy
```

## Usage
**complete_example_use_selective_fine_tuning.py** demonstrates how to use kappaTune to fine-tune a TinyLlama-1.1B model on a text classification dataset (ag_news), selectively updating parameters based on their condition numbers. Note that while ag_news is a classification dataset, the example code performs a language modeling (next-token prediction) task only to illustrate the LLM adaptation.
