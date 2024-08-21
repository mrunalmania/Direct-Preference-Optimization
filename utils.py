import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from tqdm import tqdm

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

def collate_fn(batch, tokenizer, max_length, device):
    prompts = ['Instruct: ' + item['prompt'] + '\n' for item in batch]
    chosen_responses = ['Output: ' + item['chosen'] for item in batch]
    rejected_responses = ['Output: ' + item['rejected'] for item in batch]

    prompt_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    prefered_ids = tokenizer.batch_encode_plus(chosen_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)
    disprefered_ids = tokenizer.batch_encode_plus(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True)['input_ids'].to(device)

    prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
    prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

    prompt_prefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1)
    prompt_disprefered_mask = torch.cat([torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1)

    return {'prompt_prefered_ids': prompt_prefered_ids,
            'prompt_disprefered_ids': prompt_disprefered_ids,
            'prompt_prefered_mask': prompt_prefered_mask,
            'prompt_disprefered_mask': prompt_disprefered_mask}