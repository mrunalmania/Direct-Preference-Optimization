# so here we need to build our train loop. 

# let's import the libs first
import torch
from utils import *

import torch.nn as nn
import torch.functional as F
import numpy
import pandas

from tqdm import tqdm
import wandb


def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, beta=0.1):
    model.train()
    ref_model.eval()

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            prompt_prefered_ids = batch['prompt_prefered_ids']
            prompt_disprefered_ids = batch['prompt_disprefered_ids']
            prompt_prefered_mask = batch['prompt_prefered_mask']
            prompt_disprefered_mask = batch['prompt_disprefered_mask']

            model_prefered_log_prob = get_log_prob(model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            model_disprefered_log_prob = get_log_prob(model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            ref_prefered_log_prob = get_log_prob(ref_model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits, prompt_prefered_ids)
            ref_disprefered_log_prob = get_log_prob(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, prompt_disprefered_ids)

            loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(model_prefered_log_prob, model_disprefered_log_prob,
                                          ref_prefered_log_prob, ref_disprefered_log_prob,
                                          beta=beta)

            loss.backward()
            optimizer.step()

            wandb.log({'loss': loss.item(),
                       'prefered_relative_logprob': prefered_relative_logprob,
                       'disprefered_relative_logprob': disprefered_relative_logprob,
                       'reward_accuracy': reward_accuracies,
                       'reward_margin': reward_margins})