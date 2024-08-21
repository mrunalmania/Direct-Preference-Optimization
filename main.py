import torch
import wandb
from train import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from functools import partial
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    dataset = load_dataset(args.dataset_name, split="train")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device))

    train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=args.epochs, beta=args.beta)

    model.save_pretrained("model-DPO.pt")

if __name__ == "__main__":
    main()