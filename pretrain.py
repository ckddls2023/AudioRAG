import argparse
import os
import math
from torch.utils.data import Dataset
from torch import optim
from tools.utils import setup_seed, set_logger, AverageMeter, decode_output
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from data_handling.pretrain_dataset import pretrain_dataloader
from accelerate import Accelerator
from models import CLAP2GPT2, CLAP2LLAMA

accelerator = Accelerator()


def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def train(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    epoch_loss = AverageMeter()
    start_time = time.time()
    for batch_id, (audio, text, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        step = len(dataloader) * (epoch - 1) + batch_id
        if scheduler is not None:
            scheduler(step)
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
        with accelerator.autocast():
            loss = model(audio, text)
        accelerator.backward(loss)
        optimizer.step()
        epoch_loss.update(loss.cpu().item())
    elapsed_time = time.time() - start_time
    wandb.log({"loss": epoch_loss.avg, "epoch": epoch})
    return {
        "loss": epoch_loss.avg, "time": elapsed_time
    }


def main():
    config = get_config()
    setup_seed(config.seed)
    exp_name = config.exp_name + f"lr_{config.optim_args.lr}_seed_{config.seed}"
    wandb.init(project="audio-captioning", name=exp_name, config=config)
    model = CLAP2LLAMA()
    train_dataloader = pretrain_dataloader(config, bucket=False, is_distributed=False, num_tasks=1, global_rank=0)
    optimizer = optim.AdamW(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay, fused=True)
    scheduler = cosine_lr(optimizer, base_lr=config.optim_args.lr, warmup_length=config.optim_args.warmup_epochs * len(dataloader), steps=len(dataloader) * config.training.epochs)
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)
    # TODO : Impl evaluate metric
    for epoch in range(1, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        train_statics = train(model, train_loader, optimizer, scheduler, epoch)
        print(train_statics)
