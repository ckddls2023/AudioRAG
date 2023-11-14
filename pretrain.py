import argparse
import os
import math
import time
import torch
from torch.utils.data import Dataset
from torch import optim
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import setup_seed, AverageMeter, decode_output
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from data_handling.pretrain_dataset import pretrain_dataloader
from accelerate import Accelerator
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict
from models.audio_caption import CLAP2LLAMA, FrozenArgs
import evaluate
from metrics import SpiceMetric, CocoTokenizer, CiderMetric

accelerator = Accelerator(gradient_accumulation_steps=8, log_with="wandb")


def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def train(model, dataloader, optimizer, scheduler, epoch, max_grad=1.0):
    model.train()
    epoch_loss = AverageMeter()
    start_time = time.time()
    for batch_id, (audio, text, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            step = len(dataloader) * (epoch - 1) + batch_id
            with accelerator.autocast():
                output = model(audio, text)
            accelerator.backward(output["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad) # 1.0
                accelerator.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
            optimizer.step()
            scheduler.step()
        epoch_loss.update(output["loss"].cpu().item())
    elapsed_time = time.time() - start_time
    accelerator.log({"loss": epoch_loss.avg, "epoch": epoch})
    return {
        "loss": epoch_loss.avg, "time": elapsed_time
    }


@torch.no_grad()
def validate(data_loader, model, epoch):
    model.eval()
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    spice = SpiceMetric()
    cider = CiderMetric()
    unwrapped_model = accelerator.unwrap_model(model)
    gen_captions = []
    ref_captions = []
    file_names_all = []
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        audios, caption_dict, audio_names = batch_data
        with accelerator.autocast():
            output = unwrapped_model.generate_caption(samples=audios)
            gen_captions.extend(output)
        ref_captions.extend(caption_dict)
        file_names_all.extend(audio_names)
    sacrebleu_score = sacrebleu.compute(predictions=gen_captions, references=ref_captions)
    meteor_score = meteor.compute(predictions=gen_captions, references=ref_captions)
    tokenizer = CocoTokenizer(gen_captions, ref_captions)
    tokens = tokenizer.tokenize()
    if isinstance(ref_captions, list) and all(isinstance(caption, str) for caption in ref_captions):
        ref_captions = [[caption] for caption in ref_captions] # List of list, val or test may include 5 captions
    spice_score = spice.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
    cider_score = cider.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
    spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])
    metrics_all = {
        "sacrebleu": sacrebleu_score['score'],
        "meteor": meteor_score['meteor'],
        "spice": spice_score['average_score'],
        "cider": cider_score['score'],
        "spider": spider_score,
    }
    accelerator.log(metrics_all)
    return metrics_all


def main():
    config = get_config()
    setup_seed(config.seed)
    exp_name = config.exp_name + f"lr_{config.optim_args.lr}_seed_{config.seed}"
    accelerator.init_trackers(
        project_name="audio-captioning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": {"name": exp_name}}
    )
    model = CLAP2LLAMA(args=FrozenArgs(freeze_lm=True,freeze_am=True))
    train_dataloader = pretrain_dataloader(config, subset="train_jsons", bucket=False, is_distributed=False,num_tasks=1,global_rank=0,shuffle=True)
    val_dataloader = pretrain_dataloader(config, subset="val_jsons", bucket=False, is_distributed=False, num_tasks=1,global_rank=0,shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay, fused=False)
    warmup_steps = int(len(train_dataloader)*config.training.warmup_epochs/accelerator.gradient_accumulation_steps)
    train_steps = int(len(train_dataloader)*config.training.epochs/accelerator.gradient_accumulation_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, scheduler)
    spiders = []
    for epoch in range(1, config.training.epochs + 1): # 1~10
        train_statics = train(model, train_dataloader, optimizer, scheduler, epoch, config.training.clip_grad)
        accelerator.print(train_statics)
        if accelerator.is_main_process:
            metrics = validate(val_dataloader, model, epoch)
            spiders.append(metrics["spider"])
            if metrics["spider"] >= max(spiders):
                # Better to use get_peft_model_state_dict, hard coded save. Please hotfix
                unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.decoder.save_pretrained("pretrained_models/audio_caption/") # PEFT model
                torch.save(unwrapped_model.enc_to_dec_proj.state_dict(), "pretrained_models/audio_caption/mm_projector.bin")
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
