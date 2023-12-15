import argparse
import warnings
import os
import sys
import math
import time
import torch
import json
from torch.utils.data import Dataset
from torch import optim
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import setup_seed, AverageMeter, decode_output
import transformers
from data_handling.pretrain_dataset import pretrain_dataloader
from accelerate import Accelerator, DistributedDataParallelKwargs
from models.audio_caption import CLAP2LLAMA
import evaluate
from metrics import SpiceMetric, CocoTokenizer, CiderMetric

warnings.simplefilter("ignore", UserWarning)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=False)
accelerator = Accelerator(gradient_accumulation_steps=8, log_with="wandb", kwargs_handlers=[ddp_kwargs], even_batches=True)


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
    start_time = time.time()
    for batch_id, (audio, caption, audio_filenames, retr_audios, retr_captions) in enumerate(pbar := tqdm(dataloader, total=len(dataloader))):
        iter_time = time.time() - start_time
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            step = len(dataloader) * (epoch - 1) + batch_id
            with accelerator.autocast():
                output = model(audio, caption, retr_audios=retr_audios, retr_captions=retr_captions)
            accelerator.backward(output["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad) # 1.0
                accelerator.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
            optimizer.step()
            scheduler.step()
        epoch_loss.update(output["loss"].cpu().item())
        pbar.set_description(f"loss: {epoch_loss.avg}, data: {iter_time}")
        start_time = time.time()
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
    rouge = evaluate.load("rouge")
    spice = SpiceMetric()
    cider = CiderMetric()
    unwrapped_model = accelerator.unwrap_model(model)
    gen_captions = []
    ref_captions = []
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        audio, caption, audio_names, retr_audios, retr_captions = batch_data
        if not retr_captions: # If retrieved results is missing, num_captions = 5, choose 1
            retr_captions = [[texts[0] for texts in caption]]
        #retr_audios = [] # Force to only put captions
        #retr_captions = [] # Force to only put captions
        prompt = "Similar audio sounds like "
        with accelerator.autocast():
            gen_caption = unwrapped_model.generate_caption(audio=audio, retr_audios=retr_audios, retr_captions=retr_captions, prompt=prompt)
            print(gen_caption)
            print(caption)
            gen_captions.extend(gen_caption)
            ref_captions.extend(caption)
    if accelerator.is_main_process:
        sacrebleu_score = sacrebleu.compute(predictions=gen_captions, references=ref_captions)
        meteor_score = meteor.compute(predictions=gen_captions, references=ref_captions)
        tokenizer = CocoTokenizer(gen_captions, ref_captions)
        tokens = tokenizer.tokenize()
        if isinstance(ref_captions, list) and all(isinstance(caption, str) for caption in ref_captions):
            ref_captions = [[caption] for caption in ref_captions] # List of list, val or test may include 5 captions
        spice_score = spice.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        cider_score = cider.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        rouge_score = rouge.compute(predictions=gen_captions, references=ref_captions)
        spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])
        metrics_all = {
            "sacrebleu": sacrebleu_score['score'],
            "meteor": meteor_score['meteor'],
            "spice": spice_score['average_score'],
            "cider": cider_score['score'],
            "rougeL": rouge_score['rougeL'],
            "spider": spider_score,
        }
        accelerator.log(metrics_all)
        accelerator.print(metrics_all)
        return metrics_all
    else:
        return None


def main():
    config = get_config()
    setup_seed(config.seed)
    exp_name = config.exp_name + "_" + config.model_args.align.model_name + f"_lr_{config.optim_args.lr}_seed_{config.seed}"
    accelerator.init_trackers(
        project_name="audio-captioning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": {"name": exp_name}}
    )
    model = CLAP2LLAMA(config.model_args)
    accelerator.gradient_accumulation_steps = config.data_args.global_batch_size // (config.data_args.batch_size*accelerator.state.num_processes)
    if config.model_args.checkpoint_path:
        model.load_ckpt(config.model_args.checkpoint_path)
    train_dataloader = pretrain_dataloader(config, subset="train_jsons", bucket=False, is_distributed=False,num_tasks=1,global_rank=0,shuffle=True,retrieve_map=config.index_args.index_path,top_k=config.index_args.top_k)
    val_dataloader = pretrain_dataloader(config, subset="val_jsons", bucket=False, is_distributed=False, num_tasks=1,global_rank=0,shuffle=False,retrieve_map=config.index_args.index_path,top_k=config.index_args.top_k)
    optimizer = optim.AdamW(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay, fused=False)
    warmup_steps = int(len(train_dataloader)*config.training.warmup_epochs/accelerator.gradient_accumulation_steps)
    train_steps = int(len(train_dataloader)*config.training.epochs/accelerator.gradient_accumulation_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, scheduler)
    spiders = []
    save_ckpt = accelerator.is_main_process
    if accelerator.state.mixed_precision == "no": # Ampere, Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.training.eval: # Load checkpoint & Eval only,
        validate(val_dataloader, model, 0)
        accelerator.wait_for_everyone()
        accelerator.end_training()
        sys.exit()
    for epoch in range(1, config.training.epochs + 1): # 1~10
        train_statics = train(model, train_dataloader, optimizer, scheduler, epoch, config.training.clip_grad)
        if config.training.validate: # Load checkpoint & Eval only,
            metrics = validate(val_dataloader, model, epoch)
            if accelerator.is_main_process:
                spiders.append(metrics["spider"])
                save_ckpt = metrics["spider"] >= max(spiders) 
        if save_ckpt: 
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_ckpt(config.training.output_path)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
