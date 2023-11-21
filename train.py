import argparse
import warnings
import os
import sys
import math
import time
import torch
from torch.utils.data import Dataset
from torch import optim
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import setup_seed, AverageMeter, decode_output
import transformers
from data_handling.pretrain_dataset import pretrain_dataloader
from data_handling.retrieval_dataset import RetrievalIndex
from accelerate import Accelerator, DistributedDataParallelKwargs
from models.audio_caption import CLAP2LLAMA
import evaluate
from metrics import SpiceMetric, CocoTokenizer, CiderMetric

warnings.simplefilter("ignore", UserWarning)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=False)
accelerator = Accelerator(gradient_accumulation_steps=8, log_with="wandb", kwargs_handlers=[ddp_kwargs])


def get_config():
    parser = argparse.ArgumentParser(description="Generate Faiss index from PyTorch DataLoader")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def train(model, dataloader, optimizer, scheduler, epoch, max_grad=1.0, index=None):
    model.train()
    epoch_loss = AverageMeter()
    start_time = time.time()
    retr_texts = None
    retr_audios = None
    for batch_id, (audio, text, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with accelerator.accumulate(model):
            if index:  # RetrieVve audio and texts
                _, _, retr_texts, retr_audios = index.get_nns(audio)
            optimizer.zero_grad()
            step = len(dataloader) * (epoch - 1) + batch_id
            with accelerator.autocast():
                output = model(audio, text, retr_audios=retr_audios, retr_texts=retr_texts)
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
def validate(data_loader, model, epoch, index=None):
    model.eval()
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    spice = SpiceMetric()
    cider = CiderMetric()
    unwrapped_model = accelerator.unwrap_model(model)
    gen_captions = []
    ref_captions = []
    file_names_all = []
    retr_texts = None
    retr_audios = None
    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        audio, caption_dict, audio_names = batch_data
        if index:  # RetrieVve pair of audio and texts
            _, _, retr_texts, retr_audios = index.get_nns(audio)
        with accelerator.autocast():
            # print([[caption[0] for caption in caption_dict]])
            # print(retr_texts[0])
            if retr_texts is None:
                retr_texts = [[caption[0] for caption in caption_dict]]
            output = unwrapped_model.generate_caption(audio=audio, retr_audios=retr_audios, retr_texts=retr_texts)
            gen_captions.extend(output)
            # print(output)
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
    model = CLAP2LLAMA(config.model_args)
    accelerator.gradient_accumulation_steps = config.data_args.global_batch_size // (config.data_args.batch_size*accelerator.state.num_processes)
    if config.model_args.checkpoint_path:
        model.load_ckpt(config.model_args.checkpoint_path)
    train_dataloader = pretrain_dataloader(config, subset="train_jsons", bucket=False, is_distributed=False,num_tasks=1,global_rank=0,shuffle=True)
    val_dataloader = pretrain_dataloader(config, subset="val_jsons", bucket=False, is_distributed=False, num_tasks=1,global_rank=0,shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay, fused=False)
    warmup_steps = int(len(train_dataloader)*config.training.warmup_epochs/accelerator.gradient_accumulation_steps)
    train_steps = int(len(train_dataloader)*config.training.epochs/accelerator.gradient_accumulation_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, scheduler)
    spiders = []
    index = None
    if config.training.use_retrieval:
        index = RetrievalIndex(
            n_probe=16,
            index_path=config.index_args.index_save_path,
            top_k=config.index_args.top_k,
            device=accelerator.device
        )
    if accelerator.state.mixed_precision == "no": # Ampere, Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.training.eval and accelerator.is_main_process: # Load checkpoint & Eval only,
        metrics = validate(val_dataloader, model, 0, index)
        accelerator.print(metrics)
        accelerator.end_training()
        sys.exit()
    for epoch in range(1, config.training.epochs + 1): # 1~10
        train_statics = train(model, train_dataloader, optimizer, scheduler, epoch, config.training.clip_grad, index)
        accelerator.print(train_statics)
        if accelerator.is_main_process:
            metrics = validate(val_dataloader, model, epoch, index)
            spiders.append(metrics["spider"])
            if metrics["spider"] >= max(spiders):
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_ckpt(config.training.output_path)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
