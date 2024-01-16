import argparse
import warnings
import os
import sys
import math
import time
import torch
import json
import numpy as np
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
from sentence_transformers import SentenceTransformer
from models.align2text import align2text
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig

warnings.simplefilter("ignore", UserWarning)
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True, static_graph=False
)
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    log_with="wandb",
    kwargs_handlers=[ddp_kwargs],
    even_batches=True,
)


def get_config():
    parser = argparse.ArgumentParser(
        description="Generate Faiss index from PyTorch DataLoader"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    return config


def train(
    model,
    audio_encoder,
    text_encoder,
    dataloader,
    optimizer,
    scheduler,
    epoch,
    max_grad,
):
    model.train()
    epoch_loss = AverageMeter()
    start_time = time.time()
    start_time = time.time()
    for batch_id, (
        audio,
        caption,
        audio_filenames,
        retr_audios,
        retr_captions,
        lm_attn
    ) in enumerate(pbar := tqdm(dataloader, total=len(dataloader))):
        lm_attn = torch.tensor(lm_attn).to(device=accelerator.device, dtype=torch.float, non_blocking=True)
        iter_time = time.time() - start_time
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            step = len(dataloader) * (epoch - 1) + batch_id
            with accelerator.autocast():
                audio_embed = audio_encoder(audio).last_hidden_state  # B, 64, 768
                text_embed = text_encoder.encode(
                    caption, normalize_embeddings=True, convert_to_tensor=True
                )
                output = model(audio_embed, text_embed, lm_attn=lm_attn)
            accelerator.backward(output["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad)  # 1.0
                accelerator.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
            optimizer.step()
            scheduler.step()
        epoch_loss.update(output["loss"].cpu().item())
        pbar.set_description(f"loss: {epoch_loss.avg}, data: {iter_time}")
        start_time = time.time()
    elapsed_time = time.time() - start_time
    accelerator.log({"loss": epoch_loss.avg, "epoch": epoch})
    return {"loss": epoch_loss.avg, "time": elapsed_time}


def main():
    config = get_config()
    setup_seed(config.seed)
    exp_name = (
        "align_audio2text_"
        + config.model_args.align.model_name
        + f"_lr_{config.optim_args.lr}_seed_{config.seed}"
    )
    accelerator.init_trackers(
        project_name="audio-captioning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": {"name": exp_name}},
    )
    config.optim_args.lr = 1e-3 # Language Bind
    config.optim_args.weight_decay = 1e-4 # Language Bind 0.2, LiT 1e-4
    config.index_args.index_path = ""
    config.index_args.top_k = 0
    config.training.epochs = 30 # 0.6 * 30 = 18M
    config.train_jsons = [
        "./data/json_files/AudioSet/train.json",  # 48K
        "./data/json_files/Clotho/train.json",  # 6K
        "./data/json_files/Auto_ACD/train.json",  # 1903803
        # "./data/json_files/BBC_Sound_Effects/bbc_final.json",
        # "./data/json_files/FreeSound/fsd_final.json",
    ]
    config.data_args.global_batch_size = 4096  # 1024 Global batch size
    config.data_args.train_batch_size = 256  # 1024 Global batch size
    config.training.output_path = "./retriever_models_lm_attn"
    # config.model_args.unfreeze_am = [
    #     "linear_a_q",
    #     "linear_b_q",
    #     "linear_a_v",
    #     "linear_b_v",
    # ]  # Always finetune
    config.model_args.unfreeze_am = []  # Always finetune
    config.model_args.encoder.use_lora = True
    config.model_args.checkpoint_path = (
        "./pretrained_models/pretrained_MLP_clap_lora_AutoACD/"
    )
    config.model_args.encoder.window_size = 4 # (32,32) = [B,32,H], (16,16) = [B,64,H], (8,8) = [B,128,H] (4,4) = [B,256,H]
    config.model_args.encoder.step_size = 4
    encoder_config = CLAPEncoderConfig.from_dict(
        OmegaConf.to_container(config.model_args.encoder, resolve=True)
    )
    audio_encoder = CLAPAudioTower(encoder_config)
    accelerator.gradient_accumulation_steps = config.data_args.global_batch_size // (
        config.data_args.batch_size * accelerator.state.num_processes
    )
    if config.model_args.checkpoint_path:
        audio_encoder_ckpt = os.path.join(
            config.model_args.checkpoint_path, "audio_encoder.bin"
        )
        if os.path.exists(audio_encoder_ckpt):
            audio_encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=False)
    audio_encoder = audio_encoder.to(accelerator.device)
    audio_encoder.eval()
    train_dataloader = pretrain_dataloader(
        config,
        subset="train_jsons",
        bucket=False,
        is_distributed=False,
        num_tasks=1,
        global_rank=0,
        shuffle=True,
        retrieve_map=config.index_args.index_path,
        top_k=config.index_args.top_k,
    )
    align_model = align2text(hidden_size=768, num_latents=64, num_layers=2)
    align_model = align_model.to(accelerator.device)
    # combined_parameters = [p for p in audio_encoder.parameters() if p.requires_grad] + \
    #                   [p for p in align_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        align_model.parameters(),
        lr=config.optim_args.lr,
        betas=config.optim_args.betas,
        eps=config.optim_args.eps,
        weight_decay=config.optim_args.weight_decay,
        fused=False,
    )
    warmup_steps = int(
        len(train_dataloader)
        * config.training.warmup_epochs
        / accelerator.gradient_accumulation_steps
    )
    train_steps = int(
        len(train_dataloader)
        * config.training.epochs
        / accelerator.gradient_accumulation_steps
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, train_steps
    )
    text_encoder = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    text_encoder.eval()
    train_dataloader, align_model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, align_model, optimizer, scheduler
    )
    save_ckpt = accelerator.is_main_process
    if accelerator.state.mixed_precision == "no":  # Ampere, Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    attention_map = None
    for epoch in range(1, config.training.epochs + 1):  # 1~10
        train(
            align_model,
            audio_encoder,
            text_encoder,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            config.training.clip_grad,
        )
        if save_ckpt:
            unwrapped_model = accelerator.unwrap_model(align_model)
            torch.save(
                unwrapped_model.state_dict(),
                os.path.join(config.training.output_path, f"epoch_{epoch}.pt"),
            )
            torch.save(
                audio_encoder.state_dict(),
                os.path.join(config.training.output_path, "audio_encoder.bin"),
            )
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
