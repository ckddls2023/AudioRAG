import argparse
import warnings
import os
import sys
import math
import time
import torch
import json
from torch.utils.data import Dataset
from torch import optim, nn
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from utils import setup_seed, AverageMeter, decode_output
from accelerate.utils.deepspeed import (
    DeepSpeedEngineWrapper,
    DeepSpeedOptimizerWrapper,
    DeepSpeedSchedulerWrapper,
    DummyOptim,
    DummyScheduler,
)

# Monkey patch efficient xformer attn
# from llama_xformers_attn_monkey_patch import (
#     replace_llama_attn_with_xformers_attn,
# )
# replace_llama_attn_with_xformers_attn()
    
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

# from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

import transformers
from data_handling.pretrain_dataset import pretrain_dataloader
from accelerate import Accelerator, DistributedDataParallelKwargs, FullyShardedDataParallelPlugin
from deepspeed import DeepSpeedEngine
from models.audio_caption import CLAP2LLAMA, SALMONN
import evaluate
from metrics import SpiceMetric, CocoTokenizer, CiderMetric
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

accelerator = Accelerator(log_with="wandb")

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
    for batch_id, (audio, caption, audio_filenames, retr_audios, retr_captions, _) in enumerate(pbar := tqdm(dataloader, total=len(dataloader))):
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
        audio, caption, audio_names, retr_audios, retr_captions, _ = batch_data
        with accelerator.autocast():
            gen_caption = unwrapped_model.generate_caption(audio=audio, retr_audios=retr_audios, retr_captions=retr_captions)
            if "creating" in gen_caption[0]:
                index = gen_caption[0].index("creating")
                gen_caption[0] = gen_caption[0][:index]
            accelerator.print(retr_captions)
            accelerator.print(gen_caption)
            accelerator.print(caption)
            gen_captions.extend(gen_caption)
            ref_captions.extend(caption)
    if accelerator.is_main_process:
        meteor_score = meteor.compute(predictions=gen_captions, references=ref_captions)
        if isinstance(ref_captions, list) and all(isinstance(caption, str) for caption in ref_captions):
            ref_captions = [[caption] for caption in ref_captions] # List of list, val or test may include 5 captions
        tokenizer = CocoTokenizer(gen_captions, ref_captions)
        tokens = tokenizer.tokenize()
        spice_score = spice.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        cider_score = cider.compute(predictions=gen_captions, references=ref_captions, tokens=tokens)
        rouge_score = rouge.compute(predictions=gen_captions, references=ref_captions)
        spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])
        metrics_all = {
            "meteor": meteor_score['meteor'],
            "spice": spice_score['average_score'],
            "cider": cider_score['score'],
            "rougeL": rouge_score['rougeL'],
            "spider": spider_score,
        }
        if all(len(caption) == len(ref_captions[0]) for caption in ref_captions): # All have same length, so we can compute average score
            sacrebleu_score = sacrebleu.compute(predictions=gen_captions, references=ref_captions)
            metrics_all["sacrebleu"] = sacrebleu_score['score']
        accelerator.print(metrics_all)
        accelerator.log(metrics_all)
        return metrics_all
    else:
        return None


def main():
    # torch.autograd.set_detect_anomaly(True)
    config = get_config()
    warnings.simplefilter("ignore", UserWarning)
    setup_seed(config.seed)
    exp_name = config.exp_name + "_" + config.model_args.align.model_name + f"_lr_{config.optim_args.lr}_seed_{config.seed}"
    accelerator.init_trackers(
        project_name="audio-captioning",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        init_kwargs={"wandb": {"name": exp_name}}
    )
    # model = SALMONN(
    #     whisper_path = "/home/ckddls1321/.cache/checkpoints/whisper/",
    #     beats_path = "/home/ckddls1321/.cache/checkpoints/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    #     device = accelerator.device
    # )
    model = CLAP2LLAMA(config.model_args)
    model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_real.weight = nn.Parameter(model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_real.weight.contiguous())
    model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_imag.weight = nn.Parameter(model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_imag.weight.contiguous())
    model.encoder.clap.model.audio_branch.logmel_extractor.melW = nn.Parameter(model.encoder.clap.model.audio_branch.logmel_extractor.melW.contiguous())
    model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_real.weight.requires_grad = False
    model.encoder.clap.model.audio_branch.spectrogram_extractor.stft.conv_imag.weight.requires_grad = False
    model.encoder.clap.model.audio_branch.logmel_extractor.melW.requires_grad = False
        
    if accelerator.state.deepspeed_plugin is None:
        accelerator.gradient_accumulation_steps = config.data_args.global_batch_size // (config.data_args.batch_size*accelerator.state.num_processes)
    if config.model_args.checkpoint_path:
        # accelerator.load_state(config.training.output_path) # FSDP
        model.load_ckpt(config.model_args.checkpoint_path)
    train_dataloader = pretrain_dataloader(config, subset="train_jsons", bucket=False, is_distributed=False,num_tasks=1,global_rank=0,shuffle=True,retrieve_map=config.index_args.index_path,top_k=config.index_args.top_k)
    val_dataloader = pretrain_dataloader(config, subset="val_jsons", bucket=False, is_distributed=False, num_tasks=1,global_rank=0,shuffle=False,retrieve_map=config.index_args.index_path,top_k=config.index_args.top_k)
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.parameters(), lr=config.optim_args.lr, betas=config.optim_args.betas, eps=config.optim_args.eps, weight_decay=config.optim_args.weight_decay)
    warmup_steps = int(len(train_dataloader)*config.training.warmup_epochs/accelerator.gradient_accumulation_steps)
    train_steps = int(len(train_dataloader)*config.training.epochs/accelerator.gradient_accumulation_steps)
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=train_steps, warmup_num_steps=warmup_steps
        )
    train_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, model, optimizer, scheduler)
    print(model)
    print(f"Total iteration : {train_steps}")
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
            # accelerator.save_state(config.training.output_path) # FSDP, save all parmeter
            if isinstance(model, FSDP):
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    state = accelerator.get_state_dict(unwrapped_model.decoder_proj)
                torch.save(state, os.path.join(config.training.output_path,"decoder_proj.bin"))
                # unwrapped_model.decoder.save_pretrained( # Save PEFT model
                #     config.training.output_path,
                #     is_main_process=accelerator.is_main_process,
                #     save_function=accelerator.save,
                #     state_dict=accelerator.get_state_dict(model),
                # )
            elif isinstance(model, DeepSpeedEngine):
                success = model.save_checkpoint(config.training.output_path, epoch, exclude_frozen_parameters=True)
                if success:
                    print("Deepspeed checkpoint has been succefully saved")
            else:
                unwrapped_model.save_ckpt(config.training.output_path)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
