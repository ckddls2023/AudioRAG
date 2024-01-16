import argparse
import warnings
import os
import sys
import math
import time
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import optim
from omegaconf import OmegaConf
import transformers
from data_handling.pretrain_dataset import pretrain_dataloader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
from models.audio_caption import CLAP2LLAMA


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


warnings.simplefilter("ignore", UserWarning)
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=False, static_graph=False
)
accelerator = Accelerator(
    gradient_accumulation_steps=8,
    log_with="wandb",
    kwargs_handlers=[ddp_kwargs],
    even_batches=True,
)


def main():
    audio_file_paths = []
    # train_json = './data/json_files/AudioSet/train.json'
    # embedding_path = "./data/embeddings/AudioSet"
    # train_json = './data/json_files/Clotho/train.json'
    # embedding_path = "./data/embeddings/Clotho"
    train_json = './data/json_files/Auto_ACD/train.json'
    embedding_path = "./data/embeddings/Auto_ACD/"
    # train_json = "./data/json_files/BBC_Sound_Effects/bbc_final.json"
    # embedding_path = "./data/embeddings/BBC_Sound_Effects"
    # train_json = './data/json_files/FreeSound/fsd_final.json'
    # embedding_path = "./data/embeddings/FreeSound"
    # train_json = 'data/json_files/SoundBible/sb_final.json'
    # embedding_path = "./data/embeddings/SoundBible"
    with open(train_json, "r") as file:
        data = json.load(file)
        audio_file_paths.extend([item["audio"] for item in data["data"]])
    config = get_config()
    config.model_args.retr_prompt = ""
    config.model_args.task_prompt = ""
    config.model_args.caption_prompt = ""
    config.index_args.index_path = ""
    config.index_args.top_k = 0
    config.data_args.batch_size = 2  # 1024 Global batch size
    config.data_args.train_batch_size = 2  # 1024 Global batch size
    config.train_jsons = [train_json]
    config.training.output_path = "./retriever_models"
    config.model_args.unfreeze_am = [
        "linear_a_q",
        "linear_b_q",
        "linear_a_v",
        "linear_b_v",
    ]  # Always finetune
    config.model_args.encoder.use_lora = True
    config.model_args.encoder.window_size = 4 # (32,32) = [B,32,H], (16,16) = [B,64,H], (8,8) = [B,128,H] (4,4) = [B,256,H]
    config.model_args.encoder.step_size = 4
    config.model_args.checkpoint_path = (
        "./pretrained_models/pretrained_MLP_clap_lora_AutoACD/"
    )
    model = CLAP2LLAMA(config.model_args)
    if config.model_args.checkpoint_path:
        model.load_ckpt(config.model_args.checkpoint_path)
    dataloader = pretrain_dataloader(
        config,
        subset="train_jsons",
        bucket=False,
        is_distributed=False,
        num_tasks=1,
        global_rank=0,
        shuffle=False,
        retrieve_map=config.index_args.index_path,
        top_k=config.index_args.top_k,
    )
    model = accelerator.prepare(model)
    model.eval()
    if accelerator.state.mixed_precision == "no":  # Ampere, Hopper
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    max_index = 0
    with torch.no_grad():
        for batch_id, (
            audio,
            caption,
            audio_filenames,
            retr_audios,
            retr_captions,
            _
        ) in enumerate(pbar := tqdm(dataloader, total=len(dataloader))):
            indices = [
                audio_file_paths.index(audio_path) for audio_path in audio_filenames
            ]
            file_names = [f"{index:07d}.npy" for index in indices]
            all_files_exist = all(
                os.path.exists(os.path.join(embedding_path, file_name))
                for file_name in file_names
            )
            max_index = max(max_index, max(indices))
            if all_files_exist:
                continue
            with accelerator.autocast():
                output = model(
                    audio, caption, retr_audios=retr_audios, retr_captions=retr_captions
                )  # should be output_attentions=True
            attention_score = torch.concat(
                output.attentions, dim=1
            )  # [[B,nH,S,S], [B,nH,S,S]] # WARN: remember to use output_attentions=True
            audio_attention_score = attention_score[
                :, :, 257:, 1:257
            ]  # [B, 1024?, 64+Max_S, 64+Max_S] # B, nL*nH, src, tgt
            grouped_attention_score = audio_attention_score.view(
                attention_score.shape[0], 2, 512, *audio_attention_score.shape[2:]
            )  # [B,2,512, :, :]
            averaged_attention_score = grouped_attention_score.mean(
                dim=2
            )  # [B,2,caption length, audio_length] # Number of layer is 2
            audio_attention_score = averaged_attention_score.mean(
                dim=2
            )  # [B,2, 256,
            # audio_attention_score = audio_attention_score.mean(dim=1) # [B,audio length]
            for audio_path, score in zip(audio_filenames, audio_attention_score):
                index = audio_file_paths.index(audio_path)
                file_name = f"{index:07d}.npy"
                file_path = os.path.join(embedding_path, file_name)
                np.save(file_path, score.cpu().numpy())
            accelerator.wait_for_everyone()

        # if accelerator.is_main_process:
            # train_json = './data/json_files/AudioSet/train.json'
            # train_json = './data/json_files/Clotho/train.json'
            # train_json = './data/json_files/Auto_ACD/train.json'
            # train_json = "./data/json_files/BBC_Sound_Effects/bbc_final.json"
            # train_json = './data/json_files/FreeSound/fsd_final.json'
            # train_json = 'data/json_files/SoundBible/sb_final.json'
            # embedding_path = "./data/embeddings/SoundBible"
            # with open('./data/json_files/Auto_ACD/train.json', 'r') as file:
            #     data = json.load(file)
            # for i, item in enumerate(data["data"]):
            #     filename = f"{i:07d}.npy"
            #     attention_score_filepath = os.path.join("./data/embeddings/Auto_ACD/", filename)
            #     item["attention_score"] = attention_score_filepath
            # with open('./data/json_files/Auto_ACD/train.json', 'w') as file:
            #     json.dump(data, file, indent=4)
                
            # with open('./data/json_files/AudioSet/train.json', 'r') as file:
            #     data = json.load(file)
            # for i, item in enumerate(data["data"]):
            #     filename = f"{i:07d}.npy"
            #     attention_score_filepath = os.path.join("./data/embeddings/AudioSet/", filename)
            #     item["attention_score"] = attention_score_filepath
            # with open('./data/json_files/AudioSet/train.json', 'w') as file:
            #     json.dump(data, file, indent=4)

        print(f"The train audio paths : {len(audio_file_paths)}")
        print(f"The processed embedding files : {max_index+1}")
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
