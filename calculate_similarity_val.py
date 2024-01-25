
import laion_clap
import glob
import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from models.align2text import align2text
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig
# from find_similar_sentences import encode_texts, encode_audio
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig, PromptEncoderConfig, AdaptionPromptConfig

# Load the model : Laion-CLAP
device = torch.device('cuda:0')
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()

# text_encoder = SentenceTransformer("all-mpnet-base-v2")
# encoder_config = {
#     "model_name": "CLAPAudioEncoder",
#     "pretrained": True,
#     "freeze": True,
#     "use_lora": True,
#     "spec_augment": False,
#     "select_feature": "fine_grained_embedding",
#     "sequence_length": 1024,
#     "hidden_size": 768,
#     "window_size": 4,
#     "step_size": 4,
# }
# encoder_config = CLAPEncoderConfig.from_dict(encoder_config)
# audio_encoder = CLAPAudioTower(encoder_config)
# align_model = align2text(hidden_size=768, num_latents=64, num_layers=1)

# checkpoint_path =  "./retriever_models_lm_attn4/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_9.pt")
# sentence_peft_config = {
#     'r': 16,
#     'lora_alpha': 16,
#     'lora_dropout': 0.1,
#     'bias': "none",
#     'task_type': "MPNetForMaskedLM",
#     'modules_to_save': [],
#     'target_modules': ["attn.q", "attn.k", "attn.v","attn.o","pooler.dense"]
# }
# peft_config = LoraConfig(**sentence_peft_config)
# text_encoder[0].auto_model = PeftModel.from_pretrained(text_encoder[0].auto_model, checkpoint_path, config=peft_config)  # suppose don't use get_peft_model
# checkpoint_path =  "./retriever_models_lm_attn3/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_3.pt")
# checkpoint_path =  "./retriever_models_lm_attn2/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_12.pt")
# align_model = align2text(hidden_size=768, num_latents=64, num_layers=2)
# checkpoint_path = "./retriever_models_lm_attn/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_5.pt")
# checkpoint_path = "./retriever_models_lm_attn/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_15.pt")

# audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
# if os.path.exists(audio_encoder_ckpt):
#     audio_encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=False)
# if os.path.exists(align_model_ckpt):
#     align_model.load_state_dict(torch.load(align_model_ckpt), strict=True)
# text_encoder = text_encoder.to("cuda")
# audio_encoder = audio_encoder.to("cuda")
# align_model = align_model.to("cuda")
# text_encoder.eval()
# audio_encoder.eval()
# align_model.eval()

val_jsons = [
    "data/json_files/AudioSet/val.json",
    # "data/json_files/AudioSet/test.json",
    # "data/json_files/Clotho/val.json",
    # "data/json_files/Auto_ACD/val.json",
    # "data/json_files/MACS/val.json",
]

val_sentences = []
val_audio_paths = []
sentence_counts = []
for val_json in val_jsons:
    with open(val_json, "r") as file:
        val_data = json.load(file)
        for entry in val_data["data"]:
            if val_data["num_captions_per_audio"] > 1:
                val_sentences.extend(entry["caption"])
                val_audio_paths.append(entry["audio"])
                sentence_counts.append(len(entry["caption"]))
            else:
                val_sentences.append(entry["caption"])
                val_audio_paths.append(entry["audio"])
                sentence_counts.append(1)
print("audio_files: ", len(val_audio_paths))
print("text_files: ", len(val_sentences))

# Load retrieval results
# with open("./data/index/audiocaps_clotho_sentence_embed.json", "r") as file:
#     retrieval_results = json.load(file)
    
# with open("./data/index/final_atc_lm_attn/audiocaps_clotho_audio2audio_baseKB.json", "r") as file:
#     retrieval_results = json.load(file)
    
# with open("./data/index/final_index_big_kb/audio2audio_val.json", "r") as file:
#     retrieval_results = json.load(file)

with open("./data/index/final_index_big_kb/audio2text_val.json", "r") as file:
    retrieval_results = json.load(file)
    
retr_sentences = [result[1] for audio_path in val_audio_paths for result in retrieval_results[audio_path]]
retr_audio_paths = [result[0] for audio_path in val_audio_paths for result in retrieval_results[audio_path]]

with torch.no_grad():

    # text_embed = model.get_text_embedding(val_sentences)
    chunk_size = min(len(val_sentences) // 5, 100)
    text_embed_list = []
    for i in range(int(len(val_sentences)//chunk_size)+1):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_sentences = val_sentences[start_idx:end_idx]
        chunk_text_embed = model.get_text_embedding(chunk_sentences)
        text_embed_list.append(chunk_text_embed)
    text_embed = np.concatenate(text_embed_list, axis=0)
    # audio_embed = model.get_audio_embedding_from_filelist(x=val_audio_paths)
    chunk_size = min(len(val_audio_paths) // 5, 100)
    audio_embed_list = []
    for i in range(int(len(val_audio_paths)//chunk_size)+1):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_audio_paths = val_audio_paths[start_idx:end_idx]
        chunk_audio_embed = model.get_audio_embedding_from_filelist(x=chunk_audio_paths)
        audio_embed_list.append(chunk_audio_embed)
    audio_embed = np.concatenate(audio_embed_list, axis=0)
    # text_embed = text_embed.view(-1, 5, 512).mean(dim=1) # [5*B] => [B,512]
    text_embed = text_embed.reshape(-1, 5, 512).mean(axis=1)
    # text_embed = encode_texts(text_encoder, align_model, val_sentences)
    # audio_embed, _, _ = encode_audio(audio_encoder, align_model, val_audio_paths)
    # print(text_embed.shape)
    # print(audio_embed.shape)
    
    # retr_text_embed = model.get_text_embedding(retr_sentences)
    chunk_size = min(len(retr_sentences) // 5, 100)
    retr_text_embed_list = []
    for i in range(int(len(retr_sentences)//chunk_size)+1):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_sentences = retr_sentences[start_idx:end_idx]
        chunk_text_embed = model.get_text_embedding(chunk_sentences)
        retr_text_embed_list.append(chunk_text_embed)
    retr_text_embed = np.concatenate(retr_text_embed_list, axis=0)
    retr_text_embed = retr_text_embed.reshape(-1, 5, 512)
    # retr_audio_embed = model.get_audio_embedding_from_filelist(x=retr_audio_paths)
    chunk_size = min(len(retr_audio_paths) // 5, 100)
    retr_audio_embed_list = []
    for i in range(int(len(retr_audio_paths)//chunk_size)+1):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_audio_paths = retr_audio_paths[start_idx:end_idx]
        chunk_audio_embed = model.get_audio_embedding_from_filelist(x=chunk_audio_paths)
        retr_audio_embed_list.append(chunk_audio_embed)
    retr_audio_embed = np.concatenate(retr_audio_embed_list, axis=0)
    retr_audio_embed = retr_audio_embed.reshape(-1, 5, 512)
    print(text_embed.shape)
    print(audio_embed.shape)
    print(retr_text_embed.shape)
    print(retr_audio_embed.shape)
    # text_inner_product = text_embed.unsqueeze(1) * retr_text_embed.view(-1,5,512)
    # text_similarity = text_inner_product.sum(dim=-1).mean(dim=-1) # [B,5,512] -> [B,5] -> [B]
    text_inner_product = text_embed[:, np.newaxis, :] * retr_text_embed
    print(text_inner_product.shape)
    text_similarity = text_inner_product.sum(axis=-1).mean(axis=-1)
    # audio_inner_product = audio_embed.unsqueeze(1) * retr_audio_embed.view(-1,5,512)
    # audio_similarity = audio_inner_product.sum(dim=-1).mean(dim=-1) # [B,5,512] -> [B,5] -> [B]
    audio_inner_product = audio_embed[:, np.newaxis, :] * retr_audio_embed
    print(audio_inner_product.shape)
    audio_similarity = audio_inner_product.sum(axis=-1).mean(axis=-1)
    audio2text_inner_product = audio_embed[:, np.newaxis, :] * retr_text_embed
    audio2text_similarity = audio2text_inner_product.sum(axis=-1).mean(axis=-1)
    text2audio_similarity = text_embed[:, np.newaxis, :] * retr_audio_embed
    text2audio_similarity = text2audio_similarity.sum(axis=-1).mean(axis=-1)
    
    print(f"(val<->retr)Audio2Audio Similarity : {audio_similarity.mean()}")
    print(f"(val<->retr)Text2Text Similarity : {text_similarity.mean()}")
    print(f"(val<->retr)Audio2Text Similarity : {audio2text_similarity.mean()}")
    print(f"(val<->retr)Text2Audio Similarity : {text2audio_similarity.mean()}")
