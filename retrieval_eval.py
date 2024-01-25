
import laion_clap
import glob
import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from models.align2text import align2text
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig
from find_similar_sentences import encode_texts, encode_audio
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig, PromptEncoderConfig, AdaptionPromptConfig

device = torch.device('cuda:0')

# Load the model : Laion-CLAP
# model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
# model.load_ckpt()

text_encoder = SentenceTransformer("all-mpnet-base-v2")
encoder_config = {
    "model_name": "CLAPAudioEncoder",
    "pretrained": True,
    "freeze": True,
    "use_lora": True,
    "spec_augment": False,
    "select_feature": "fine_grained_embedding",
    "sequence_length": 1024,
    "hidden_size": 768,
    "window_size": 4,
    "step_size": 4,
}
encoder_config = CLAPEncoderConfig.from_dict(encoder_config)
audio_encoder = CLAPAudioTower(encoder_config)
align_model = align2text(hidden_size=768, num_latents=64, num_layers=1)
checkpoint_path =  "./retriever_models_lm_attn4/"
align_model_ckpt = os.path.join(checkpoint_path, "epoch_14.pt")
sentence_peft_config = {
    'r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'bias': "none",
    'task_type': "MPNetForMaskedLM",
    'modules_to_save': [],
    'target_modules': ["attn.q", "attn.k", "attn.v","attn.o","pooler.dense"]
}
peft_config = LoraConfig(**sentence_peft_config)
text_encoder[0].auto_model = PeftModel.from_pretrained(text_encoder[0].auto_model, checkpoint_path, config=peft_config)  # suppose don't use get_peft_model
# checkpoint_path =  "./retriever_models_lm_attn3/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_3.pt")
# checkpoint_path =  "./retriever_models_lm_attn2/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_12.pt")
# align_model = align2text(hidden_size=768, num_latents=64, num_layers=2)
# checkpoint_path = "./retriever_models_lm_attn/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_5.pt")
# checkpoint_path = "./retriever_models_lm_attn/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_15.pt")
audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
if os.path.exists(audio_encoder_ckpt):
    audio_encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=False)
if os.path.exists(align_model_ckpt):
    align_model.load_state_dict(torch.load(align_model_ckpt), strict=True)
text_encoder = text_encoder.to("cuda")
audio_encoder = audio_encoder.to("cuda")
align_model = align_model.to("cuda")
text_encoder.eval()
audio_encoder.eval()
align_model.eval()

val_jsons = [
    # "data/json_files/AudioSet/val.json",
    "data/json_files/AudioSet/test.json",
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
device = torch.device('cuda:0')
print("audio_files: ", len(val_audio_paths))
print("text_files: ", len(val_sentences))
ground_truth_idx = [[i]*sentence_count for i, sentence_count in enumerate(sentence_counts)]

with torch.no_grad():
    expanded_ground_truth = torch.tensor(ground_truth_idx).flatten()

    # text_embed = model.get_text_embedding(val_sentences)
    # audio_embed = model.get_audio_embedding_from_filelist(x=val_audio_paths)
    text_embed = encode_texts(text_encoder, align_model, val_sentences, batch_size=128)
    audio_embed, _, _ = encode_audio(audio_encoder, align_model, val_audio_paths, batch_size=128)
    print(text_embed.shape)
    print(audio_embed.shape)
    
    top_k = 10
    top_k_indices = torch.topk(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), k=top_k, dim=0).indices
    
    # Calculate recall at k
    recall_at_k = {}
    for k in [1, 5, 10]:
        correct_at_k = top_k_indices[:k, :].eq(expanded_ground_truth.unsqueeze(0)).any(0)
        recall_at_k[f'R@{k}'] = correct_at_k.float().mean().item()

    print("Recall at K:", recall_at_k)
