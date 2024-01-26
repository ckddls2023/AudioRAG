import laion_clap
import glob
import json
import os
import torch
import numpy as np
from msclap import CLAP
from sentence_transformers import SentenceTransformer
from models.align2text import align2text
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig
from find_similar_sentences import encode_texts, encode_audio
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig, PromptEncoderConfig, AdaptionPromptConfig

device = torch.device('cuda:0')

# Load the model : Laion-CLAP
# model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
# model.load_ckpt()

# text_encoder = sentencetransformer("all-mpnet-base-v2")
# encoder_config = {
#     "model_name": "clapaudioencoder",
#     "pretrained": true,
#     "freeze": true,
#     "use_lora": true,
#     "spec_augment": false,
#     "select_feature": "fine_grained_embedding",
#     "sequence_length": 1024,
#     "hidden_size": 768,
#     "window_size": 4,
#     "step_size": 4,
# }
# encoder_config = clapencoderconfig.from_dict(encoder_config)
# audio_encoder = clapaudiotower(encoder_config)
# align_model = align2text(hidden_size=768, num_latents=64, num_layers=1)
# checkpoint_path =  "./retriever_models_lm_attn4/"
# align_model_ckpt = os.path.join(checkpoint_path, "epoch_15.pt")
# sentence_peft_config = {
#     'r': 16,
#     'lora_alpha': 16,
#     'lora_dropout': 0.1,
#     'bias': "none",
#     'task_type': "mpnetformaskedlm",
#     'modules_to_save': [],
#     'target_modules': ["attn.q", "attn.k", "attn.v","attn.o","pooler.dense"]
# }
# peft_config = loraconfig(**sentence_peft_config)
# text_encoder[0].auto_model = peftmodel.from_pretrained(text_encoder[0].auto_model, checkpoint_path, config=peft_config)  # suppose don't use get_peft_model
# # checkpoint_path =  "./retriever_models_lm_attn3/"
# # align_model_ckpt = os.path.join(checkpoint_path, "epoch_3.pt")
# # checkpoint_path =  "./retriever_models_lm_attn2/"
# # align_model_ckpt = os.path.join(checkpoint_path, "epoch_12.pt")
# # align_model = align2text(hidden_size=768, num_latents=64, num_layers=2)
# # checkpoint_path = "./retriever_models_lm_attn/"
# # align_model_ckpt = os.path.join(checkpoint_path, "epoch_5.pt")
# # checkpoint_path = "./retriever_models_lm_attn/"
# # align_model_ckpt = os.path.join(checkpoint_path, "epoch_15.pt")
# audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
# if os.path.exists(audio_encoder_ckpt):
#     audio_encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=false)
# if os.path.exists(align_model_ckpt):
#     align_model.load_state_dict(torch.load(align_model_ckpt), strict=true)
# text_encoder = text_encoder.to("cuda")
# audio_encoder = audio_encoder.to("cuda")
# align_model = align_model.to("cuda")
# text_encoder.eval()
# audio_encoder.eval()
# align_model.eval()

clap_model = CLAP(version = '2023', use_cuda=True)


device = torch.device('cuda:0')
esc50_test_dir = './data/ESC50/test/'
class_index_dict_path = './data/ESC50/class_labels/ESC50_class_labels_indices_space.json'

audio_files = sorted(glob.glob(esc50_test_dir + 'waveforms/*.flac', recursive=True))
json_files = sorted(glob.glob(esc50_test_dir + 'json_files/*.json', recursive=True))
print("audio_files: ", len(audio_files))
print("json_files: ", len(json_files))
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}
ground_truth_idx = [class_index_dict[json.load(open(jf))['tag'][0]] for jf in json_files]

with torch.no_grad():
    ground_truth = torch.tensor(ground_truth_idx).view(-1, 1).cuda()

    all_texts = ["this is a sound of " + t for t in class_index_dict.keys()]
    # text_embed = model.get_text_embedding(all_texts)
    # audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)
    text_embeddings = clap_model.get_text_embeddings(all_texts)
    audio_embeddings = clap_model.get_audio_embeddings(audio_files)
    similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    # text_embed = encode_texts(text_encoder, align_model, all_texts)
    # audio_embed, _, _ = encode_audio(audio_encoder, align_model, audio_files)
    
    # load the model : ours

    # ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)
    ranking = torch.argsort(similarities, descending=True)
    preds = torch.where(ranking == ground_truth)[1]
    preds = preds.cpu().numpy()

    metrics = {}
    metrics[f"mean_rank"] = preds.mean() + 1
    metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

    print(
        f"Zeroshot Classification Results: "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )