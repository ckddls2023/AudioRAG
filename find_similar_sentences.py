from sentence_transformers import SentenceTransformer
import json
import os
import faiss
import numpy as np
import torch
import librosa
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import gather_object
from laion_clap.training.data import (
    get_audio_features,
    int16_to_float32,
    float32_to_int16,
)
from sentence_transformers import SentenceTransformer

from models.align2text import align2text
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig

import ray
from ray.util.multiprocessing import Pool

ray.init(num_cpus=32)


# from angle_emb import AnglE, Prompts
# angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# angle.set_prompt(prompt=Prompts.C)
def average_embeddings(embeddings, window_size=5):
    reshaped_embeddings = embeddings.reshape(-1, window_size, embeddings.shape[1])
    averaged_embeddings = reshaped_embeddings.mean(axis=1)
    return averaged_embeddings


def average_dynamic_embeddings(embeddings, caption_counts):
    averaged_embeddings = []
    start_idx = 0
    for count in caption_counts:
        end_idx = start_idx + count
        avg_embedding = np.mean(embeddings[start_idx:end_idx], axis=0)
        averaged_embeddings.append(avg_embedding)
        start_idx = end_idx  # Update start index for the next set
    return np.array(averaged_embeddings)


def encode_texts(text_encoder, align_model, sentences, batch_size=256):
    with torch.no_grad():
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            captions = sentences[i : i + batch_size]
            text_embed = text_encoder.encode(
                captions, normalize_embeddings=True, convert_to_tensor=True
            )
            output = align_model(None, text_embed)
            embeddings.append(
                output["text_features"].detach().to("cpu", non_blocking=True).numpy()
            )
    torch.cuda.synchronize()
    return np.vstack(embeddings)


@ray.remote
def preprocess_audio_file(audio_path):
    audio_cfg = {
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
        "model_type": "HTSAT",
        "model_name": "base",
    }
    waveform, sr = librosa.load(audio_path, sr=48000)
    audio_waveform = int16_to_float32(float32_to_int16(waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    temp_dict = {}
    temp_dict = get_audio_features(
        temp_dict,
        audio_waveform,
        480000,  # 10s
        data_truncating="fusion",
        data_filling="repeatpad",
        audio_cfg=audio_cfg,
        require_grad=False,
    )
    return temp_dict


def encode_audio(audio_encoder, align_model, audio_paths, batch_size=256):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(audio_paths), batch_size)):
            audio_batch = audio_paths[i : i + batch_size]
            refs = [
                preprocess_audio_file.remote(audio_path) for audio_path in audio_batch
            ]
            audio_features = ray.get(refs)
            audio_embed = audio_encoder(audio_features).last_hidden_state  # B, 64, 768
            output = align_model(audio_embed, None)
            embeddings.append(
                output["audio_features"].detach().to("cpu", non_blocking=True).numpy()
            )
    torch.cuda.synchronize()
    return np.vstack(embeddings)


def encode_audio_texts(
    audio_encoder, text_encoder, align_model, audio_paths, sentences, batch_size=256
):
    audio_embeddings = []
    text_embeddings = []
    mixed_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            captions = sentences[i : i + batch_size]
            audio_batch = audio_paths[i : i + batch_size]
            refs = [
                preprocess_audio_file.remote(audio_path) for audio_path in audio_batch
            ]
            audio_features = ray.get(refs)
            audio_embed = audio_encoder(audio_features).last_hidden_state  # B, 64, 768
            text_embed = text_encoder.encode(
                captions, normalize_embeddings=True, convert_to_tensor=True
            )
            output = align_model(audio_embed, text_embed)
            mixed_embed = output["audio_features"] + output["text_features"]
            audio_embeddings.append(
                output["audio_features"].detach().to("cpu", non_blocking=True).numpy()
            )
            text_embeddings.append(
                output["text_features"].detach().to("cpu", non_blocking=True).numpy()
            )
            mixed_embeddings.append(
                mixed_embed.detach().to("cpu", non_blocking=True).numpy()
            )
    torch.cuda.synchronize()
    return (
        np.vstack(audio_embeddings),
        np.vstack(text_embeddings),
        np.vstack(mixed_embeddings),
    )  # 512


if __name__ == "__main__":
    # Load pre-trained model
    text_encoder = SentenceTransformer("all-mpnet-base-v2")
    # pool = model.start_multi_process_pool(target_devices=['cpu']*8)
    # pool = text_encoder.start_multi_process_pool(target_devices=['cuda:0','cuda:1','cuda:2','cuda:3'])
    encoder_config = {
        "model_name": "CLAPAudioEncoder",
        "pretrained": True,
        "freeze": True,
        "use_lora": True,
        "spec_augment": True,
        "select_feature": "fine_grained_embedding",
        "sequence_length": 1024,
        "hidden_size": 768,
        "window_size": 4,
        "step_size": 4,
    }
    encoder_config = CLAPEncoderConfig.from_dict(encoder_config)
    audio_encoder = CLAPAudioTower(encoder_config)
    align_model = align2text(hidden_size=768, num_latents=64, num_layers=2)
    checkpoint_path = "./retriever_models/"
    audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
    align_model_ckpt = os.path.join(checkpoint_path, "epoch_5.pt")
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

    # Function to encode sentences in batches
    # large
    # base
    train_jsons = [
        "data/json_files/BBC_Sound_Effects/bbc_final.json",
        "data/json_files/FreeSound/fsd_final.json",
        "data/json_files/SoundBible/sb_final.json",
        "data/json_files/AudioSet/train.json",
        #"data/json_files/AudioSet_SL/as_final.json",
        "data/json_files/Clotho/train.json",
        # "data/json_files/Auto_ACD/train.json",
    ] # huge

    val_jsons = [
        # "data/json_files/AudioSet/train.json",
        # "data/json_files/Clotho/train.json",
        "data/json_files/AudioSet/val.json",
        "data/json_files/Clotho/val.json",
        # "data/json_files/Auto_ACD/val.json",
        # "data/json_files/MACS/val.json",
    ]

    train_sentences = []
    train_audio_paths = []
    for train_json in train_jsons:
        with open(train_json, "r") as file:
            train_data = json.load(file)
            for entry in train_data["data"]:
                if entry["duration"] > 40 or entry["duration"] < 5:
                    continue  # Skip it
                if train_data["num_captions_per_audio"] > 1:
                    train_sentences.extend(entry["caption"])
                    train_audio_paths.extend([entry["audio"]] * len(entry["caption"]))
                else:
                    train_sentences.append(entry["caption"])
                    train_audio_paths.append(entry["audio"])

    # If it is last one, base, large, huge
    # "./data/index/final_atc_lm_attn/"
    # mixed_index_file_path = "./data/index/train_sentence_mixed_embed.bin"
    # text_index_file_path = "./data/index/train_sentence_audio_embed.bin"
    # audio_index_file_path = "./data/index/train_sentence_text_embed.bin"
    mixed_index_file_path = "./data/index/train_sentence_mixed_embed_largeKB.bin"
    text_index_file_path = "./data/index/train_sentence_audio_embed_largeKB.bin"
    audio_index_file_path = "./data/index/train_sentence_text_embed_largeKB.bin"
    # mixed_index_file_path = "./data/index/train_sentence_mixed_embed_hugeKB.bin"
    # text_index_file_path = "./data/index/train_sentence_audio_embed_hugeKB.bin"
    # audio_index_file_path = "./data/index/train_sentence_text_embed_hugeKB.bin"
    if os.path.exists(mixed_index_file_path):
        audio_index = faiss.read_index(audio_index_file_path)
        text_index = faiss.read_index(text_index_file_path)
        mixed_index = faiss.read_index(mixed_index_file_path)
    else:
        audio_embeddings, text_embeddings, mixed_embeddings = encode_audio_texts(
            audio_encoder, text_encoder, align_model, train_audio_paths, train_sentences
        )
        audio_index = faiss.IndexFlatIP(audio_embeddings.shape[1])
        audio_index.add(audio_embeddings)
        text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
        text_index.add(text_embeddings)
        mixed_index = faiss.IndexFlatIP(mixed_embeddings.shape[1])
        mixed_index.add(mixed_embeddings)
        faiss.write_index(audio_index, audio_index_file_path)
        faiss.write_index(text_index, text_index_file_path)
        faiss.write_index(mixed_index, mixed_index_file_path)

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

    # If we use sentence embedding
    # val_embeddings = encode_texts(text_encoder, val_sentences)
    # averaged_val_embeddings = average_dynamic_embeddings(
    #     val_embeddings, sentence_counts
    # )
    # print(f"Total audio paths in validation jsons : {len(val_audio_paths)}")
    # print(f"Total averaged val embeddings : {averaged_val_embeddings.shape[0]}")
    # k = 5  # Number of nearest neighbors to find
    # D, I = index.search(val_embeddings, k)
    # results = {}
    # for val_idx, neighbors in enumerate(I):
    #     val_audio = val_audio_paths[val_idx]
    #     similar_pairs = []
    #     for neighbor_idx in neighbors:
    #         train_audio = train_audio_paths[neighbor_idx]
    #         train_caption = train_sentences[neighbor_idx]
    #         similar_pairs.append([train_audio, train_caption])
    #     results[val_audio] = similar_pairs

    # Use ATC embedding
    print(f"Total audio paths in train jsons : {len(train_audio_paths)}")
    print(f"Total averaged train embeddings : {mixed_index.ntotal}")
    val_embeddings = encode_audio(audio_encoder, align_model, val_audio_paths)
    print(f"Total audio paths in validation jsons : {len(val_audio_paths)}")
    print(f"Total averaged val embeddings : {val_embeddings.shape[0]}")

    k = 5  # Number of nearest neighbors to find
    D, I = audio_index.search(val_embeddings, k)
    results = {}
    for val_idx, neighbors in enumerate(I):
        val_audio = val_audio_paths[val_idx]
        similar_pairs = []
        for neighbor_idx in neighbors:
            train_audio = train_audio_paths[neighbor_idx]
            train_caption = train_sentences[neighbor_idx]
            similar_pairs.append([train_audio, train_caption])
        results[val_audio] = similar_pairs


    # k = 2  # Number of nearest neighbors to find
    # MD, MI = mixed_index.search(val_embeddings, k)
    # AD, AI = audio_index.search(val_embeddings, k)
    # TD, TI = mixed_index.search(val_embeddings, k)
    # results = {}
    # for val_idx, val_audio_path in tqdm(enumerate(val_audio_paths)):
    #     similar_pairs = []
    #     m_neighbors = MI[val_idx]
    #     a_neighbors = AI[val_idx]
    #     t_neighbors = TI[val_idx]
    #     for m_idx, a_idx, t_idx in zip(m_neighbors, a_neighbors, t_neighbors):
    #         val_audio_basename = os.path.basename(val_audio_path)
    #         if (
    #             val_audio_path != train_audio_paths[m_idx]
    #             and os.path.basename(train_audio_paths[m_idx]) != val_audio_basename
    #         ):
    #             similar_pairs.append([train_audio_paths[m_idx], train_sentences[m_idx]])
    #         if (
    #             val_audio_path != train_audio_paths[a_idx]
    #             and os.path.basename(train_audio_paths[a_idx]) != val_audio_basename
    #         ):
    #             similar_pairs.append([train_audio_paths[a_idx], train_sentences[a_idx]])
    #         if (
    #             val_audio_path != train_audio_paths[t_idx]
    #             and os.path.basename(train_audio_paths[t_idx]) != val_audio_basename
    #         ):
    #             similar_pairs.append([train_audio_paths[t_idx], train_sentences[t_idx]])
    #     results[val_audio_path] = similar_pairs

    # Save results to JSON
    with open("retrieved_results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    ray.shutdown()