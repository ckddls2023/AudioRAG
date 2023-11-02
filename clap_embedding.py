from msclap import CLAP
import librosa
import numpy as np
import librosa
import torch
import laion_clap
from datasets import load_dataset
from transformers import AutoProcessor, ClapAudioModel, ClapModel, ClapProcessor

text_file = ["./examples/yapping-dog.txt"]
audio_file = ["./examples/yapping-dog.wav"]

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAP(version = '2023', use_cuda=True)

# Extract text embeddings
text_embeddings = clap_model.get_text_embeddings(text_file)
print(text_embeddings.shape)

# Extract audio embeddings
audio_embeddings = clap_model.get_audio_embeddings(audio_file)
print(audio_embeddings.shape) # (1,1024)

# Compute similarity between audio and text embeddings
similarities = clap_model.compute_similarity(audio_embeddings, text_embeddings)
print(similarities)

audio_data, _ = librosa.load('./examples/yapping-dog.wav', sr=48000) # sample rate should be 48000
audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
model = laion_clap.CLAP_Module(enable_fusion=False) # 615M
model.load_ckpt() # download the default pretrained checkpoint.
audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
print(audio_embed.shape)   # (1,512)

# # Extract audio embeddings
# audio_data input format : (1, 1, T)
# Better to API
# def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):

dataset = load_dataset("ashraq/esc50")
audio_sample = dataset["train"]["audio"][0]["array"]
print(audio_sample.shape)
model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused") # 615M
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
inputs = processor(audios=audio_sample, return_tensors="pt")
outputs = model(**inputs)
pooler_output = outputs.pooler_output
print(pooler_output.shape) # (1, 768)

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = librispeech_dummy[0]
model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(0)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt").to(0)
audio_embed = model.get_audio_features(**inputs)
print(audio_embed.shape) # (1,512) => after projection
