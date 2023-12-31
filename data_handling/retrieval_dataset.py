import types
import faiss
import librosa
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from laion_clap import CLAP_Module
from laion_clap.training.data import int16_to_float32, float32_to_int16, get_audio_features

# 필요하다면 나중에 utils 파일에 넣어도 되는 함수.
def load_caption_wav_mapping(csv_path):
    df = pd.read_csv(csv_path)
    return df['caption'], df['wav_path']
    
class RetrievalIndex:
    def __init__(self, n_probe=16, index_path="./data/index/audio_faiss_index.bin", top_k=3, query_mode="audio2audio", device=None):
        self.query_mode = query_mode
        self.datastore = faiss.read_index(index_path)
        self.datastore.nprobe = n_probe
        self.captions, self.wav_paths = load_caption_wav_mapping(str(Path(index_path).parent /"caption_wav_path.csv"))

        # Very redundant and should be avoided, currently
        self.clap = CLAP_Module(enable_fusion=True, device = device)  # 615M
        self.clap.load_ckpt()
        self.clap.eval()

        if self.query_mode == "frame2audio":
            def get_audio_embedding_before_projection(self, data):
                """Get the audio embedding from the model

                Parameters
                ----------
                data: a list of dict
                    the audio input dict list from 'get_audio_feature' method

                Returns
                ----------
                audio_embed: torch.Tensor
                    a tensor of audio_embeds (N, D)

                """
                device = next(self.parameters()).device
                input_dict = {}
                keys = data[0].keys()
                for k in keys:
                    input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
                audio_embeds = self.encode_audio(input_dict, device=device)["fine_grained_embedding"]
                return audio_embeds
            self.clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_before_projection, self.clap.model)
        self.top_k = top_k
        self.query_mode = query_mode
        self.audio_cfg = {
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
            "model_name": "base"
        }

        # if device: # use_gpu
        #     co = faiss.GpuMultipleClonerOptions()
        #     co.useFloat16 = True
        #     co.useFloat16CoarseQuantizer = True
        #     co.usePrecomputed = False
        #     co.indicesOptions = faiss.INDICES_32_BIT
        #     co.verbose = True
        #     co.shard = False  # the replicas will be made "manually"
        #     res = [faiss.StandardGpuResources() for i in range(faiss.get_num_gpus())]
        #     self.datastore = faiss.index_cpu_to_gpu_multiple_py(res, self.datastore, co)
        #     faiss.GpuParameterSpace().set_index_parameter(self.datastore, 'nprobe', n_probe)

    def is_index_trained(self) -> bool:
        return self.datastore.is_trained

    # modal에 따른 query embedding을 만들어주는 함수
    def query_embedding(self, samples):
        if all(isinstance(item, str) for item in samples): # If it's text, but we don't have any cases that use text
            text_embed = self.clap.get_text_embedding(samples, use_tensor=True)
            text_embed = torch.unsqueeze(text_embed, 0)
            return text_embed
        else:
            audio_embed = self.clap.model.get_audio_embedding(samples)
            if self.query_mode == "frame2audio":
                chunks = audio_embed.chunk(4, dim=1)
                averaged_chunks = [chunk.mean(dim=1, keepdim=True) for chunk in chunks]
                audio_embed = torch.cat(averaged_chunks, dim=1).permute(1, 0, 2) # S, B, 768 -> EX. 4, 2, 768
            return audio_embed

    def get_nns(self, queries):
        """
        Retrieves nearest neighbors for given queries from the datastore.

        Args:
            queries (Tensor): The batch of queries for which nearest neighbors are to be found.

        Returns:
            - D (List[List[float]]): A lists of distances between each query and its nearest neighbors.
            - I (List[List[int]]): A list of lists containing indices of the nearest neighbors for each query.
            - texts (List[List[str]]): A list of lists of strings, each list containing the captions corresponding to the nearest neighbor indices for each query.
            - audio_samples (List[Tensor]): A list of tensors, each tensor is an audio sample loaded and transformed into a PyTorch tensor. Audio samples are loaded with librosa at a sampling rate of 48000, mono, and truncated to a duration of 10 seconds.

        Note:
            The method assumes the datastore, captions, and wav_paths attributes are already set in the class.
        """
        queries_embed = self.query_embedding(queries).detach().cpu().numpy() # GPU Query
        D, I = self.datastore.search(queries_embed, self.top_k)  # In torch, it may return LongTensor
        I_transposed = list(map(list, zip(*I)))  # B,top_k -> top_k, B
        texts = [self.captions[idx_list].tolist() for idx_list in I_transposed]
        audio_paths = [self.wav_paths[idx_list].tolist() for idx_list in I_transposed]
        audio_sample_batchs = []
        for audio_path in audio_paths:  # Batch of list k=1, k=2, k=3
            audio_samples = []
            for wav_path in audio_path:
                waveform, sr = librosa.load(wav_path, sr=self.audio_cfg["sample_rate"], mono=True)
                audio_waveform = int16_to_float32(float32_to_int16(waveform))
                audio_waveform = torch.from_numpy(audio_waveform).float()
                temp_dict = {}
                temp_dict = get_audio_features(
                    temp_dict, audio_waveform, 480000,
                    data_truncating='fusion',
                    data_filling='repeatpad',
                    audio_cfg=self.audio_cfg,
                    require_grad=False,
                )
                audio_samples.append(temp_dict)
            audio_sample_batchs.append(audio_samples)
        return D, I, texts, audio_sample_batchs

if __name__ == "__main__":
    text_data = ["a dog is barking at a man walking by", "Wind and a man speaking are heard, accompanied by buzzing and ticking."]
    audio_files = ["./examples/yapping-dog.wav", "./examples/Yb0RFKhbpFJA.flac"]
    
    device = 'cuda:3'
    
    # 1번 경우, audio2audio, audio2text: topk = 4
    # 2번 경우, frame2audio: topk = 1 -> 각 frame의 top1이 모여서 4개 리턴.
    # 1번, 2번 경우 모두 batch x topk -> (2,4) 리턴합니다.
    index = RetrievalIndex(n_probe=16, index_path="./data/original_pretrain_index", top_k=4, query_mode="audio2text", device = device)
    audio_samples = [torch.tensor(librosa.load(audio_file, sr=48000, mono=True, duration=10)[0]) for audio_file in audio_files]

    audio_query_embedding = index.query_embedding(audio_samples)
    # text_query_embedding = index.query_embedding(text_data, query_mode)
    
    index.get_nns(audio_query_embedding, device = device)