#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import random
import librosa
import torch
import ruamel.yaml as yaml
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from data_handling.datamodule import collate_fn
from data_handling.sampler import BySequenceLengthSampler, BySequenceBatchSampler
from data_handling.text_transform import text_preprocess


def load_json_file(files, blacklist=None):
    json_data = []
    audio_id = 0
    if blacklist is not None:
        with open(blacklist, 'r') as f:
            blacklist = json.load(f)
    for file in files:
        with open(file, "r") as f:
            json_obj = json.load(f)
            if json_obj["num_captions_per_audio"] == 1:
                for item in json_obj["data"]:
                    if "FreeSound" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    elif "AudioSet" in file and blacklist is not None:
                        if item["id"] in blacklist["AudioSet"]:
                            continue
                    if item["duration"] > 40.0: # Avoid too much long audios
                        continue
                    temp_dict = {"audio": item["audio"], "caption": item["caption"], "id": audio_id,
                                 "duration": item["duration"]}
                    json_data.append(temp_dict)
                    audio_id += 1
            else:
                for item in json_obj["data"]: # Suppose only stands for validation (AudioCaps, CLOTHO_v2.1)
                    json_data.append(item)
                    audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, json_files, audio_config, blacklist=None):

        self.json_data = load_json_file(json_files, blacklist)
        self.lengths = [item["duration"] for item in self.json_data]

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):

        item = self.json_data[index]
        wav_path = item["audio"]
        duration = item["duration"]
        waveform, sr = librosa.load(wav_path, sr=self.sr, mono=True, duration=duration)

        if self.max_length != 0:
            # if audio length is longer than max_length, we randomly crop it to mac length
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + self.max_length]

        caption = text_preprocess(item["caption"])
        return torch.tensor(waveform), caption, wav_path


def pretrain_dataloader(config,
                        subset: str = "train_jsons",
                        bucket: bool = True,
                        bucket_boundaries: tuple = (5, 30, 6),
                        is_distributed: bool = False,
                        num_tasks: int = 0,
                        global_rank: int = 0,
                        shuffle=True):
    blacklist = None if 'val' in subset else config.blacklist
    batch_size = 8 if 'val' in subset else config.data_args.batch_size
    dataset = AudioLanguagePretrainDataset(config[subset], config["audio_args"], blacklist)
    if bucket:
        sampler = BySequenceLengthSampler(lengths=dataset.lengths,
                                          bucket_boundaries=bucket_boundaries,
                                          batch_size=batch_size,
                                          drop_last=True,
                                          seed=config["seed"])
        return DataLoader(dataset=dataset,
                          batch_sampler=BySequenceBatchSampler(sampler, batch_size=batch_size, drop_last=False),
                          shuffle=shuffle,
                          pin_memory=True,
                          num_workers=config["data_args"]["num_workers"],
                          collate_fn=collate_fn)
    elif is_distributed:
        sampler = DistributedSampler(dataset,
                                     num_replicas=num_tasks,
                                     rank=global_rank,
                                     shuffle=True)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["data_args"]["num_workers"],
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collate_fn,
    )


if __name__ == '__main__':

    with open("../../WavCaps/captioning/settings/pretrain.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(config)
