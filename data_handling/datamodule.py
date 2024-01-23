import torch
from torch import Tensor
from typing import Optional, List, Tuple
import numpy as np
from torch.nn.utils.rnn import pad_sequence
# from pytorch_lightning import LightningDataModule
from data_handling.caption_dataset import AudioCaptionDataset
from torch.utils.data import DistributedSampler, DataLoader
import torch.nn.functional as F


class AudioCaptionDataModule:

    def __init__(self,
                 config: dict,
                 dataset: str
                 ):
        super(AudioCaptionDataModule, self).__init__()

        audio_config = config["audio_args"]
        self.train_set = AudioCaptionDataset(audio_config,
                                             dataset,
                                             split="train")

        self.val_set = AudioCaptionDataset(audio_config,
                                           dataset,
                                           split="val")

        self.test_set = AudioCaptionDataset(audio_config,
                                            dataset,
                                            split="test")

        self.batch_size = config["data_args"]["batch_size"]
        self.num_workers = config["data_args"]["num_workers"]

    def _get_sampler(self,
                     dataset,
                     shuffle,
                     is_distributed,
                     num_tasks,
                     global_rank):
        # do not return a sampler if is not in distributed mode
        # a default RandomSampler is used in this case
        if not is_distributed:
            return None

        return DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )

    def train_dataloader(self,
                         is_distributed=False,
                         num_tasks=0,
                         global_rank=0):
        sampler = self._get_sampler(
            dataset=self.train_set,
            shuffle=True,
            is_distributed=is_distributed,
            num_tasks=num_tasks,
            global_rank=global_rank)
        shuffle = sampler is None

        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          sampler=None,
                          shuffle=False,
                          collate_fn=collate_fn,
                          drop_last=False
                          )

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          sampler=None,
                          shuffle=False,
                          collate_fn=collate_fn,
                          drop_last=False
                          )


def collate_fn(batch):
    audio_features, captions, audio_paths, retr_audio_features, retr_captions, embeddings = zip(*batch)
    max_length = max([wav.shape[-1] for wav in audio_features])
    audio_features = [np.pad(wav, pad_width=(0, max_length - wav.shape[-1]), mode='constant', constant_values=0.0) for wav in audio_features]
    if retr_audio_features[0]:
        retr_audio_features = list(map(list, zip(*retr_audio_features))) # B, top_k to top_k, B
        retr_captions = list(map(list, zip(*retr_captions)))
    else:
        retr_audio_features = []
        retr_captions = []
    # embeddings = np.stack(embeddings)
    # embeddings = torch.from_numpy(embeddings)
    return audio_features, captions, audio_paths, retr_audio_features, retr_captions, embeddings
