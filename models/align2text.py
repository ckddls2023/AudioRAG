import torch
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from models.Qformer import *

class align2text(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_latents=256,
                 num_layers=2,
                 ):
        super().__init__()
        self.num_latents=num_latents
        self.hidden_size = hidden_size
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = num_layers
        config.encoder_width = self.hidden_size
        config.add_cross_attention = False
        config.cross_attention_freq = 1
        config.query_length = num_latents # number of latents
        self.align = BertLMHeadModel(config)   # cross-attention with audio token
        self.align.cls = None
        self.align.bert.embeddings.word_embeddings = None
        self.align.bert.embeddings.position_embeddings = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.cls_token.data.normal_(mean=0.0, std=config.initializer_range)
        #self.audio_projection = nn.Linear(hidden_size, hidden_size)
        # self.audio_ln = nn.LayerNorm(hidden_size)  # LayerNorm for audio features
        # self.text_ln = nn.LayerNorm(hidden_size)   # LayerNorm for text features
        self.audio_projection = nn.Linear(hidden_size, 512)
        self.text_projection = nn.Linear(hidden_size, 512)
        
    def forward(self, audio_embed, text_embed, lm_attn=None, output_attentions=False):
        if self.training:
            cls_tokens = self.cls_token.expand(audio_embed.shape[0], -1, -1)  # Replicating cls_token for the batch
            audio_embed = torch.cat((cls_tokens, audio_embed), dim=1)  # Concatenating along the sequence dimension
            output = self.align.bert(
                query_embeds=audio_embed,  # [B, 64, H]
                output_attentions=True,
                return_dict=True,
            )
            cls_output = output.last_hidden_state[:, 0, :]  # Extracting the CLS token output
            cls_output = F.normalize(cls_output, p=2, dim=1) # L2 normalize, struggles with converge
            audio_features = self.audio_projection(cls_output)  # Projecting the CLS token output
            text_features = self.text_projection(text_embed)
            attn_score = torch.concat(output.attentions, dim=1)  # [[B,nH,S,S], [B,nH,S,S]] # WARN: remember to use output_attentions=True
            cls_attn = attn_score[:, :, 0, 1:] # [B,2*nH,1,256]}
            grouped_cls_attn = cls_attn.view(cls_attn.shape[0], 2, 12, 256)  # [B,2,12, :, :]
            averaged_cls_attn = grouped_cls_attn.mean(dim=2)  # [B,2,64] # => CLS token except
            #averaged_cls_attn = averaged_cls_attn[:,0,:]
            if lm_attn is not None:
                lm_attn = F.normalize(lm_attn, p=1, dim=-1) # since it's softmax score/ check....
                # cls_attn_log = F.log_softmax(averaged_cls_attn, dim=-1) # B,nH,S / really critical...  / 100
                cls_attn_log = torch.log(F.normalize(averaged_cls_attn, p=1, dim=-1)) # since it's softmax score/ check....
                kl_loss = F.kl_div(cls_attn_log, lm_attn, reduction='mean')  # or 'sum', 'mean', or 'none'
            
            # BYOL loss style
            # z_audio = F.normalize(audio_features, dim=1)
            # z_text = F.normalize(text_embed, dim=1)
            # loss = F.mse_loss(z_audio, z_text)
            # return {
            #     'loss': loss
            # }

            # Cosine similarity
            # audio_features = F.normalize(audio_features, p=2, dim=1)
            # cosine_sim = torch.sum(audio_features * text_embed, dim=1) / self.temperature
            # loss = 1 - torch.mean(cosine_sim)
            # return {
            #     'loss': loss
            # }

            # LiT Contrastive Style
            all_audio_features = torch.cat(torch.distributed.nn.all_gather(audio_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            logits_per_audio = all_audio_features @ all_text_features.T
            logits_per_text = logits_per_audio.T
            labels = torch.arange(0, logits_per_audio.shape[0]).to(logits_per_audio.device)
            total_loss = (
                F.cross_entropy(logits_per_audio, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2
            if lm_attn is not None:
                total_loss += 0.01 * kl_loss
            return {
                "loss": total_loss
            }
        else:
            audio_features = None
            text_features = None
            if audio_embed is not None:
                cls_tokens = self.cls_token.expand(audio_embed.shape[0], -1, -1)  # Replicating cls_token for the batch
                audio_embed = torch.cat((cls_tokens, audio_embed), dim=1)  # Concatenating along the sequence dimension
                output = self.align.bert(
                    query_embeds=audio_embed,  # [B, 64, H]
                    output_attentions=True,
                    return_dict=True,
                )
                cls_output = output.last_hidden_state[:, 0, :]  # Extracting the CLS token output
                audio_features = self.audio_projection(cls_output)  # Projecting the CLS token output
            if text_embed is not None:
                text_features = self.text_projection(text_embed)
            out = { # For evaluation
                "audio_features": audio_features,
                "text_features": text_features
            }
            if output_attentions:
                attn_score = torch.concat(output.attentions, dim=1)  # [[B,nH,S,S], [B,nH,S,S]] # WARN: remember to use output_attentions=True
                cls_attn = attn_score[:, :, 0, 1:] # [B,2*nH,1,256]}
                grouped_cls_attn = cls_attn.view(cls_attn.shape[0], 2, 12, 256)  # [B,2,12,256]
                averaged_cls_attn = grouped_cls_attn.mean(dim=2)  # [B,2,256] # => CLS token except
                out["cls_attn"] = averaged_cls_attn.mean(dim=1) # [B,256]
            return out
