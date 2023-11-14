# A.K.A Language Guided Token Merger

import torch
from torch import nn
from torch.nn import TransformerDecoderLayer
from transformers import T5EncoderModel



# Language Guided Token Selector
# Do we need Gumbel Relaxation here? or Gated Networks on MoE?
# Or make Switch Transformer Network?, EViT
class LGTS(nn.Module):
    def __init__(self,num_latents=64):
        super().__init__()
        self.num_latents = num_latents

    def forward(self, audio_embed, text_embed):
        logits = torch.einsum("bac,btc->bat",audio_embed, text_embed)
        logits_per_img_feat = logits.max(-1)[0]
        topk_proposals_idx = torch.topk(logits_per_img_feat,self.num_latents, dim=1)[1]
        pass

class LGTM(nn.Module):
    def __init__(self,
                 dim=768,
                 num_latents=64,
                 num_layers=2,
                 grad_checkpoint=False,
                 ):
        super().__init__()
        self.num_latents=64
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        # self.token_selector =
        # self.transformer_layers = nn.ModuleList(
        #     [TransformerDecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        # )
        # for i in range(num_layers):
        #     layers.append(LGTS(num_latents))
        #     layers.append(
        #         TransformerDecoderLayer(
        #             d_model=dim,nhead=12,dim_feedforward=2048,dropout=0.1,activation="gelu"
        #         )
        #     )
        # self.layers = nn.Sequential([])

    def forward(self, audio_embed, text):
        pass
