# A.K.A Language Guided Token Merger

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from transformers import T5EncoderModel
from models.Qformer import *

class LinearTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, gate_noise=1.0, **options):
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=True)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.gate_noise = gate_noise

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        logits = self.wg(x) # Suppose BF16 is safe up to stable network
        if self.training and self.gate_noise > 0:
            logits_w_noise = logits + self.gate_noise * torch.randn_like(logits) / self.top_k
            scores = F.softmax(logits_w_noise,dim=1)
        else:
            logits_w_noise = logits
            scores = F.softmax(logits_w_noise,dim=1)
        return scores




# Language Guided Token Selector
# Do we need Gumbel Relaxation here? or Gated Networks on MoE?
# Or make Switch Transformer Network?, EViT
# CLS attention score(In-modality) + Language guided selection(cross-modality)
class LGTS(nn.Module):
    def __init__(self,num_latents=64, hidden_size=768):
        super().__init__()
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        self.softmax_gate = LinearTopKGate(model_dim=hidden_size, num_global_experts=2, k=2,gate_noise=0.1)

    def forward(self, audio_embed, text_embed):
        # Pick Important token with relevant text token
        logits = torch.einsum("bac,btc->bat",audio_embed, text_embed) # [B,A,T]
        logits_per_audio = logits.max(-1)[0] # [B, A]
        # Soft Merging using gated softmax, learnable score
        logits_per_audio = self.softmax_gate(audio_embed) * logits_per_audio
        topk_proposals_idx = torch.topk(logits_per_audio,self.num_latents, dim=1)[1]
        return topk_proposals_idx

class LGTM(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_latents=64,
                 num_layers=2,
                 grad_checkpoint=False,
                 ):
        super().__init__()
        self.num_latents=64
        self.hidden_size = hidden_size
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size))
        self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        self.token_selector = LGTS(num_latents=64, hidden_size=hidden_size)
        # TODO : Add BERTLayer from Qformer to cross attention, one layer for merge, one layer for T5
        # TODO : Add Learnable Query

    def forward(self, audio_embed, text):
        # Embed text using T5 encoder
        # TODO : Test T5 Encoder

        # Token selection

        #
        pass
