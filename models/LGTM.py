# A.K.A Language Guided Token Merger

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from transformers import T5EncoderModel, T5Tokenizer
from models.Qformer import *
class LGTM(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_latents=64,
                 num_layers=2,
                 ):
        super().__init__()
        self.num_latents=64
        self.hidden_size = hidden_size
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.text_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base")
        self.text_encoder.eval()
        for param in self.text_encoder.parameters(): # Freezing T5
            param.requires_grad = False
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = num_layers
        config.encoder_width = self.hidden_size
        config.add_cross_attention = True
        config.cross_attention_freq = 1
        config.query_length = num_latents # number of latents
        self.audio2text_xattn = BertLMHeadModel(config=config) # cross-attention with text encoder
        self.audio2text_xattn.cls = None
        self.audio2text_xattn.bert.embeddings.word_embeddings = None
        self.audio2text_xattn.bert.embeddings.position_embeddings = None
        for layer in self.audio2text_xattn.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.token_merger = BertLMHeadModel(config=config)   # cross-attention with audio token
        self.token_merger.cls = None
        self.token_merger.bert.embeddings.word_embeddings = None
        self.token_merger.bert.embeddings.position_embeddings = None
        for layer in self.token_merger.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.temperature = 0.2
        # self.projection_head = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, audio_embeds, caption):
        # Embed text using T5 encoder
        B, S = audio_embeds.size()[:2]
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt")
        input_ids = text["input_ids"].to(audio_embeds.device)
        attention_mask = text["attention_mask"].to(audio_embeds.device)  # [B, 768], fine-grained [B,32,768]
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True
        )
        text_embeds = text_encoder_output.last_hidden_state # [B,S,H]
        # Cross Attend to T5, single layer
        output = self.audio2text_xattn.bert(
            query_embeds=audio_embeds,  # [B,S,H]
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        audio_text_embeds = output.last_hidden_state
        attention_scores = output.cross_attentions
        attention_score = attention_scores[-1] # Get last layer attention
        attention_score = attention_score.mean(dim=1) # [B,S,T]
        logits_per_audio_feat = attention_score.max(-1)[0] # [B,S]
        token_index = torch.topk(logits_per_audio_feat, self.num_latents, dim=1)[1]
        sorted_index = torch.sort(token_index, dim=1)[0]
        feat = output.last_hidden_state # Fused with text information
        fused_embed = audio_text_embeds + audio_embeds
        audio_embed_query = fused_embed[torch.arange(B).unsqueeze(1), token_index]
        mask = torch.ones(B, S, dtype=torch.bool, device=audio_embeds.device) # Inverted index... other fancy way..?
        mask[torch.arange(B).unsqueeze(1), sorted_index] = False # This way is very slow compares to flops... painful..
        audio_embed_key_value = fused_embed[mask.unsqueeze(-1).expand_as(audio_embeds)].reshape(B,S-self.num_latents,-1)
        attn_mask = torch.ones(audio_embed_key_value.size()[:-1], dtype=torch.long).to(audio_embeds.device)
        # Self token merger
        output = self.token_merger.bert(
            query_embeds=audio_embed_query, # [B,64,H]
            encoder_hidden_states=audio_embed_key_value, # [B,S-64,H]
            encoder_attention_mask=attn_mask, # [B,S-64,H]
            return_dict=True,
        )
        return output, None
        # ATC : Audio-Text Contrastive Alignment, add loss as auxilarity loss
        #pooled_audio_text_embeds = torch.mean(audio_text_embeds, dim=1) 
        #projected_audio_embeds = self.projection_head(output.last_hidden_state)
        #pooled_audio_embeds = torch.mean(projected_audio_embeds, dim=1) # [B.H]
        #cos_sim = F.cosine_similarity(pooled_audio_embeds[None, :], pooled_audio_text_embeds[:, None], dim=-1)
        #pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        #cos_sim = cos_sim / self.temperature
        #nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        #nll = nll.mean()
        #return output, nll
