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
                 num_latents=32,
                 num_layers=2,
                 ):
        super().__init__()
        self.num_latents=num_latents
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
        self.text_token_merger = BertLMHeadModel(config=config)   # cross-attention with audio token
        self.text_token_merger.cls = None
        self.text_token_merger.bert.embeddings.word_embeddings = None
        self.text_token_merger.bert.embeddings.position_embeddings = None
        for layer in self.text_token_merger.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.audio_token_merger = BertLMHeadModel(config=config)   # cross-attention with latent query audio
        self.audio_token_merger.cls = None
        self.audio_token_merger.bert.embeddings.word_embeddings = None
        self.audio_token_merger.bert.embeddings.position_embeddings = None
        for layer in self.audio_token_merger.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.token_merger = BertLMHeadModel(config=config)   # select token of original modality and map again
        self.token_merger.cls = None
        self.token_merger.bert.embeddings.word_embeddings = None
        self.token_merger.bert.embeddings.position_embeddings = None
        for layer in self.audio_token_merger.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        config.add_cross_attention = False
        self.token_selection = BertLMHeadModel(config=config) # cross-attention with text encoder
        self.token_selection.cls = None
        self.token_selection.bert.embeddings.word_embeddings = None
        self.token_selection.bert.embeddings.position_embeddings = None
        for layer in self.token_selection.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.audio_query_tokens = nn.Parameter(torch.zeros(1, self.num_latents, self.hidden_size))
        self.audio_query_tokens.data.normal_(mean=0.0, std=config.initializer_range)
        self.text_query_tokens = nn.Parameter(torch.zeros(1, self.num_latents, self.hidden_size))
        self.text_query_tokens.data.normal_(mean=0.0, std=config.initializer_range)


    def forward(self, audio_embeds, caption):
        # Embed text using T5 encoder
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt")
        input_ids = text["input_ids"].to(audio_embeds.device)
        attention_mask = text["attention_mask"].to(audio_embeds.device)  # [B, 768], fine-grained [B,32,768]
        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=True
        )
        # Extract text information to learnable latent query
        text_embeds = text_encoder_output.last_hidden_state # [B,S,H]
        output = self.text_token_merger.bert(
            query_embeds=self.text_query_tokens,
            encoder_hidden_states=text_embeds, 
            encoder_attention_mask=attention_mask, 
            return_dict=True,
        )
        latent_text_embeds = output.last_hidden_state  # B. 32, H
        
        # Extract audio information to learnable latent query
        attn_mask = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)
        output = self.audio_token_merger.bert(
            query_embeds=self.audio_query_tokens,
            encoder_hidden_states=audio_embeds, 
            encoder_attention_mask=attn_mask, 
            output_attentions=True,
            return_dict=True,
        )
        latent_audio_embeds = output.last_hidden_state  # B. 32, H
        latent_attention_score = output.cross_attentions[-1] # [B,nH,32,S], S=128
        latent_attention_score = latent_attention_score.mean(dim=1) # [B,32,S]
        latent_embeds = torch.concat([latent_audio_embeds, latent_text_embeds], dim=1) # B, 64, H
        output = self.token_selection.bert(
            query_embeds=latent_embeds,  # [B, 64, H]
            output_attentions=True,
            return_dict=True,
        )
        fused_embed = output.last_hidden_state # [B, 64, H]
        attention_score = torch.concat(output.attentions,dim=1) # [[B,nH,S,S], [B,nH,S,S]]
        audio_attention_score = attention_score[:, :, :self.num_latents, self.num_latents:]  # [B, nH, 32, 32]
        audio_attention_score = audio_attention_score.mean(dim=1) # [B,32,32]
        audio_token_importance = torch.matmul(audio_attention_score, latent_attention_score) # [B,32,S]
        audio_token_importance = audio_token_importance.mean(dim=1) # [B,S}]
        token_index = torch.topk(audio_token_importance, 64, dim=1)[1]
        sorted_index = torch.sort(token_index, dim=1)[0]
        B, S, H = audio_embeds.shape
        mask = torch.ones(B, S, dtype=torch.bool, device=audio_embeds.device) # 
        mask[torch.arange(B).unsqueeze(1), sorted_index] = False # for K,V
        audio_embed_query = audio_embeds[~mask].view(B,-1,H) # top_k token
        audio_embed_key_value = audio_embeds[mask].view(B,-1,H) # other token
        audio_embed_key_value = torch.concat([audio_embed_key_value, fused_embed],dim=1) # add latent abstractor
        output = self.token_merger.bert(
            query_embeds=audio_embed_query,  # [B, 64, H]
            encoder_hidden_states=audio_embed_key_value, 
            encoder_attention_mask=attn_mask, 
            output_attentions=True,
            return_dict=True,
        )
        return output, None