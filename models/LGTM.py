# A.K.A Language Guided Token Merger

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer
from transformers import T5EncoderModel, T5Tokenizer
from models.Qformer import *

class GatedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GatedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
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
        #self.projection_head = nn.Linear(hidden_size, hidden_size, bias=False) # From SimSiam
        self.audio_query_tokens = nn.Parameter(torch.zeros(1, 64, self.hidden_size))
        self.audio_query_tokens.data.normal_(mean=0.0, std=config.initializer_range)
        self.dropout = nn.Dropout(p=0.2)
        self.gated_network = GatedNetwork(input_dim=self.hidden_size, hidden_dim=128, output_dim=2)


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
        audio_text_embeds = self.dropout(audio_text_embeds) # Regularize text
        attention_scores = output.cross_attentions
        attention_score = attention_scores[-1] # Get last layer attention
        attention_score = attention_score.mean(dim=1) # [B,S,T]
        logits_per_audio_feat = attention_score.max(-1)[0] # [B,S]
        token_index = torch.topk(logits_per_audio_feat, self.num_latents, dim=1)[1]
        sorted_index = torch.sort(token_index, dim=1)[0]
        #fused_embed = torch.concat([audio_embeds,audio_text_embeds],dim=1)
        fused_embed = audio_embeds + audio_text_embeds
        B, S, H = fused_embed.size()
        mask = torch.ones(B, S, dtype=torch.bool, device=audio_embeds.device) # 
        mask[torch.arange(B).unsqueeze(1), sorted_index] = False # for K,V
        audio_embed_query = fused_embed[~mask].view(B,-1,H) # top_k token
        audio_embed_key_value = fused_embed[mask].view(B,-1,H) # other tokens
        attn_mask = torch.ones(audio_embed_key_value.size()[:-1], dtype=torch.long).to(audio_embeds.device)
        # Self token merger, mergy surrounding tokens in query tokens
        output = self.token_merger.bert(
            query_embeds=self.audio_query_tokens, # [B,64,H]
            encoder_hidden_states=audio_embed_key_value, # [B,S-64,H]
            encoder_attention_mask=attn_mask, # [B,S-64,H]
            return_dict=True,
        )
        B, _, H = fused_embed.size()
        gate_input = torch.stack([output.last_hidden_state, audio_embed_query], dim=1)
        gate_input = gate_input.view(-1, H)  # Reshape for the gated network
        gating_weights = self.gated_network(gate_input).view(B, 2, -1)
        output.last_hidden_state = gating_weights[:, 0, :, None] * output.last_hidden_state + \
                       gating_weights[:, 1, :, None] * audio_embed_query
        return output, None
        # ATC : Audio-Text Contrastive Alignment, add loss as auxilarity loss, no contrastive, SimSiam
        # labels = torch.arange(audio_embeds.shape[0], device=audio_embeds.device, dtype=torch.long)
        # #pooled_text_embeds = torch.mean(text_embeds, dim=1)  # B, H
        # pooled_audio_text_embeds = torch.mean(audio_text_embeds, dim=1)  # B, H
        # #projected_audio_embeds = self.projection_head(output.last_hidden_state)
        # #pooled_audio_embeds = torch.mean(projected_audio_embeds, dim=1) # [B.H]
        # pooled_audio_embeds = torch.mean(output.last_hidden_state, dim=1) # [B.H]
        # logits_per_audio = pooled_audio_embeds @ pooled_audio_text_embeds.T
        # logits_per_text = pooled_audio_text_embeds @ pooled_audio_embeds.T 
        # #cos_sim = F.cosine_similarity(pooled_audio_embeds[None, :], pooled_text_embeds[:, None], dim=-1)
        # #pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # #cos_sim = cos_sim / self.temperature
        # #nll = -cos_sim[pos_mask]
        # #loss = nll.mean()
        # #x = F.normalize(audio_embeds, p=2, dim=-1) # B, S, H
        # #y = F.normalize(audio_text_embeds, p=2, dim=-1) # B, H, S
        # #loss = 2 - 2 * (x * y).sum(dim=-1).mean()
        # total_loss = (
        #     F.cross_entropy(logits_per_audio, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        # ) / 2
        #return output, total_loss
