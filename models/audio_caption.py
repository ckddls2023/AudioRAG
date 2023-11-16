# Mainly Refer to pengi and SALMONN, MSCLAP
from dataclasses import dataclass
import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, GPT2Config, LlamaConfig
from optimum.bettertransformer import BetterTransformer
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig, HTSATAudioTower, HTSATEncoderConfig
from models.flamingo_pytorch import PerceiverResampler
from models.Qformer import *
from models.LGTM import *


@dataclass
class FrozenArgs:
    freeze_lm: bool = True
    freeze_am: bool = True


class CLAP2GPT2(nn.Module):

    def __init__(self):
        super(CLAP2GPT2, self).__init__()
        self.encoder_config = CLAPEncoderConfig()
        self.prefix_length = 1  # Only use embedding before projection, if fine-grained, re-calculate
        self.encoder = CLAPAudioTower(self.encoder_config)
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.tokenizer.model_max_length = 256
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id
        self.decoder_config = self.decoder.config

        # Freeze all CLAP parameters
        if self.encoder_config.freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Freeze all GPT2 parameters
        # if self.decoder_config.freeze:
        #     for p in self.gpt.parameters():
        #         p.requires_grad = False

        # Set 2Layer-MLP following LLAVA paper in NUIPS 2023
        modules = [
            nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder_config.hidden_size, self.decoder_config.hidden_size)
        ]
        self.enc_to_dec_proj = nn.Sequential(*modules)

    def forward_encoder(self, audios):
        outputs = self.encoder(audios).last_hidden_state
        outputs = outputs / outputs.norm(2, -1).reshape(-1, 1)  # Normalize embedding
        outputs = self.enc_to_dec_proj(outputs)
        return outputs

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), -100, dtype=torch.int64, device=device)

    def get_dummy_attn_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), 1, dtype=torch.int64, device=device)

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text, padding='longest', truncation=True, max_length=256, return_tensors="pt")
        input_ids = text["input_ids"].to(encoder_outputs.device)
        attention_mask = text["attention_mask"].to(encoder_outputs.device)
        embedding_text = self.decoder.transformer.wte(input_ids)
        embedding_cat = torch.cat((encoder_outputs.unsqueeze(1), embedding_text), dim=1)
        dummy_token = self.get_dummy_token(input_ids.shape[0], input_ids.device)
        labels = torch.cat((dummy_token, input_ids), dim=1)
        dummy_mask = self.get_dummy_attn_mask(input_ids.shape[0], input_ids.device)
        attention_mask = torch.cat((dummy_mask, attention_mask), dim=1)
        output = self.decoder(inputs_embeds=embedding_cat, labels=labels, attention_mask=attention_mask)
        return output

    def forward(self, audio, text):
        audio_embeds = self.forward_encoder(audio)
        output = self.forward_decoder(text, audio_embeds)
        return output

    # TODO : Re-design generate methods


class CLAP2LLAMA(nn.Module):

    def __init__(self, args=FrozenArgs()):
        super(CLAP2LLAMA, self).__init__()
        self.encoder_config = CLAPEncoderConfig(select_feature="fine_grained_embedding")
        self.encoder = CLAPAudioTower(self.encoder_config)
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")  # v1.5 : LLAMA2 + VICUNA
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Bug fix, CLAP needs lower version, but have model.vocab !=tokenizer.vocab
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.tokenizer.padding_side = "right"
        self.decoder_config = self.decoder.config
        self.args = args
        self.prefix_length = 1  # Only use embedding before projection, if fine-grained, re-calculate

        # Set 2Layer-MLP following LLAVA paper in NEUIPS 2023
        # modules = [
        #     nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(self.decoder_config.hidden_size, self.decoder_config.hidden_size)
        # ]
        # self.enc_to_dec_proj = nn.Sequential(*modules)
        # if self.encoder_config.select_feature == "fine_grained_embedding":
        #     self.prefix_length = int((self.encoder_config.sequence_length - self.encoder_config.window_size) / self.encoder_config.step_size + 1)

        # Set Perceiver Resampler following FLAMINGO paper 2021
        # modules = [
        #     PerceiverResampler(
        #         dim=self.encoder_config.hidden_size,
        #         depth=2,
        #         dim_head=64,
        #         heads=8,
        #         num_latents=64,  # the number of latents to shrink your media sequence to, perceiver style
        #         num_media_embeds=1,  # say you have 4 images maximum in your dialogue
        #         grad_checkpoint=True
        #     ),
        #     nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size)
        # ]
        # self.enc_to_dec_proj = nn.Sequential(*modules)
        # if isinstance(self.enc_to_dec_proj[0], PerceiverResampler):
        #     self.prefix_length = self.enc_to_dec_proj[0].latents.shape[0]  # Reduce tokens up to num_latents


        # Set Qformer following BLIP-2, Video-LLAMA
        # enc_to_dec_proj_config = BertConfig.from_pretrained("bert-base-uncased")
        # enc_to_dec_proj_config.num_hidden_layers = 2
        # enc_to_dec_proj_config.encoder_width = self.encoder_config.hidden_size
        # # insert cross-attention layer every other block
        # enc_to_dec_proj_config.add_cross_attention = True
        # enc_to_dec_proj_config.cross_attention_freq = 1
        # enc_to_dec_proj_config.query_length = 64 # number of latents
        # self.enc_to_dec_proj = BetterTransformer.transform(BertLMHeadModel(config=enc_to_dec_proj_config))
        # self.audio_query_tokens = nn.Parameter(
        #     torch.zeros(1, 64, enc_to_dec_proj_config.hidden_size)
        # )
        # self.audio_query_tokens.data.normal_(mean=0.0, std=enc_to_dec_proj_config.initializer_range)
        # self.enc_to_dec_proj.cls = None
        # self.enc_to_dec_proj.bert.embeddings.word_embeddings = None
        # self.enc_to_dec_proj.bert.embeddings.position_embeddings = None
        # for layer in self.enc_to_dec_proj.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.decoder_proj = nn.Linear(
        #     self.encoder_config.hidden_size, self.decoder_config.hidden_size
        # )
        # self.audio_position_embedding = nn.Embedding(256, self.encoder_config.hidden_size)
        # self.prefix_length = 64

        # Ours, token merge with langauge guided selection
        self.enc_to_dec_proj = LGTM(hidden_size=self.encoder_config.hidden_size, num_latents=64)
        self.decoder_proj = nn.Linear(
            self.encoder_config.hidden_size, self.decoder_config.hidden_size
        )
        self.prefix_length = 64


        # Freeze all CLAP parameters
        if args.freeze_am:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Freeze all LM parameters
        if args.freeze_lm:
            self.decoder.eval()
            print("Freezing the LLAMA.")
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            # target_modules = ["k_proj", "q_proj", "up_proj", "down_proj", "o_proj", "gate_proj", "v_proj"] # need ZERO 3..
            peft_config = LoraConfig(  # Following QLoRA, Recent Model use all target to tune LORA
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["k_proj", "q_proj", "o_proj", "v_proj"]
            )
            # config = IA3Config(
            #     peft_type="IA3",
            #     task_type="CAUSAL_LM",
            #     target_modules=['q_proj', 'k_proj', 'down_proj'],
            #     feedforward_modules=["down_proj"],
            # )
            self.decoder = get_peft_model(self.decoder, peft_config)
        # If prefix length > 64 and GPU memory exceed than we thought
        self.decoder.gradient_checkpointing_enable()

    def forward_encoder(self, audios, text=None):
        outputs = self.encoder(audios).last_hidden_state
        # Perceiver
        # if isinstance(self.enc_to_dec_proj[0], PerceiverResampler):
        #     outputs = outputs.unsqueeze(1) # [B,1,S,H]
        # outputs = self.enc_to_dec_proj(outputs)

        # Qformer
        # B, S = outputs.size()[:2]
        # position_ids = torch.arange(S, dtype=torch.long, device=outputs.device)
        # position_ids = position_ids.unsqueeze(0).expand(B, -1)
        # audio_position_embeddings = self.audio_position_embedding(position_ids)
        # outputs = outputs + audio_position_embeddings
        # audio_query_tokens = self.audio_query_tokens.expand(outputs.shape[0], -1, -1)
        # frame_atts = torch.ones(outputs.size()[:-1], dtype=torch.long).to(outputs.device)
        # audio_query_output = self.enc_to_dec_proj.bert(
        #     query_embeds=audio_query_tokens,  # [32,768]
        #     encoder_hidden_states=outputs,
        #     encoder_attention_mask=frame_atts,
        #     return_dict=True,
        # )
        # outputs = self.decoder_proj(audio_query_output.last_hidden_state)

        # Ours, Token merge with language guided selection
        audio_query_output = self.enc_to_dec_proj(outputs, text)
        outputs = self.decoder_proj(audio_query_output.last_hidden_state)

        # Perceiver
        # if isinstance(self.enc_to_dec_proj[0], PerceiverResampler):
        #     outputs = outputs.squeeze(1) # [B,S,H]
        return outputs


    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), -100, dtype=torch.int64, device=device)

    def get_dummy_attn_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), 1, dtype=torch.int64, device=device)

    def get_decoder_embeddings(self):
        if self.args.freeze_lm:
            return self.decoder.get_input_embeddings()
        else:
            return self.decoder.base_model.get_input_embeddings()

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
        input_ids = text["input_ids"].to(encoder_outputs.device)
        attention_mask = text["attention_mask"].to(encoder_outputs.device)  # [B, 768], fine-grained [B,32,768]
        embedding_text = self.get_decoder_embeddings()(input_ids)  # PEFT : model.base_model
        embedding_cat = torch.cat((encoder_outputs, embedding_text), dim=1)
        batch_size, seq_length = input_ids.shape
        shifted_input_ids = input_ids.new_zeros((batch_size, seq_length + self.prefix_length))
        shifted_input_ids[:, self.prefix_length:] = input_ids.clone()
        shifted_input_ids[:, :self.prefix_length] = -100
        shifted_input_ids.masked_fill(shifted_input_ids == self.tokenizer.pad_token_id, -100)
        dummy_mask = self.get_dummy_attn_mask(input_ids.shape[0], input_ids.device)
        attention_mask = torch.cat((dummy_mask, attention_mask), dim=1)
        output = self.decoder(inputs_embeds=embedding_cat, labels=shifted_input_ids, attention_mask=attention_mask)
        return output

    def forward(self, audio, text):
        # audio_embeds = self.forward_encoder(audio)
        audio_embeds = self.forward_encoder(audio, text) # Only when needs text..
        if audio_embeds.dim() == 2:  # B, H -> B, 1, H
            audio_embeds = audio_embeds.unsqueeze(1)
        output = self.forward_decoder(text, audio_embeds)
        return output

    def generate_caption(self, samples, prompt=None, beam_size: int = 5, entry_length=67, temperature=1.):
        r"""Generate audio captions for each audio recording in a batch"""
        # We might can use decoder.generate instead of impl directly
        with torch.no_grad():
            encoder_outputs = self.forward_encoder(samples, prompt)
            if encoder_outputs.dim() == 2:
                encoder_outputs = encoder_outputs.unsqueeze(1)
            outputs = self.decoder.generate(
                inputs_embeds=encoder_outputs,
                num_beams=2,
                min_length=0,
                max_length=256,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


if __name__ == "__main__":
    # SR(48000) * 10 => is_longer, should be used CLAP-fused
    # sample rate should be 48000, if not match resampled when load
    audio_data, _ = librosa.load('./examples/Yb0RFKhbpFJA.flac', sr=48000)
    text_data = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    audio_data = torch.tensor(audio_data).to("cuda")

    audio_caption_model = CLAP2GPT2().to("cuda")
    output = audio_caption_model(audio_data, text_data)
    print(f"loss : {output['loss']}")
    print(f"logits : {output['logits'].shape}")  # logits : torch.Size([1, 15, 50257])

    audio_caption_model = CLAP2LLAMA().to("cuda")
    output = audio_caption_model(audio_data, text_data)
    print(f"loss : {output['loss']}")
    print(f"logits : {output['logits'].shape}")  # logits : torch.Size([1, 19, 32000])

    captions = audio_caption_model.generate_caption(audio_data)
    print(f"captions : {captions}")

    # captions = audio_caption_model.generate_beam(embed=None, prompt="What is dog? Explain short and breif way. ")
    # print(f"generation with prompt : {captions[0]}")
