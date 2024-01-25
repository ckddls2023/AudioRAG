# Mainly Refer to pengi and SALMONN, MSCLAP

import soundfile as sf
from dataclasses import dataclass
import torch
import random
import librosa
from omegaconf import OmegaConf
from itertools import chain
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, GPT2Config, LlamaConfig, GenerationConfig
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig, PromptEncoderConfig, AdaptionPromptConfig
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig
from models.flamingo_pytorch import PerceiverResampler
from models.Qformer import *
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    LlamaForCausalLM,
    LlamaTokenizer
)
import librosa
from beats.BEATs import BEATsConfig, BEATs
import torch.nn.functional as F



DEFAULT_AUD_TOKEN = "[AUD]"
DEFAULT_AUD_END_TOKEN = "[/AUD]"
DEFAULT_AUDIO_TOKEN = "<audio>"

class CLAP2LLAMA(nn.Module):

    def __init__(self, config):
        super(CLAP2LLAMA, self).__init__()
        self.config = config
        self.encoder_config = CLAPEncoderConfig.from_dict(OmegaConf.to_container(config.encoder, resolve=True))
        self.encoder = CLAPAudioTower(self.encoder_config)
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")  # v1.5 : LLAMA2 + VICUNA
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, add_eos_token=True,add_bos_token=False)
        self.generation_config, unused_kwargs = GenerationConfig.from_pretrained("lmsys/vicuna-7b-v1.5", max_new_tokens=200, return_unused_kwargs=True)
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        if num_new_tokens > 0:
            self.decoder.resize_token_embeddings(len(self.tokenizer))
            input_embeddings = self.decoder.get_input_embeddings().weight.data
            output_embeddings = self.decoder.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = 256
        self.retr_prompt = config.retr_prompt
        self.task_prompt = config.task_prompt
        self.caption_prompt = config.caption_prompt
        self.decoder_config = self.decoder.config
        
        if self.config.align.model_name == "MLP":
            modules = [
                nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.decoder_config.hidden_size, self.decoder_config.hidden_size),
                #nn.LayerNorm(self.decoder_config.hidden_size)
            ]
            self.enc_to_dec_proj = nn.Sequential(*modules)
            self.forward_align = self.forward_mlp

        if self.config.align.model_name == "Perceiver":
            modules = [
                PerceiverResampler(
                    dim=self.encoder_config.hidden_size,
                    depth=2,
                    dim_head=64,
                    heads=8,
                    num_latents=64,  # the number of latents to shrink your media sequence to, perceiver style
                    num_media_embeds=1,  # say you have 4 images maximum in your dialogue
                    grad_checkpoint=True
                ),
                nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size)
            ]
            self.enc_to_dec_proj = nn.Sequential(*modules)
            self.forward_align = self.forward_mlp

        if self.config.align.model_name == "Qformer":
            enc_to_dec_proj_config = BertConfig.from_pretrained("bert-base-uncased")
            enc_to_dec_proj_config.num_hidden_layers = 2
            enc_to_dec_proj_config.encoder_width = self.encoder_config.hidden_size
            # insert cross-attention layer every other block
            enc_to_dec_proj_config.add_cross_attention = True
            enc_to_dec_proj_config.cross_attention_freq = 1
            enc_to_dec_proj_config.query_length = 64 # number of latents
            self.audio_query_tokens = nn.Parameter(torch.zeros(1, 64, enc_to_dec_proj_config.hidden_size))
            self.audio_query_tokens.data.normal_(mean=0.0, std=enc_to_dec_proj_config.initializer_range)
            self.enc_to_dec_proj = BertLMHeadModel(config=enc_to_dec_proj_config)   # cross-attention with audio token
            self.enc_to_dec_proj.cls = None
            self.enc_to_dec_proj.bert.embeddings.word_embeddings = None
            self.enc_to_dec_proj.bert.embeddings.position_embeddings = None
            for layer in self.enc_to_dec_proj.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.decoder_proj = nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size)
            self.audio_position_embedding = nn.Embedding(256, self.encoder_config.hidden_size)
            self.forward_align = self.forward_qformer


        self.freeze_am = config.freeze_am
        self.unfreeze_am = config.unfreeze_am
        self.freeze_lm = config.freeze_lm
        self.freeze_align = config.freeze_align

        # Freeze all CLAP parameters
        self.decoder.gradient_checkpointing_enable()
        if self.freeze_am:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False
                if any(prefix in name for prefix in config.unfreeze_am):
                    p.requires_grad = True
        if self.unfreeze_am: # Print trainable parameters of LORA
            total_params = 0
            for param in self.encoder.parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    total_params += num_params
            print("Total trainable parameters:", total_params)
            
        # Freeze all alignment paramters
        if self.freeze_align:
            for param in self.enc_to_dec_proj.parameters():
                param.requires_grad = False
            if self.config.align.model_name == "LGTM" or self.config.align.model_name == "Qformer":
                for param in self.decoder_proj.parameters():
                    param.requires_grad = False

        # Freeze all LM parameters
        if self.freeze_lm:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            peft_type = config.peft_config.pop('type', None)
            if peft_type == "LORA":
                self.peft_config = LoraConfig(**config.peft_config)
            if peft_type == "IA3":
                self.peft_config = IA3Config(**config.peft_config)
            if peft_type == "PTUNING": 
                self.peft_config = PromptEncoderConfig(**config.peft_config)
            if peft_type == "LLAMA-ADAPTER":
                self.peft_config = AdaptionPromptConfig(**config.peft_config)
            self.decoder = get_peft_model(self.decoder, self.peft_config)
            if peft_type == "LORA":
                self.decoder.base_model.model.model.embed_tokens.weight.requires_grad = False
                self.decoder.base_model.model.lm_head.weight.requires_grad = False # slightly finetune lm_head
            if peft_type == "IA3":
                pass
            else:
                self.decoder.model.model.embed_tokens.weight.requires_grad = False
                self.decoder.model.lm_head.weight.requires_grad = False # slightly finetune lm_head
            self.decoder.print_trainable_parameters() # Should be below 1%... otherwise, embed_token or lm_head saved


    def forward_mlp(self, x):
        outputs = self.enc_to_dec_proj(x)
        return outputs

    def forward_qformer(self, outputs):
        B, S = outputs.size()[:2]
        position_ids = torch.arange(S, dtype=torch.long, device=outputs.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        audio_position_embeddings = self.audio_position_embedding(position_ids)
        outputs = outputs + audio_position_embeddings
        audio_query_tokens = self.audio_query_tokens.expand(outputs.shape[0], -1, -1)
        frame_atts = torch.ones(outputs.size()[:-1], dtype=torch.long).to(outputs.device)
        audio_query_output = self.enc_to_dec_proj.bert(
            query_embeds=audio_query_tokens,  # [32,768]
            encoder_hidden_states=outputs,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )
        outputs = self.decoder_proj(audio_query_output.last_hidden_state)
        return outputs

    def forward_encoder(self, audios):
        outputs = self.encoder(audios).last_hidden_state
        #outputs = outputs / outputs.norm(2, -1).unsqueeze(-1)  # Normalize embedding
        outputs = self.forward_align(outputs) # loss is None for MLP, Perceiver, Q-former
        return outputs

    def get_decoder_embeddings(self):
        if self.freeze_lm:
            return self.decoder.get_input_embeddings()
        else:
            return self.decoder.base_model.get_input_embeddings()

    def shift_and_pad_input(self, input_ids, attn_mask, prefix_length):
        batch_size, seq_length = input_ids.shape
        shifted_input_ids = input_ids.new_zeros((batch_size, seq_length + prefix_length))
        shifted_input_ids[:, prefix_length:] = input_ids.clone()
        shifted_input_ids[:, :prefix_length] = -100
        shifted_input_ids.masked_fill_(shifted_input_ids == self.tokenizer.pad_token_id, -100)
        shifted_attn_mask = attn_mask.new_ones((batch_size, seq_length + prefix_length))
        shifted_attn_mask[:, prefix_length:] = attn_mask.clone()
        return shifted_input_ids, shifted_attn_mask

    def prepare_text_input(self, caption, device, add_special_tokens=True):
        tokenized_text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_ids = tokenized_text["input_ids"].to(device)
        attn_mask = tokenized_text["attention_mask"].to(device)
        return input_ids, attn_mask
    
    def insert_prompt(self, prompt, input_embeds, shifted_input_ids, shifted_attn_mask):
        if prompt:
            prompts = [prompt] * input_embeds.shape[0]
            prompt_ids, prompt_mask = self.prepare_text_input(prompts, input_embeds.device, add_special_tokens=False)
            input_embeds = torch.cat((self.get_decoder_embeddings()(prompt_ids), input_embeds), dim=1)
            prompt_ids[:, :] = -100  # Ignore prompt
            if shifted_input_ids is not None:
                shifted_input_ids = torch.cat((prompt_ids, shifted_input_ids), dim=1)
            shifted_attn_mask = torch.cat((prompt_mask, shifted_attn_mask), dim=1)
        return input_embeds, shifted_input_ids, shifted_attn_mask
    
    def forward(self, audio, caption, retr_audios=None, retr_captions=None):
        # audio_lengths = [len(elem) if isinstance(elem, list) else 1 for elem in audio] # 10s segments
        # flattend_audio = list(chain.from_iterable(elem if isinstance(elem, list) else [elem] for elem in audio))
        # audio_embed = self.forward_encoder(flattend_audio)  # Only for LGTM # 
        # audio_embed = [audio_embed[sum(audio_lengths[:i]):sum(audio_lengths[:i + 1]),:,:] for i in range(len(audio_lengths))]
        # audio_embed = [embed.reshape(1, -1, embed.shape[-1]) for embed in audio_embed]
        # max_length = max(embed.shape[1] for embed in audio_embed)
        # padded_embed = [torch.nn.functional.pad(embed, (0, 0, 0, max_length - embed.shape[1]), 'constant', 0) for embed in audio_embed]
        # padded_embed = torch.cat(padded_embed, dim=0)
        audio_embed = self.forward_encoder(audio)  # Only for LGTM # 
        retr_audio_embeds = []
        if retr_audios:
            for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)):
                retr_embed = self.forward_encoder(retr_audio)
                retr_audio_embeds.append(retr_embed.detach())
        output = self.forward_decoder(audio_embed, caption, retr_audio_embeds, retr_captions)
        return output

    def forward_decoder(self, audio_embed, caption, retr_audio_embeds=None, retr_captions=None):
        input_ids, attn_mask = self.prepare_text_input(caption, audio_embed.device)
        # zero_padded_mask = torch.any(audio_embed != 0, dim=-1).to(audio_embed.device) # B,S
        input_embeds = self.get_decoder_embeddings()(input_ids)
        input_embeds, input_ids, attn_mask = self.insert_prompt(self.caption_prompt, input_embeds, input_ids, attn_mask)
        input_embeds = torch.cat((audio_embed, input_embeds), dim=1)
        shifted_input_ids, shifted_attn_mask = self.shift_and_pad_input(input_ids, attn_mask, audio_embed.shape[1])
        # shifted_attn_mask[:,:audio_embed.shape[1]] = zero_padded_mask  # WARN: Consider padding exists in audio_embed
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt(self.task_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)
        retr_nums = max(len(retr_audio_embeds), len(retr_captions))
        retr_nums = random.randint(0, retr_nums) # Randomly retreive
        for i in range(retr_nums): # Suppose we have only case ([],retr_captions), (retr_audio_embeds,[]), both pair with same length
            if retr_captions: 
                retr_input_ids, retr_attn_mask = self.prepare_text_input(retr_captions[i], audio_embed.device, add_special_tokens=False)
                input_embeds = torch.cat((self.get_decoder_embeddings()(retr_input_ids), input_embeds), dim=1) 
                retr_input_ids[:,:] = -100 # Ignore all Wavcaps style
                # retr_input_ids.masked_fill_(retr_input_ids == self.tokenizer.pad_token_id, -100)
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
            if retr_audio_embeds: 
                input_embeds = torch.cat((retr_audio_embeds[i], input_embeds), dim=1) 
                retr_input_ids = input_ids.new_zeros(input_ids.shape[0],retr_audio_embeds[i].shape[1]) # B, prefix_length
                retr_input_ids[:,:] = -100 # Ignore all embeds
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat([shifted_attn_mask.new_ones(retr_audio_embeds[i].size()[:2]), shifted_attn_mask], dim=1)
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)
        bos_token_embed = self.get_decoder_embeddings()(torch.tensor([self.tokenizer.bos_token_id]).to(input_embeds.device))
        bos_token_embed = bos_token_embed.repeat(input_embeds.shape[0], 1, 1)
        input_embeds = torch.cat((bos_token_embed, input_embeds), dim=1)
        shifted_attn_mask = torch.cat((shifted_attn_mask.new_ones((input_embeds.shape[0], 1)),  shifted_attn_mask), dim=1)
        bos_token_ids = torch.full((shifted_input_ids.size(0), 1), self.tokenizer.bos_token_id, dtype=torch.long, device=shifted_input_ids.device)
        shifted_input_ids = torch.cat((bos_token_ids, shifted_input_ids), dim=1)
        output = self.decoder(inputs_embeds=input_embeds, labels=shifted_input_ids, attention_mask=shifted_attn_mask, output_attentions=True)
        return output

    def save_ckpt(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(self.enc_to_dec_proj.state_dict(), os.path.join(checkpoint_path,"enc_to_dec_proj.bin"))
        if self.config.align.model_name == "Qformer":
            torch.save(self.decoder_proj.state_dict(), os.path.join(checkpoint_path,"decoder_proj.bin"))
            torch.save(self.audio_query_tokens, os.path.join(checkpoint_path,'audio_query_tokens.bin'))
        if self.unfreeze_am:
            torch.save(self.encoder.state_dict(), os.path.join(checkpoint_path,"audio_encoder.bin"))
        if not self.freeze_lm:
            self.decoder.save_pretrained(checkpoint_path)

    def load_ckpt(self, checkpoint_path):
        print("Load model from checkpoint")
        self.enc_to_dec_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,"enc_to_dec_proj.bin")), strict=True)
        if self.config.align.model_name == "Qformer":
            self.decoder_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,"decoder_proj.bin")), strict = True)
            self.audio_query_tokens.data.copy_(torch.load(os.path.join(checkpoint_path,'audio_query_tokens.bin')))
            self.audio_position_embedding.load_state_dict(torch.load(os.path.join(checkpoint_path,"audio_position_embedding.bin")), strict = True)
        audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
        if os.path.exists(audio_encoder_ckpt):
            self.encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=False)
        if not self.freeze_lm and 'finetuned' in checkpoint_path:
            peft_type = self.config.peft_config.pop('type', None)
            if peft_type == "LORA":
                self.decoder = PeftModel.from_pretrained(self.decoder.base_model, checkpoint_path, config=self.peft_config)  # suppose don't use get_peft_model
            if peft_type == "IA3":
                self.decoder = PeftModel.from_pretrained(self.decoder.model, checkpoint_path, config=self.peft_config)  # suppose don't use get_peft_model
            self.decoder.enable_adapter_layers()

    def generate_caption(self, audio, caption=None, retr_audios=None, retr_captions=None):
        r"""Generate audio captions for each audio recording in a batch"""
        with torch.no_grad():
            if isinstance(audio[0], list): 
                audio = audio[0]
            input_embeds = self.forward_encoder(audio) 
            if len(audio) > 1: # longer than 10s, we consider only batch_size=1, concatenated in Sequence 
                B, S, H = input_embeds.shape
                input_embeds = input_embeds.view(1, B*S, H)
            batch_size, seq_length, _ = input_embeds.shape
            shifted_attn_mask = input_embeds.new_ones((batch_size, seq_length)).long()
            if self.caption_prompt:
                input_ids, attn_mask = self.prepare_text_input(self.caption_prompt, input_embeds.device, add_special_tokens=False)
                input_embeds = torch.cat((input_embeds, self.get_decoder_embeddings()(input_ids)), dim=1)  
                shifted_attn_mask = torch.cat((shifted_attn_mask, attn_mask), dim=1)
            retr_audio_embeds = []
            for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)): # Only LGTM needs retr_text, others ignore
                retr_embed = self.forward_encoder(retr_audio)
                retr_audio_embeds.append(retr_embed)
            input_embeds, _, shifted_attn_mask = self.insert_prompt(self.task_prompt, input_embeds, None, shifted_attn_mask)
            retr_nums = max(len(retr_audio_embeds), len(retr_captions))
            for i in range(retr_nums): # Suppose we have only case ([],retr_captions), (retr_audio_embeds,[]), both pair with same length
                if retr_captions: 
                    retr_input_ids, retr_attn_mask = self.prepare_text_input(retr_captions[i], input_embeds.device, add_special_tokens=False)
                    input_embeds = torch.cat((self.get_decoder_embeddings()(retr_input_ids), input_embeds), dim=1) 
                    shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
                    #input_embeds, _, shifted_attn_mask = self.insert_prompt("\nCaption: ", input_embeds, None, shifted_attn_mask)
                if retr_audio_embeds: 
                    input_embeds = torch.cat((retr_audio_embeds[i], input_embeds), dim=1) 
                    retr_attn_mask = shifted_attn_mask.new_ones(retr_audio_embeds[i].size()[:2]) # B, prefix
                    shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
                    #input_embeds, _, shifted_attn_mask = self.insert_prompt("\nAudio: ", input_embeds, None, shifted_attn_mask)
            if retr_nums:
                input_embeds, _, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, None, shifted_attn_mask)
            else: # We randomly don't input retr pairs
                input_embeds, _, shifted_attn_mask = self.insert_prompt("", input_embeds, None, shifted_attn_mask)
            bos_token_embed = self.get_decoder_embeddings()(torch.tensor([self.tokenizer.bos_token_id]).to(input_embeds.device))
            bos_token_embed = bos_token_embed.repeat(input_embeds.shape[0], 1, 1)
            input_embeds = torch.cat((bos_token_embed, input_embeds), dim=1)
            shifted_attn_mask = torch.cat((shifted_attn_mask.new_ones((batch_size, 1)),  shifted_attn_mask), dim=1)
            outputs = self.decoder.generate(
                inputs_embeds=input_embeds,
                attention_mask=shifted_attn_mask,
                num_beams=4,
                max_new_tokens=256,
                min_length=0,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
                #generation_config=self.generation_config,
            )
            captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
    
    
    
class SALMONN(nn.Module):
    def __init__(
        self,
        whisper_path,
        beats_path,
        device,
        second_per_frame=0.333333,
        second_stride=0.333333,
    ):
        super().__init__()
        self.device = device
        
        # Encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats_cfg = BEATsConfig(beats_checkpoint['cfg'])
        beats = BEATs(beats_cfg)
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        self.dtype = torch.bfloat16
        
        for name, p in self.beats.named_parameters():
            p.requires_grad = False
        for name, p in self.speech_encoder.named_parameters():
            p.requires_grad = False
        
        # decoder
        # self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")  # v1.5 : LLAMA2 + VICUNA
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")  # v1.5 : LLAMA2 + VICUNA
        self.llama_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, add_eos_token=True,add_bos_token=False)
        num_new_tokens = self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.decoder.config.pad_token_id = self.llama_tokenizer.pad_token_id
        if num_new_tokens > 0:
            self.decoder.resize_token_embeddings(len(self.llama_tokenizer))
            input_embeddings = self.decoder.get_input_embeddings().weight.data
            output_embeddings = self.decoder.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        self.llama_tokenizer.model_max_length = 4096

        modules = [
            nn.Linear(2048, self.decoder.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.decoder.config.hidden_size, self.decoder.config.hidden_size),
        ]
        self.decoder_proj = nn.Sequential(*modules)
        self.freeze_lm = True
        if self.freeze_lm:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
        else:
            self.peft_config = LoraConfig(
                r=16, lora_alpha=16, lora_dropout=0.1, bias="none", task_type= "CAUSAL_LM",
                modules_to_save=[],target_modules=["q_proj", "v_proj", "k_proj","o_proj"]
            )
            #   target_modules: ["q_proj", "v_proj", "k_proj","o_proj","lm_head","embed_tokens"]
            self.decoder = get_peft_model(self.decoder, self.peft_config)
            self.decoder.base_model.model.model.embed_tokens.weight.requires_grad = False
            self.decoder.base_model.model.lm_head.weight.requires_grad = False # slightly finetune lm_head
            self.decoder.print_trainable_parameters() # Should be below 1%... otherwise, embed_token or lm_head saved
        
    def save_ckpt(self, checkpoint_path, use_fsdp=False):
        torch.save(self.decoder_proj.state_dict(), os.path.join(checkpoint_path,"decoder_proj.bin"))
        torch.save(self.ln_speech.state_dict(), os.path.join(checkpoint_path,"ln_speech.bin"))
        torch.save(self.ln_audio.state_dict(), os.path.join(checkpoint_path,"ln_audio.bin"))
        if not self.freeze_lm:
            self.decoder.save_pretrained(checkpoint_path)
            
    def load_ckpt(self, checkpoint_path, use_fsdp=False):
        self.decoder_proj.load_state_dict(torch.load(os.path.join(checkpoint_path,"decoder_proj.bin")), strict = True)
        self.ln_speech.load_state_dict(torch.load(os.path.join(checkpoint_path,"ln_speech.bin")), strict = True)
        self.ln_audio.load_state_dict(torch.load(os.path.join(checkpoint_path,"ln_audio.bin")), strict = True)
        if not self.freeze_lm and 'finetuned' in checkpoint_path:
            self.decoder = PeftModel.from_pretrained(self.decoder.base_model, checkpoint_path, config=self.peft_config)  # suppose don't use get_peft_model
            self.decoder.enable_adapter_layers()
        
    def prepare_text_input(self, caption, device, add_special_tokens=True):
        tokenized_text = self.llama_tokenizer(caption, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=add_special_tokens)
        input_ids = tokenized_text["input_ids"].to(device)
        attn_mask = tokenized_text["attention_mask"].to(device)
        return input_ids, attn_mask

    def get_decoder_embeddings(self):
        if self.freeze_lm:
            return self.decoder.get_input_embeddings()
        else:
            return self.decoder.base_model.get_input_embeddings()
        
    def encode_audio(self, wav):
        with torch.no_grad():
            device = self.device
            spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(device) # [1, 80, 3000]
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
        
            # beats
            if isinstance(wav, list):
                wav = np.vstack(wav)
                raw_wav = torch.from_numpy(wav).to(speech_embeds.device)
            else:
                raw_wav = torch.from_numpy(wav).to(speech_embeds.device).unsqueeze(0)
            audio_padding_mask = torch.zeros(raw_wav.shape, device=speech_embeds.device).bool()
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

            # auditory embeds
            with torch.autocast(device_type="cuda"):
                speech_embeds = self.ln_speech(speech_embeds.float())
                audio_embeds = self.ln_audio(audio_embeds.float())
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
            speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)
            
            # fold embeds # [B, 1500, 2048]
            unfolded = speech_embeds.unfold(1, 15, 15) # [B,1024/S,768,W]
            averaged_embeds = unfolded.mean(dim=-1) # [B,1024/S,768]
        return averaged_embeds
    
    def insert_prompt(self, prompt, input_embeds, shifted_input_ids, shifted_attn_mask):
        if prompt:
            prompts = [prompt] * input_embeds.shape[0]
            prompt_ids, prompt_mask = self.prepare_text_input(prompts, input_embeds.device, add_special_tokens=False)
            input_embeds = torch.cat((self.get_decoder_embeddings()(prompt_ids), input_embeds), dim=1)
            prompt_ids[:, :] = -100  # Ignore prompt
            if shifted_input_ids is not None:
                shifted_input_ids = torch.cat((prompt_ids, shifted_input_ids), dim=1)
            shifted_attn_mask = torch.cat((prompt_mask, shifted_attn_mask), dim=1)
        return input_embeds, shifted_input_ids, shifted_attn_mask
    
    def shift_and_pad_input(self, input_ids, attn_mask, prefix_length):
        batch_size, seq_length = input_ids.shape
        shifted_input_ids = input_ids.new_zeros((batch_size, seq_length + prefix_length))
        shifted_input_ids[:, prefix_length:] = input_ids.clone()
        shifted_input_ids[:, :prefix_length] = -100
        shifted_input_ids.masked_fill_(shifted_input_ids == self.llama_tokenizer.pad_token_id, -100)
        shifted_attn_mask = attn_mask.new_ones((batch_size, seq_length + prefix_length))
        shifted_attn_mask[:, prefix_length:] = attn_mask.clone()
        return shifted_input_ids, shifted_attn_mask
    
    def forward_decoder(self, audio_embed, caption, retr_audio_embeds=None, retr_captions=None):
        input_ids, attn_mask = self.prepare_text_input(caption, audio_embed.device)
        input_embeds = self.get_decoder_embeddings()(input_ids)
        # | instruction | audio | prompt <- here | caption | 
        # input_embeds, input_ids, attn_mask = self.insert_prompt(self.caption_prompt, input_embeds, input_ids, attn_mask)
        input_embeds = torch.cat((audio_embed, input_embeds), dim=1)
        shifted_input_ids, shifted_attn_mask = self.shift_and_pad_input(input_ids, attn_mask, audio_embed.shape[1])
        # | instruction <- here| audio | prompt <- here | caption | 
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt("Describe this audio sound ", input_embeds, shifted_input_ids, shifted_attn_mask)
        retr_nums = max(len(retr_audio_embeds), len(retr_captions))
        retr_nums = random.randint(0, retr_nums) # Randomly retreive
        for i in range(retr_nums): # Suppose we have only case ([],retr_captions), (retr_audio_embeds,[]), both pair with same length
            if retr_captions: 
                retr_input_ids, retr_attn_mask = self.prepare_text_input(retr_captions[i], audio_embed.device, add_special_tokens=False)
                input_embeds = torch.cat((self.get_decoder_embeddings()(retr_input_ids), input_embeds), dim=1) 
                retr_input_ids[:,:] = -100 # Ignore all Wavcaps style
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
            if retr_audio_embeds: 
                input_embeds = torch.cat((retr_audio_embeds[i], input_embeds), dim=1) 
                retr_input_ids = input_ids.new_zeros(input_ids.shape[0],retr_audio_embeds[i].shape[1]) # B, prefix_length
                retr_input_ids[:,:] = -100 # Ignore all embeds
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat([shifted_attn_mask.new_ones(retr_audio_embeds[i].size()[:2]), shifted_attn_mask], dim=1)
        # input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)
        bos_token_embed = self.get_decoder_embeddings()(torch.tensor([self.llama_tokenizer.bos_token_id]).to(input_embeds.device))
        bos_token_embed = bos_token_embed.repeat(input_embeds.shape[0], 1, 1)
        input_embeds = torch.cat((bos_token_embed, input_embeds), dim=1)
        shifted_attn_mask = torch.cat((shifted_attn_mask.new_ones((input_embeds.shape[0], 1)),  shifted_attn_mask), dim=1)
        bos_token_ids = torch.full((shifted_input_ids.size(0), 1), self.llama_tokenizer.bos_token_id, dtype=torch.long, device=shifted_input_ids.device)
        shifted_input_ids = torch.cat((bos_token_ids, shifted_input_ids), dim=1)
        output = self.decoder(inputs_embeds=input_embeds, labels=shifted_input_ids, attention_mask=shifted_attn_mask, output_attentions=True)
        return output
    
    def forward(self, audio, caption, retr_audios=None, retr_captions=None):
        audio_embed = self.encode_audio(audio)  # Only for LGTM # 
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            audio_embed_proj = self.decoder_proj(audio_embed)
            retr_audio_embeds = []
            if retr_audios:
                for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)):
                    retr_embed = self.encode_audio(retr_audio)
                    retr_embed_proj = self.decoder_proj(retr_embed)
                    retr_audio_embeds.append(retr_embed_proj.detach())
            output = self.forward_decoder(audio_embed_proj, caption, retr_audio_embeds, retr_captions)
        return output
        
    def generate_caption(self, audio, caption=None, retr_audios=None, retr_captions=None):
        r"""Generate audio captions for each audio recording in a batch"""
        with torch.no_grad():
            audio_embed = self.encode_audio(audio) 
            input_embeds = self.decoder_proj(audio_embed) # [B,100,H]
            batch_size, seq_length, _ = input_embeds.shape
            shifted_attn_mask = input_embeds.new_ones((batch_size, seq_length)).long()
            retr_audio_embeds = []
            for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)): # Only LGTM needs retr_text, others ignore
                retr_audio_embed = self.encode_audio(retr_audio) 
                retr_audio_embed_proj = self.decoder_proj(retr_audio_embed)
                retr_audio_embeds.append(retr_audio_embed_proj)
            input_embeds, _ , shifted_attn_mask = self.insert_prompt("Describe this audio sound ", input_embeds, None, shifted_attn_mask)
            retr_nums = max(len(retr_audio_embeds), len(retr_captions))
            for i in range(retr_nums): # Suppose we have only case ([],retr_captions), (retr_audio_embeds,[]), both pair with same length
                if retr_captions: 
                    retr_input_ids, retr_attn_mask = self.prepare_text_input(retr_captions[i], input_embeds.device, add_special_tokens=False)
                    input_embeds = torch.cat((self.get_decoder_embeddings()(retr_input_ids), input_embeds), dim=1) 
                    shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
                if retr_audio_embeds: 
                    input_embeds = torch.cat((retr_audio_embeds[i], input_embeds), dim=1) 
                    retr_attn_mask = shifted_attn_mask.new_ones(retr_audio_embeds[i].size()[:2]) # B, prefix
                    shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
            # input_embeds, _, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, None, shifted_attn_mask)
            bos_token_embed = self.get_decoder_embeddings()(torch.tensor([self.llama_tokenizer.bos_token_id]).to(input_embeds.device))
            bos_token_embed = bos_token_embed.repeat(input_embeds.shape[0], 1, 1)
            input_embeds = torch.cat((bos_token_embed, input_embeds), dim=1)
            shifted_attn_mask = torch.cat((shifted_attn_mask.new_ones((batch_size, 1)),  shifted_attn_mask), dim=1)
            outputs = self.decoder.generate(
                inputs_embeds=input_embeds,
                attention_mask=shifted_attn_mask,
                num_beams=4,
                max_new_tokens=256,
                min_length=0,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                pad_token_id=self.llama_tokenizer.pad_token_id
            )
            captions = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


if __name__ == "__main__":
    # SR(48000) * 10 => is_longer, should be used CLAP-fused
    # sample rate should be 48000, if not match resampled when load
    # audio_data, _ = librosa.load('./examples/Yb0RFKhbpFJA.flac', sr=48000)
    # text_data = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    # audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    # audio_data = torch.tensor(audio_data).to("cuda")
    
    def read_wav(wav_path):
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if len(wav) > 30 * sr:
            wav = wav[: 30 * sr]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
        return wav
    wav_1 = read_wav("./examples/Yb0RFKhbpFJA.flac")
    text_1 = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    wav_2 = read_wav("./examples/clotho_example.wav")
    text_2 = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    max_length = max(wav_1.shape[-1], wav_2.shape[-1])
    if wav_1.shape[-1] < max_length:
        pad_length = max_length - wav_1.shape[-1]
        wav_1 = np.pad(wav_1, pad_width=(0, pad_length), mode='constant', constant_values=0.0)
    if wav_2.shape[-1] < max_length:
        pad_length = max_length - wav_2.shape[-1]
        wav_2 = np.pad(wav_2, pad_width=(0, pad_length), mode='constant', constant_values=0.0)
    model = SALMONN(
        whisper_path = "/home/ckddls1321/.cache/checkpoints/whisper/",
        beats_path = "/home/ckddls1321/.cache/checkpoints/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        vicuna_path = "/home/ckddls1321/.cache/checkpoints/vicuna_13B/"
    )
    audio_embed = model.encode_audio([wav_1, wav_2])
    print(audio_embed.shape)
    
    model.train()
    output = model([wav_1, wav_2],[text_1,text_2], [], [])
    print(output)

    # audio_caption_model = CLAP2LLAMA().to("cuda")
    # output = audio_caption_model(audio_data, text_data)
    # print(f"loss : {output['loss']}")
    # print(f"logits : {output['logits'].shape}")  # logits : torch.Size([1, 19, 32000])

    # captions = audio_caption_model.generate_caption(audio_data)
    # print(f"captions : {captions}")

    # captions = audio_caption_model.generate_beam(embed=None, prompt="What is dog? Explain short and breif way. ")
    # print(f"generation with prompt : {captions[0]}")
