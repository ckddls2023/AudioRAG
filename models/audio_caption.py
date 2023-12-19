# Mainly Refer to pengi and SALMONN, MSCLAP
from dataclasses import dataclass
import torch
import librosa
from omegaconf import OmegaConf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, GPT2Config, LlamaConfig, GenerationConfig
from peft import LoraConfig, TaskType, IA3Config, get_peft_model, get_peft_model_state_dict, PeftModel, PeftConfig, PromptEncoderConfig, AdaptionPromptConfig
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig, HTSATAudioTower, HTSATEncoderConfig
from models.flamingo_pytorch import PerceiverResampler
from models.Qformer import *
from models.LGTM import *


class CLAP2LLAMA(nn.Module):

    def __init__(self, config):
        super(CLAP2LLAMA, self).__init__()
        self.config = config
        self.encoder_config = CLAPEncoderConfig.from_dict(OmegaConf.to_container(config.encoder, resolve=True))
        self.encoder = CLAPAudioTower(self.encoder_config)
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")  # v1.5 : LLAMA2 + VICUNA
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, add_eos_token=True,add_bos_token=False)
        #self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.generation_config, unused_kwargs = GenerationConfig.from_pretrained("lmsys/vicuna-7b-v1.5", max_new_tokens=200, return_unused_kwargs=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = 256
        self.retr_prompt = config.retr_prompt
        self.task_prompt = config.task_prompt
        #if self.tokenizer.pad_token is None:
        #    self.tokenizer.pad_token = self.tokenizer.eos_token
        #    self.decoder.config.pad_token_id = self.decoder.config.eos_token_id
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
            self.enc_to_dec_proj.cls = None
            self.enc_to_dec_proj.bert.embeddings.word_embeddings = None
            self.enc_to_dec_proj.bert.embeddings.position_embeddings = None
            for layer in self.enc_to_dec_proj.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.decoder_proj = nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size)
            self.audio_position_embedding = nn.Embedding(256, self.encoder_config.hidden_size)
            self.forward_align = self.forward_qformer

        # Ours, token merge with langauge guided selection
        if self.config.align.model_name == "LGTM":
            self.enc_to_dec_proj = LGTM(hidden_size=self.encoder_config.hidden_size, num_latents=64)
            modules = [
                nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size),
                nn.LayerNorm(self.decoder_config.hidden_size)
            ]
            self.decoder_proj = nn.Sequential(*modules)
            self.forward_align = self.forward_lgtm

        self.freeze_am = config.freeze_am
        self.unfreeze_am = config.unfreeze_am
        self.freeze_lm = config.freeze_lm

        # Freeze all CLAP parameters
        self.decoder.gradient_checkpointing_enable()
        if self.freeze_am:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False
                if any(name.startswith(prefix) for prefix in config.unfreeze_am):
                    p.requires_grad = True
            
        # Freeze all alignment paramters
        # for param in self.enc_to_dec_proj.parameters():
        #     param.requires_grad = False

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
                self.decoder.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
                self.decoder.base_model.model.lm_head.original_module.weight.requires_grad = False
            self.decoder.print_trainable_parameters()


    def forward_mlp(self, x, caption=None):
        outputs = self.enc_to_dec_proj(x)
        return outputs, None

    def forward_qformer(self, outputs, caption=None):
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
        return outputs, None

    def forward_lgtm(self, outputs, caption):
        audio_query_output, loss = self.enc_to_dec_proj(outputs, caption)
        outputs = self.decoder_proj(audio_query_output.last_hidden_state)
        return outputs, loss

    def forward_encoder(self, audios, caption=None):
        outputs = self.encoder(audios).last_hidden_state
        #outputs = outputs / outputs.norm(2, -1).unsqueeze(-1)  # Normalize embedding
        outputs, loss = self.forward_align(outputs, caption) # loss is None for MLP, Perceiver, Q-former
        return outputs, loss

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

    def forward_decoder(self, audio_embed, caption, retr_audio_embeds=None, retr_captions=None):
        input_ids, attn_mask = self.prepare_text_input(caption, audio_embed.device)
        input_embeds = torch.cat((audio_embed, self.get_decoder_embeddings()(input_ids)), dim=1)
        shifted_input_ids, shifted_attn_mask = self.shift_and_pad_input(input_ids, attn_mask, audio_embed.shape[1])
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt(self.task_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)
        retr_nums = max(len(retr_audio_embeds), len(retr_captions))
        retr_nums = 0
        for i in range(retr_nums): # Suppose we have only case ([],retr_captions), (retr_audio_embeds,[]), both pair with same length
            if retr_captions: 
                retr_input_ids, retr_attn_mask = self.prepare_text_input(retr_captions[i], audio_embed.device, add_special_tokens=False)
                input_embeds = torch.cat((self.get_decoder_embeddings()(retr_input_ids), input_embeds), dim=1) 
                retr_input_ids[:,:] = -100 # Ignore all Wavcaps style
                #retr_input_ids.masked_fill_(retr_input_ids == self.tokenizer.pad_token_id, -100)
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat((retr_attn_mask, shifted_attn_mask), dim=1)
            if retr_audio_embeds: 
                input_embeds = torch.cat((retr_audio_embeds[i], input_embeds), dim=1) 
                retr_input_ids = input_ids.new_zeros(input_ids.shape[0],retr_audio_embeds[i].shape[1]) # B, prefix_length
                retr_input_ids[:,:] = -100 # Ignore all embeds
                shifted_input_ids = torch.cat((retr_input_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat((shifted_attn_mask.new_ones(retr_audio_embeds[i].size()[:2]), shifted_attn_mask), dim=1)
        input_embeds, shifted_input_ids, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, shifted_input_ids, shifted_attn_mask)
        output = self.decoder(inputs_embeds=input_embeds, labels=shifted_input_ids, attention_mask=shifted_attn_mask)
        return output

    def forward(self, audio, caption, retr_audios=None, retr_captions=None):
        if retr_captions:
            encoder_caption = [' '.join(caption) for caption in zip(*retr_captions)] # B,K to B
        audio_embed, loss = self.forward_encoder(audio, encoder_caption)  # Only for LGTM
        retr_audio_embeds = []
        if retr_audios:
            for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)):
                encoder_captions = retr_captions[:i] + retr_captions[i+1:] # B, K
                encoder_caption = [' '.join(caption) for caption in zip(*encoder_captions)] # B,K to B
                #encoder_caption = retr_caption
                retr_embed, _ = self.forward_encoder(retr_audio, encoder_caption)
                if self.config.align.model_name == "LGTM": # Should be detached..?
                    retr_embed = retr_embed.detach()
                retr_audio_embeds.append(retr_embed)
        output = self.forward_decoder(audio_embed, caption, retr_audio_embeds, retr_captions)
        if loss is not None:
            output["loss"] += output["loss"] + 0.1 * loss
        return output

    def save_ckpt(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if self.config.align.model_name == "MLP" or self.config.align.model_name == "Perceiver":
            torch.save(self.enc_to_dec_proj.state_dict(), checkpoint_path + "enc_to_dec_proj.bin")
        if self.config.align.model_name == "Qformer" or self.config.align.model_name == "LGTM":
            torch.save(self.decoder_proj.state_dict(), checkpoint_path + "decoder_proj.bin")
        if self.config.align.model_name == "LGTM":
            full_state_dict = self.enc_to_dec_proj.state_dict()
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if 'text_encoder' not in k}
            torch.save(filtered_state_dict, checkpoint_path + "enc_to_dec_proj.bin")
        if self.unfreeze_am:
            torch.save(self.encoder.state_dict(), checkpoint_path + "audio_encoder.bin")
        if not self.freeze_lm:
            self.decoder.save_pretrained(checkpoint_path)

    def load_ckpt(self, checkpoint_path):
        print("Load model from checkpoint")
        if self.config.align.model_name == "MLP" or self.config.align.model_name == "Perceiver":
            self.enc_to_dec_proj.load_state_dict(torch.load(checkpoint_path + "enc_to_dec_proj.bin"), strict=True)
        if self.config.align.model_name == "Qformer" or self.config.align.model_name == "LGTM":
            self.decoder_proj.load_state_dict(torch.load(checkpoint_path + "decoder_proj.bin"), strict = True)
        if self.config.align.model_name == "LGTM":
            self.enc_to_dec_proj.load_state_dict(torch.load(checkpoint_path+"enc_to_dec_proj.bin"), strict=False)
        if self.unfreeze_am:
            file_path = os.path.join(checkpoint_path, "audio_encoder.bin")
            if os.path.exists(file_path):
                self.encoder.load_state_dict(torch.load(file_path), strict=False)
        if not self.freeze_lm and 'finetuned' in checkpoint_path:
            print("Load LORA model")
            self.decoder = PeftModel.from_pretrained(self.decoder.base_model, checkpoint_path, config=self.peft_config)  # suppose don't use get_peft_model

    def generate_caption(self, audio, caption=None, retr_audios=None, retr_captions=None):
        r"""Generate audio captions for each audio recording in a batch"""
        with torch.no_grad():
            encoder_caption = []
            if retr_captions:
                encoder_caption = [' '.join(caption) for caption in zip(*retr_captions)] # B,K to B
            input_embeds, loss = self.forward_encoder(audio, encoder_caption) # Only for LGTM, dict type will be better...
            batch_size, seq_length, _ = input_embeds.shape
            shifted_attn_mask = input_embeds.new_ones((batch_size, seq_length)).long()
            retr_audio_embeds = []
            for i, (retr_audio, retr_caption) in enumerate(zip(retr_audios, retr_captions)): # Only LGTM needs retr_text, others ignore
                #encoder_captions = retr_captions[:i] + retr_captions[i+1:] # B, K
                #encoder_caption = [' '.join(caption) for caption in zip(*encoder_captions)] # B,K to B
                encoder_caption = retr_caption
                retr_embed, loss = self.forward_encoder(retr_audio, encoder_caption)
                retr_audio_embeds.append(retr_embed)
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
            input_embeds, _, shifted_attn_mask = self.insert_prompt(self.retr_prompt, input_embeds, None, shifted_attn_mask)
            outputs = self.decoder.generate(
                inputs_embeds=input_embeds,
                attention_mask=shifted_attn_mask,
                num_beams=2,
                min_length=0,
                max_new_tokens=256,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.0,
                use_cache=True,
                #generation_config=self.generation_config,
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

    audio_caption_model = CLAP2LLAMA().to("cuda")
    output = audio_caption_model(audio_data, text_data)
    print(f"loss : {output['loss']}")
    print(f"logits : {output['logits'].shape}")  # logits : torch.Size([1, 19, 32000])

    captions = audio_caption_model.generate_caption(audio_data)
    print(f"captions : {captions}")

    # captions = audio_caption_model.generate_beam(embed=None, prompt="What is dog? Explain short and breif way. ")
    # print(f"generation with prompt : {captions[0]}")
