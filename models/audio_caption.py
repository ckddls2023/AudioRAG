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

        # Set 2Layer-MLP following LLAVA paper in NEUIPS 2023
        # modules = [
        #     nn.Linear(self.encoder_config.hidden_size, self.decoder_config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(self.decoder_config.hidden_size, self.decoder_config.hidden_size)
        # ]
        # self.enc_to_dec_proj = nn.Sequential(*modules)

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

        # Ours, token merge with langauge guided selection
        self.enc_to_dec_proj = LGTM(hidden_size=self.encoder_config.hidden_size, num_latents=64)
        self.decoder_proj = nn.Linear(
            self.encoder_config.hidden_size, self.decoder_config.hidden_size
        )

        self.freeze_am = args.freeze_am
        self.freeze_lm = args.freeze_lm

        self.freeze_am = args.freeze_am
        self.freeze_lm = args.freeze_lm

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
            self.peft_config = LoraConfig(  # Following QLoRA, Recent Model use all target to tune LORA
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=['lm_head', 'embed_tokens'],
                target_modules=["q_proj", "v_proj"] # Mini GPT5
            )
            # config = IA3Config(
            #     peft_type="IA3",
            #     task_type="CAUSAL_LM",
            #     target_modules=['q_proj', 'k_proj', 'down_proj'],
            #     feedforward_modules=["down_proj"],
            # )
            self.decoder = get_peft_model(self.decoder, self.peft_config)
            self.decoder.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
            self.decoder.base_model.model.lm_head.original_module.weight.requires_grad = False
            self.decoder.print_trainable_parameters()
        self.decoder.gradient_checkpointing_enable()

    def forward_encoder(self, audios, text=None):
        outputs = self.encoder(audios).last_hidden_state
        outputs = self.enc_to_dec_proj(outputs)
        return outputs

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
        # return outputs

        # Ours, Token merge with language guided selection
        audio_query_output = self.enc_to_dec_proj(outputs, text)
        outputs = self.decoder_proj(audio_query_output.last_hidden_state)

        # Perceiver
        # if isinstance(self.enc_to_dec_proj[0], PerceiverResampler):
        #     outputs = outputs.squeeze(1) # [B,S,H]
        return outputs

    def get_decoder_embeddings(self):
        if self.args.freeze_lm:
            return self.decoder.get_input_embeddings()
        else:
            return self.decoder.base_model.get_input_embeddings()

    def shift_and_pad_input(self, input_ids, attn_mask, prefix_length):
        batch_size, seq_length = input_ids.shape
        shifted_input_ids = input_ids.new_zeros((batch_size, seq_length + prefix_length))
        shifted_input_ids[:, prefix_length:] = input_ids.clone()
        shifted_input_ids[:, :prefix_length] = -100
        shifted_input_ids.masked_fill_(shifted_input_ids == self.tokenizer.pad_token_id, -100)
        shifted_attn_mask = attn_mask.new_zeros((batch_size, seq_length + prefix_length))
        shifted_attn_mask[:, prefix_length:] = attn_mask.clone()
        shifted_attn_mask[:, :prefix_length] = 1
        return shifted_input_ids

    def prepare_text_input(self, text, device):
        tokenized_text = self.tokenizer(text, padding='longest', truncation=True, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(device)
        attn_mask = tokenized_text["attention_mask"].to(device)
        return input_ids, attn_mask

    def forward_decoder(self, audio_embed, text, retr_audio_embeds=None, retr_texts=None):
        input_ids, attn_mask = self.prepare_text_input(text, audio_embed.device)
        input_embeds = torch.cat((audio_embed, self.get_decoder_embeddings()(input_ids)), dim=1)
        shifted_input_ids, shifted_attn_mask = self.shift_and_pad_input(input_ids, attn_mask, audio_embed.shape[1])
        if retr_audio_embeds:
            for retr_audio_embed, retr_text in zip(retr_audio_embeds, retr_texts):
                input_ids, attn_mask = self.prepare_text_input(retr_text, audio_embed.device)
                retr_input_embeds = torch.cat((retr_audio_embed, self.get_decoder_embeddings()(input_ids)), dim=1)
                shifted_retr_ids, shifted_retr_mask = self.shift_and_pad_input(input_ids, attn_mask, retr_audio_embed.shape[1])
                input_embeds = torch.cat((retr_input_embeds, input_embeds), dim=1)
                shifted_input_ids = torch.cat((shifted_retr_ids, shifted_input_ids), dim=1)
                shifted_attn_mask = torch.cat((shifted_retr_mask, shifted_attn_mask), dim=1)
        output = self.decoder(inputs_embeds=input_embeds, labels=shifted_input_ids, attention_mask=shifted_attn_mask)
        return output

    def forward(self, audio, text, retr_audios=None, retr_texts=None):
        audio_embed = self.forward_encoder(audio, text) # Only when needs text..
        retr_audio_embeds = []
        if retr_audios is not None:
            for retr_audio, retr_text in zip(retr_audios, retr_texts):
                retr_embed = self.forward_encoder(retr_audio, retr_text)
                retr_audio_embeds.append(retr_embed)
        output = self.forward_decoder(audio_embed, text, retr_audio_embeds, retr_texts)
        return output

    def save_ckpt(self, checkpoint_path):
        # MLP, Perceiver Resamplerc
        # torch.save(unwrapped_model.enc_to_dec_proj.state_dict(), "pretrained_models/audio_caption/enc_to_dec_proj.bin")
        # For Q-Former, additionally save
        torch.save(self.decoder_proj.state_dict(), checkpoint_path + "decoder_proj.bin")
        # For Ours, LGTM, seperately save
        torch.save(self.enc_to_dec_proj.audio2text_xattn.state_dict(), checkpoint_path+"audio2text_xattn.bin")
        torch.save(self.enc_to_dec_proj.token_merger.state_dict(), checkpoint_path+"token_merger.bin")
        if not self.freeze_lm:
            self.decoder.save_pretrained(checkpoint_path)

    def load_ckpt(self, checkpoint_path):
        # For MLP
        # self.enc_to_dec_proj.load_state_dict(torch.load(checkpoint_path+"enc_to_dec_proj.bin"))
        # For Q-Former, additionally save
        self.decoder_proj.load_state_dict(torch.load(checkpoint_path + "decoder_proj.bin"))
        # For Ours, LGTM, seperately save
        self.enc_to_dec_proj.audio2text_xattn.load_state_dict(torch.load(checkpoint_path+"audio2text_xattn.bin"))
        self.enc_to_dec_proj.token_merger.load_state_dict(torch.load(checkpoint_path+"token_merger.bin"))
        if not self.freeze_lm:
            self.decoder = PeftModel.from_pretrained(self.decoder, checkpoint_path) # suppose don't use get_peft_model

    def generate_caption(self, audio, text=None, retr_audios=None, retr_texts=None, prompt=None):
        r"""Generate audio captions for each audio recording in a batch"""
        # TODO : Instruction tuning? Or Task identifier?
        # TODO : Prefix Tuning? Refer to MINI-GPT5, task adaption or fusion in later context, REVEAL
        with torch.no_grad():
            if retr_texts is not None: # Suppose we only test alignment
                text = retr_texts[0]  # Currently suppose only top-1 used, since encoder length is also important
            input_embeds = self.forward_encoder(audio, text)
            batch_size, seq_length, _ = input_embeds.shape
            shifted_attn_mask = input_embeds.new_zeros((batch_size, seq_length)).long()
            shifted_attn_mask[:, :] = 1
            if retr_audios is not None:
                retr_audio_embeds = []
                for retr_audio, retr_text in zip(retr_audios, retr_texts):
                    retr_embed = self.forward_encoder(retr_audio, retr_text)
                    retr_audio_embeds.append(retr_embed)
                for retr_audio_embed, retr_text in zip(retr_audio_embeds, retr_texts):
                    input_ids, attn_mask = self.prepare_text_input(retr_text, input_embeds.device)
                    retr_input_embeds = torch.cat((retr_audio_embed, self.get_decoder_embeddings()(input_ids)), dim=1)
                    shifted_retr_ids, shifted_retr_mask = self.shift_and_pad_input(input_ids, attn_mask, retr_audio_embed.shape[1])
                    input_embeds = torch.cat((retr_input_embeds, input_embeds), dim=1)
                    shifted_attn_mask = torch.cat((shifted_retr_mask, shifted_attn_mask), dim=1)
            outputs = self.decoder.generate(
                inputs_embeds=input_embeds,
                attention_mask=shifted_attn_mask,
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
