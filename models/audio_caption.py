# Mainly Refer to pengi and SALMONN, MSCLAP
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, GPT2Config, LlamaConfig
from audio_encoder import CLAPAudioTower, CLAPEncoderConfig, HTSATAudioTower, HTSATEncoderConfig

class CLAP2GPT2(nn.Module):

    def __init__(self):
        super(CLAP2GPT2, self).__init__()
        self.encoder_config = CLAPEncoderConfig()
        self.prefix_length = 1 # Only use embedding before projection, if fine-grained, re-calculate
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
        outputs = outputs / outputs.norm(2, -1).reshape(-1, 1) # Normalize embedding
        outputs = self.enc_to_dec_proj(outputs)
        return outputs

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), -100, dtype=torch.int64, device=device)

    def get_dummy_attn_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), 1, dtype=torch.int64, device=device)

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text,padding='longest',truncation=True,max_length=256,return_tensors="pt")
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

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        # self.decoder.force_bos_token_to_be_generated = True

        encoder_outputs = self.forward_encoder(samples)

        input_ids = torch.zeros((encoder_outputs.size(0), 1)).long().to(encoder_outputs.device)
        input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_attention_mask = torch.ones((encoder_outputs.size(0), 1)).long().to(encoder_outputs.device)

        # Maybe outdated..
        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(input_ids=None,
                                            attention_mask=None,
                                            decoder_input_ids=input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=None,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


class CLAP2LLAMA(nn.Module):

    def __init__(self):
        super(CLAP2LLAMA, self).__init__()
        self.encoder_config = CLAPEncoderConfig()
        self.prefix_length = 1 # Only use embedding before projection, if fine-grained, re-calculate
        self.encoder = CLAPAudioTower(self.encoder_config)
        self.decoder = LlamaForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
        # LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
        target_modules = None
        # target_modules = ['q_proj', 'v_proj']
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        self.decoder = get_peft_model(self.decoder, self.peft_config)
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"
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
        outputs = outputs / outputs.norm(2, -1).reshape(-1, 1) # Normalize embedding
        outputs = self.enc_to_dec_proj(outputs)
        return outputs

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), -100, dtype=torch.int64, device=device)

    def get_dummy_attn_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.prefix_length), 1, dtype=torch.int64, device=device)

    def forward_decoder(self, text, encoder_outputs):
        text = self.tokenizer(text,padding='longest',truncation=True,max_length=256,return_tensors="pt")
        input_ids = text["input_ids"].to(encoder_outputs.device)
        attention_mask = text["attention_mask"].to(encoder_outputs.device) # [B, 768], fine-grained [B,32,768]
        embedding_text = self.decoder.base_model.get_input_embeddings()(input_ids) # PEFT : model.base_model
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

    def generate(self,
                 samples,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):

        # self.decoder.force_bos_token_to_be_generated = True

        encoder_outputs = self.forward_encoder(samples)

        input_ids = torch.zeros((encoder_outputs.size(0), 1)).long().to(encoder_outputs.device)
        input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        # input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_attention_mask = torch.ones((encoder_outputs.size(0), 1)).long().to(encoder_outputs.device)

        # Maybe outdated..
        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(input_ids=None,
                                            attention_mask=None,
                                            decoder_input_ids=input_ids,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=None,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)

        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


if __name__ == "__main__":
    # SR(32000) * 10 => is_longer
    audio_data, _ = librosa.load('../examples/Yb0RFKhbpFJA.flac', sr=48000)  # sample rate should be 48000
    audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    audio_data = torch.tensor(audio_data).to("cuda")

    text_data = "Wind and a man speaking are heard, accompanied by buzzing and ticking."
    audio_caption_model = CLAP2GPT2().to("cuda")
    output = audio_caption_model(audio_data, text_data)
    print(f"loss : {output['loss']}")
    print(f"logits : {output['logits'].shape}") # logits : torch.Size([1, 15, 50257])

    audio_caption_model = CLAP2LLAMA().to("cuda")
    output = audio_caption_model(audio_data, text_data)
    print(f"loss : {output['loss']}")
    print(f"logits : {output['logits'].shape}") # logits : torch.Size([1, 19, 32000])

    # TODO : Use fine-grained embedding and Q-former...?
    # TODO : Impl retrieved embedding fusin using gated xattn?

