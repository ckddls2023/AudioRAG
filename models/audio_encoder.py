import types
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from laion_clap import CLAP_Module

class CLAPEncoderConfig(PretrainedConfig):
    model_type = "audio_encoder"

    def __init__(self,
                 model_name: str = "CLAPAudioEncoder",
                 pretrained: bool = True,
                 freeze: bool = True,
                 spec_augment: bool = True,
                 use_lora: bool = False,
                 audio_args: dict = None,
                 select_feature = "embedding",
                 **kwargs):
        super(CLAPEncoderConfig, self).__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.use_lora = use_lora
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.select_feature = select_feature # fine-grained, embedding, projected
        self.sequence_length = 1024
        self.hidden_size = 768
        self.window_size = 4 # (32,32) = [B,32,H], (16,16) = [B,64,H], (8,8) = [B,128,H] (4,4) = [B,256,H]
        self.step_size = 4

class LoRA_qkv_Linear(nn.Linear):
    def __init__(self, in_features, out_features, rank=8, bias=True):
        # Initialize the base Linear class
        super(LoRA_qkv_Linear, self).__init__(in_features, out_features*3, bias=bias)
        self.linear_a_q = nn.Linear(in_features, rank, bias=False)
        self.linear_b_q = nn.Linear(rank, in_features, bias=False)
        self.linear_a_v = nn.Linear(in_features, rank, bias=False)
        self.linear_b_v = nn.Linear(rank, in_features, bias=False)
        nn.init.zeros_(self.linear_a_q)
        nn.init.zeros_(self.linear_a_v)
        nn.init.normal_(self.linear_b_q)
        nn.init.normal_(self.linear_b_v)
        self.dropout = nn.Dropout(p=0.05)
        self.scaling = 16 # lora_alpha

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)  # B, N, 3*org_C
        new_q = self.linear_b_q(self.dropout(self.linear_a_q(x))) * self.scaling
        new_v = self.linear_b_v(self.dropout(self.linear_a_v(x))) * self.scaling
        qkv[:, :, :self.head_dim] += new_q
        qkv[:, :, -self.head_dim:] += new_v
        return qkv

def replace_qkv_with_lora(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            if child_name == 'qkv':
                lora_qkv = LoRA_qkv_Linear(module.in_features, module.out_features // 3, rank, bias=True)
                lora_qkv.weight.data.copy_(module.weight.data)
                lora_qkv.bias.data.copy_(module.bias.data)
                setattr(parent_module, child_name, lora_qkv)

class CLAPAudioTower(PreTrainedModel):
    config_class = CLAPEncoderConfig

    def __init__(self, config):
        super(CLAPAudioTower, self).__init__(config)

        self.clap = CLAP_Module(enable_fusion=True)  # 615M
        if config.pretrained:
            self.clap.load_ckpt()  # download the default pretrained checkpoint.
        if config.use_lora: # Replace qkv layer with LoRA
            relpace_qkv_with_lora(self.clap, rank=8)
        def get_audio_embedding_patch(self, data,
                                      select_feature=self.config.select_feature,
                                      window_size=self.config.window_size,
                                      step_size=self.config.step_size):
            device = next(self.parameters()).device
            input_dict = {}
            keys = data[0].keys()
            for k in keys:
                input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
            audio_embeds = self.encode_audio(input_dict, device=device)
            if select_feature == "fine_grained_embedding":
                embeds = audio_embeds[select_feature] # [B,1024,768]
                unfolded = embeds.unfold(1, window_size, step_size) # [B,1024/S,768,W]
                averaged = unfolded.mean(dim=-1) # [B,1024/S,768]
                return averaged
            else:
                return audio_embeds[select_feature]
        self.clap.model.get_audio_embedding = types.MethodType(get_audio_embedding_patch, self.clap.model)

    @torch.no_grad()
    def forward(self, data, return_dict=True):
        # Info : Suppose input_ids already processed from given data loader, we directly call get_audio_embedding
        # data : dictionary of "longer", "waveform"
        outputs = self.clap.model.get_audio_embedding(data)  # B, 768
        if not return_dict:
            return (outputs,)
        return BaseModelOutput(outputs, None, None)
