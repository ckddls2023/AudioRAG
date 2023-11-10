import types
import librosa
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput
from laion_clap import CLAP_Module
from .htsat import HTSAT_Swin_Transformer

class CLAPEncoderConfig(PretrainedConfig):
    model_type = "audio_encoder"

    def __init__(self,
                 model_name: str = "CLAPAudioEncoder",
                 pretrained: bool = True,
                 freeze: bool = True,
                 spec_augment: bool = True,
                 audio_args: dict = None,
                 select_feature = "embedding",
                 **kwargs):
        super(CLAPEncoderConfig, self).__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.select_feature = select_feature # fine-grained, embedding, projected
        self.sequence_length = 1024
        self.hidden_size = 768
        self.window_size = 32 # Fold and Unfold
        self.step_size = 16

class CLAPAudioTower(PreTrainedModel):
    config_class = CLAPEncoderConfig

    def __init__(self, config):
        super(CLAPAudioTower, self).__init__(config)

        self.clap = CLAP_Module(enable_fusion=True)  # 615M
        if config.pretrained:
            self.clap.load_ckpt()  # download the default pretrained checkpoint.
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
    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):
        outputs = self.clap.get_audio_embedding_from_data(x=input_ids, use_tensor=True)  # B, 768
        if not return_dict:
            return (outputs,)
        return BaseModelOutput(outputs, None, None)

class HTSATEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an Audio Encoder. It is used to instantiate an
    an Audio Encoder according to the specified arguments, defining the model architecture.
    The audio encoder can be a PANNs model or a HTSAT.
    """
    model_type = "audio_encoder"

    def __init__(self,
                 model_name: str = "HTSATAudioEncoder",
                 pretrained: bool = True,
                 freeze: bool = False,
                 spec_augment: bool = True,
                 audio_args: dict = None,
                 **kwargs):
        super(HTSATEncoderConfig, self).__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze = freeze
        self.hidden_size = 768
        self.spec_augment = spec_augment
        self.audio_args = audio_args
        self.num_labels = 0
        self.sr = 32000
        self.n_fft = 1024
        self.hop_length = 320
        self.f_min = 50
        self.f_max = 14000
        self.n_mels = 64
        self.max_length = 10
        self.mono = True


class HTSATAudioTower(PreTrainedModel):
    config_class = HTSATEncoderConfig

    def __init__(self, config):
        super(HTSATAudioTower, self).__init__(config)

        self.audio_enc = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            patch_stride=(4, 4),
            num_classes=527,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            config=config
        )
        if config.pretrained:
            audio_ckpt = torch.load("pretrained_models/audio_encoder/HTSAT.ckpt", map_location="cpu")["state_dict"]
            for key in list(audio_ckpt.keys()):
                if key.startswith('sed_model') and (
                        'spectrogram_extractor' not in key and 'logmel_extractor' not in key):
                    v = audio_ckpt.pop(key)
                    audio_ckpt[key[10:]] = v
            self.audio_enc.load_state_dict(audio_ckpt, strict=False)
        self.audio_width = 768

        if config.freeze:
            for name, param in self.audio_enc.named_parameters():
                if "fc1" not in name:
                    param.requires_grad = False

    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):
        audio_embeds = self.audio_enc(input_ids)
        if not return_dict:
            return (audio_embeds,)
        return BaseModelOutput(audio_embeds, None, None)


if __name__ == "__main__":
    # SR(32000) * 10 => is_longer
    audio_data, _ = librosa.load('../examples/yapping-dog.wav', sr=48000)  # sample rate should be 48000
    audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
    # When using huggingface
    # processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    # inputs = processor(audios=audio_data, return_tensors="pt")
    # # print(inputs["input_features"].shape) #([1, 4, 1001, 64]))
    # model = CLAPAudioTower(audio_tower="laion/clap-htsat-fused")
    # print(model)
    # print(model(**inputs).shape)

    config = HTSATEncoderConfig(pretrained=False)
    model = HTSATAudioTower(config)
    # # Extract audio embeddings
    # audio_data input format : (1, T), (N,T)
    a = torch.randn(1, 32000)
    output_dict = model(a)
    print(output_dict.last_hidden_state.shape) # 1, 1024, 768
    # output_dict = model(torch.from_numpy(audio_data)) # Out of Time, Only less then 10s is possible(Dataset will support)
    # print(output_dict.last_hidden_state.shape)  # OOT

    config = CLAPEncoderConfig(pretrained=False)
    model = CLAPAudioTower(config)
    output_dict = model(a)
    print(output_dict.last_hidden_state.shape) # 1, 768
    output_dict = model(torch.from_numpy(audio_data)) # Out of Time, Only less then 10s is possible(Dataset will support)
    print(output_dict.last_hidden_state.shape)  # 1, 768

    # TODO : We can use Whisper to additionally enable speech encoder.
    # TODO : We can use Beats to additionally enable speech encoder.
