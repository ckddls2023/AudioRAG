# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py


from torch import nn


class Qformer(nn.Module): # Too large
    def __init__(self,
                 num_layers=12,
                 num_attention_heads=12,
                 num_query_token=64,
                 cross_attention_freq=2,
                 hidden_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 attention_probs_dropout_prob=0.1,
                 embed_dim=256,
                 drop_path_rate=0,
                 cross_attention_frequency=2,
                 encoder_hidden_size=768,
                 grad_checkpoint=False,
                 ):
        super().__init__()

    def forward(self, audio_embed, text):
        pass
