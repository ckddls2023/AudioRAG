import matplotlib.pyplot as plt
import librosa
import numpy as np
from models.audio_encoder import CLAPAudioTower, CLAPEncoderConfig
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

# Load your audio file
# encoder_config = {
#     "model_name": "CLAPAudioEncoder",
#     "pretrained": True,
#     "freeze": True,
#     "use_lora": True,
#     "spec_augment": True,
#     "select_feature": "fine_grained_embedding",
#     "sequence_length": 1024,
#     "hidden_size": 768,
#     "window_size": 4,
#     "step_size": 4,
# }
# encoder_config = CLAPEncoderConfig.from_dict(encoder_config)
# audio_encoder = CLAPAudioTower(encoder_config)
# checkpoint_path = "./retriever_models/"
# audio_encoder_ckpt = os.path.join(checkpoint_path, "audio_encoder.bin")
# if os.path.exists(audio_encoder_ckpt):
#     audio_encoder.load_state_dict(torch.load(audio_encoder_ckpt), strict=False)
#audio_file = './examples/oPRC3zComfA_000117.wav'
#caption = "Electronic music plays as a person skillfully plays a synthesizer, creating a captivating and energetic atmosphere."
#attention_data = np.load("./data/embeddings/Auto_ACD/0000000.npy")
audio_file = './examples/cwPXoYexXrs_000311.wav'
caption = "The sound of rustling followed by digital beeping suggests the opening or closing of power windows in a vehicle."
attention_data = np.load("./data/embeddings/Auto_ACD/0000001.npy")
waveform, sr = librosa.load(audio_file)
print(f"Attention score : {np.sum(attention_data, axis=1)}")
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)
# attention_data = softmax(attention_data)
# audio_waveform = int16_to_float32(float32_to_int16(waveform))
# max_length = 480000
# audio_waveform = torch.from_numpy(audio_waveform).float()
# temp_dict = {}
# temp_dict = get_audio_features(
#     temp_dict, audio_waveform, max_length,
#     data_truncating='fusion',
#     data_filling='repeatpad',
#     audio_cfg=self.audio_cfg,
#     require_grad=False,
# )

# 

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# Plot waveform
axs[0].plot(waveform)
axs[0].set_title(caption)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')

# Plot attention results
for i in range(attention_data.shape[0]):
    axs[1].plot(attention_data[i, :], label=f'Layer {i+1}')
axs[1].set_title('Attention Results')
axs[1].set_xlabel('Tokens')
axs[1].set_ylabel('Attention Value')
axs[1].legend()

plt.tight_layout()
plt.show()