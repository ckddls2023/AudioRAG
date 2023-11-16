import faiss
import pandas as pd

# 필요하다면 나중에 utils 파일에 넣어도 되는 함수.
def load_caption_wav_mapping(csv_path):
    df = pd.read_csv(csv_path)
    return df['caption'].tolist(), df['wav_path'].tolist()
    
class RetrievalIndex:
    def __init__(self, n_probe=16, use_gpu=False, index_path="./data/index", index_types = ["audio", "text"]):
        
        # text index
        self.text_datastore = faiss.read_index(f"{index_path}/text_faiss_index.bin")
        # audio index
        self.audio_datastore = faiss.read_index(f"{index_path}/audio_faiss_index.bin")
        # caption_wav_path.csv
        self.caption_list, self.wav_path_list = load_caption_wav_mapping("./caption_wav_path.csv")
        
        # self.datastore = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        # self.datastore.nprobe = n_probe

        # 보류
        if use_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.useFloat16CoarseQuantizer = True
            co.usePrecomputed = False
            co.indicesOptions = 0
            co.verbose = True
            co.shard = True  # the replicas will be made "manually"
            res = [faiss.StandardGpuResources() for i in range(faiss.num_gpus())]
            self.datastore = faiss.index_cpu_to_gpu_multiple_py(res, self.datastore, co)
            faiss.GpuParameterSpace().set_index_parameter(self.datastore, 'nprobe', n_probe)

    # modal에 따른 query embedding을 만들어주는 함수
    def query_embedding(modal, query_embedding):
        # text_data = ["a dog is barking at a man walking by", "Wind and a man speaking are heard, accompanied by buzzing and ticking."]
        # audio_file = ["./examples/yapping-dog.wav", "./examples/Yb0RFKhbpFJA.flac"]
        
        # from laion_clap import CLAP_Module
        
        # clap_model = CLAP_Module(enable_fusion=True)  # 615M
        # clap_model.load_ckpt()

        # import torch
        
        # clap_model.eval()
        # with torch.no_grad():
        #     # text
        #     text_embed = clap_model.get_text_embedding(text_data, use_tensor=True)
        #     # audio
        #     audio_embed = clap_model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True)
        query_embeddings = query_embedding.cpu().detach().numpy().astype('float32')
        return query_embeddings
    
    # modal에 따른 search 함수
    def get_nns(self, modal, queries, k = 16):
        index = {
            "text": self.text_datastore,
            "audio": self.audio_datastore
        }
        
        D, I = index[modal].search(queries, k)
        return D, I
