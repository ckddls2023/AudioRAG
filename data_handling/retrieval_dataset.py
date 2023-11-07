import faiss
class RetrievalIndex:
    def __init__(self, n_probe=16, use_gpu=False, index_path=""):
        self.datastore = faiss.read_index(index_path)
        self.datastore.nprobe = n_probe

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
        # TODO : Add captions file to return caption

    def get_nns(self, query_audio, k=5):
        D, I = self.datastore.search(query_audio, k)
        # TODO : Also get caption and wav_path from
        return D, I[:, :k]
