# TODO : Add retrieval dataset that can handle retreived results
# When use retrieval index in training, we will use this GpuMultipleClonerOptions
# co = faiss.GpuMultipleClonerOptions()
# co.useFloat16 = use_float16
# co.useFloat16CoarseQuantizer = False
# co.usePrecomputed = use_precomputed_tables
# co.indicesOptions = 0
# co.verbose = True
# co.shard = True  # the replicas will be made "manually"
#vres, vdev = make_vres_vdev()
#vres = faiss.GpuResourcesVector()
#vdev = faiss.IntVector()
#for i in range(0, faiss.num_gpus()):
#    vdev.push_back(i)
#    vres.push_back(gpu_resources[i])
#index = faiss.index_cpu_to_gpu_multiple(
#    vres, vdev, indexall, co)

