#!/bin/bash

export hosts="grs30c"
export NUM_HOSTS=$(echo $hosts | tr ',' '\n' | wc -l)
export MASTER_ADDR=$(echo $hosts | tr ',' '\n' | head -n 1)
export MASTER_PORT=29502
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If not set, use nvidia-smi to get the number of GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    # If set, count the number of GPUs in CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
fi


accelerate launch --num_processes $NUM_GPUS --num_machines $NUM_HOSTS --multi_gpu --mixed_precision bf16 --machine_rank 0 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT train.py --config configs/train.yaml

# For Multi-node
#SSH='ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $hostn'
#export ssh_hosts=($(echo $hosts | tr ',' " "))
#(
#for hostn in $ssh_hosts
#do
#    $(eval echo $SSH) "cd $PWD; accelerate launch --num_processes $(( 4 * $NUM_HOSTS )) --num_machines $NUM_HOSTS --multi_gpu --mixed_precision fp16 --machine_rank 0 --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT pretrain.py --config_file accelerate_config.json"
#    pids+=($!);
#    wait "${pids[@]}"
#) |& tee experiment.log
#

