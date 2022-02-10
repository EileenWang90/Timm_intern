#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py "$@"
# --master_port 29501 

