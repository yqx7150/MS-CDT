#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.run --nproc_per_node=1 --master_port=4293 MSCDT/train.py -opt options/train_MSCDTS2.yml --launcher pytorch
