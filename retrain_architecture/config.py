import os
"""
    MASTER_HOST: master node ip
    MASTER_PORT: master node port
    NODE_NUM: # nodes
    MY_RANK: current node idx
    GPU_NUM: # gpus per node
"""
MASTER_HOST = os.environ["HOST_RANK0"]
MASTER_PORT = 13322
NODE_NUM = int(os.environ["WORLD_SIZE"])
MY_RANK = int(os.environ["RANK"])
GPU_NUM = int(os.environ["GPU_COUNT"])