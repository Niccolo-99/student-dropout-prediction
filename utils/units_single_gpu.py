import torch
import torch.distributed as dist
import sys
import builtins

# Verifica GPU
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

# Salva le funzioni originali
original_is_initialized = dist.is_initialized
original_get_rank = dist.get_rank
original_get_world_size = dist.get_world_size
if hasattr(dist, 'all_gather'):
    original_all_gather = dist.all_gather
if hasattr(dist, 'barrier'):
    original_barrier = dist.barrier

# Sovrascrivi le funzioni per il debugging
def mock_is_initialized():
    return False

def mock_get_rank():
    return 0

def mock_get_world_size():
    return 1

# Funzioni di barriera e gather che non fanno nulla
def mock_barrier():
    pass

def mock_all_gather(output_tensor_list, input_tensor, *args, **kwargs):
    if isinstance(output_tensor_list, list) and len(output_tensor_list) > 0:
        output_tensor_list[0].copy_(input_tensor)

# Patch per gather_tensors_from_all_gpus
def mock_gather_tensors_from_all_gpus(tensor_list, device_id, to_numpy=True):
    import numpy as np
    if to_numpy:
        return [tensor.cpu().numpy() for tensor in tensor_list]
    else:
        return tensor_list

# Applica le patch
dist.is_initialized = mock_is_initialized
dist.get_rank = mock_get_rank
dist.get_world_size = mock_get_world_size
dist.barrier = mock_barrier
dist.all_gather = mock_all_gather

# Patch per utils.ddp se è già stato importato
if 'utils.ddp' in sys.modules:
    import utils.ddp
    utils.ddp.gather_tensors_from_all_gpus = mock_gather_tensors_from_all_gpus

print("Applied debugging patches to PyTorch distributed functions")