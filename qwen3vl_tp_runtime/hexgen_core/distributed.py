"""Small distributed runtime helpers for local prototype execution with gloo."""

import os
import socket

import torch
import torch.distributed as dist


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def init_dist():
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = getenv_int("MASTER_PORT", 29501)
    rank = getenv_int("RANK", 0)
    world_size = getenv_int("WORLD_SIZE", 1)

    print(
        f"[pre-init] host={socket.gethostname()} rank={rank} "
        f"world_size={world_size} master={master_addr}:{master_port}"
    )

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --device cuda，但当前环境里 torch.cuda.is_available() 为 False。")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def all_reduce_cpu(
    local_tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
    group=None,
) -> torch.Tensor:
    # 当前原型默认走 device -> cpu -> gloo -> cpu -> device
    reduced = local_tensor.detach().to("cpu", dtype=comm_dtype)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
    return reduced.to(device=target_device, dtype=target_dtype)


def all_gather_cpu(
    local_tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
    group=None,
) -> list[torch.Tensor]:
    payload = local_tensor.detach().to("cpu", dtype=comm_dtype).contiguous()
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(payload) for _ in range(world_size)]
    dist.all_gather(gathered, payload, group=group)
    return [tensor.to(device=target_device, dtype=target_dtype) for tensor in gathered]


def broadcast_cpu(
    reference_tensor: torch.Tensor,
    tensor: torch.Tensor | None,
    src: int,
    comm_dtype: torch.dtype,
    group=None,
) -> torch.Tensor:
    # stage leader 先拿到完整激活，再在组内广播给同 stage 的 TP rank。
    if tensor is None:
        payload = torch.empty(reference_tensor.shape, dtype=comm_dtype)
    else:
        payload = tensor.detach().to("cpu", dtype=comm_dtype).contiguous()
    dist.broadcast(payload, src=src, group=group)
    return payload.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
