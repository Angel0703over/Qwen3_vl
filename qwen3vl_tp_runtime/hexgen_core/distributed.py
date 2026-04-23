"""Small distributed runtime helpers for local prototype execution with gloo."""

from contextlib import contextmanager
import io
import os
import socket
import time
from typing import Any

import torch
import torch.distributed as dist


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _startup_log_enabled() -> bool:
    value = os.getenv("HEXGEN_STARTUP_LOG", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def startup_log(component: str, message: str) -> None:
    if not _startup_log_enabled():
        return
    host = socket.gethostname()
    rank = os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    print(
        f"[startup][{component}] host={host} rank={rank} local_rank={local_rank} {message}",
        flush=True,
    )


@contextmanager
def startup_timer(component: str, message: str):
    start = time.perf_counter()
    startup_log(component, f"begin {message}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        startup_log(component, f"fail {message} after {elapsed:.2f}s: {exc!r}")
        raise
    else:
        elapsed = time.perf_counter() - start
        startup_log(component, f"done {message} in {elapsed:.2f}s")


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
        device_count = torch.cuda.device_count()
        local_rank = getenv_int("LOCAL_RANK", 0)
        device_index = 0 if device_count <= 1 else local_rank % device_count
        torch.cuda.set_device(device_index)
        return torch.device("cuda", device_index)
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


def _serialize_object_to_uint8(payload: Any) -> torch.Tensor:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    view = memoryview(buffer.getbuffer())
    return torch.frombuffer(view, dtype=torch.uint8).clone()


def _deserialize_object_from_uint8(payload: torch.Tensor) -> Any:
    if payload.numel() == 0:
        return None
    buffer = io.BytesIO(payload.cpu().numpy().tobytes())
    return torch.load(buffer, map_location="cpu")


def send_object_cpu(payload: Any, dst: int, label: str | None = None) -> None:
    serialized = _serialize_object_to_uint8(payload)
    size = torch.tensor([serialized.numel()], dtype=torch.int64)
    startup_log(
        "object-send",
        f"sending {label or 'payload'} to dst={dst} bytes={int(size.item())}",
    )
    dist.send(size, dst=dst)
    if serialized.numel() > 0:
        dist.send(serialized.contiguous(), dst=dst)
    startup_log(
        "object-send",
        f"sent {label or 'payload'} to dst={dst} bytes={int(size.item())}",
    )


def recv_object_cpu(src: int, label: str | None = None) -> Any:
    startup_log("object-recv", f"waiting {label or 'payload'} from src={src}")
    size = torch.empty(1, dtype=torch.int64)
    dist.recv(size, src=src)
    payload = torch.empty(int(size.item()), dtype=torch.uint8)
    if payload.numel() > 0:
        dist.recv(payload, src=src)
    startup_log(
        "object-recv",
        f"received {label or 'payload'} from src={src} bytes={int(size.item())}",
    )
    return _deserialize_object_from_uint8(payload)


def broadcast_object_cpu(payload: Any | None, src: int, group=None, label: str | None = None) -> Any:
    if payload is None:
        serialized = None
        size = torch.tensor([0], dtype=torch.int64)
    else:
        serialized = _serialize_object_to_uint8(payload)
        size = torch.tensor([serialized.numel()], dtype=torch.int64)

    startup_log(
        "object-bcast",
        f"broadcasting {label or 'payload'} src={src} local_has_payload={payload is not None}",
    )
    dist.broadcast(size, src=src, group=group)

    if serialized is None:
        serialized = torch.empty(int(size.item()), dtype=torch.uint8)
    elif serialized.numel() != int(size.item()):
        raise RuntimeError(
            f"broadcast_object_cpu size mismatch: local={serialized.numel()} expected={int(size.item())}"
        )

    if serialized.numel() > 0:
        dist.broadcast(serialized, src=src, group=group)
    startup_log(
        "object-bcast",
        f"broadcast done for {label or 'payload'} src={src} bytes={int(size.item())}",
    )
    return _deserialize_object_from_uint8(serialized)
