import torch
import torch.distributed as dist


def _send_shape(shape: torch.Size | tuple[int, ...], dst: int) -> None:
    ndim = torch.tensor([len(shape)], dtype=torch.int64)
    shape_tensor = torch.tensor(list(shape), dtype=torch.int64)
    dist.send(ndim, dst=dst)
    dist.send(shape_tensor, dst=dst)


def _recv_shape(src: int) -> tuple[int, ...]:
    ndim = torch.empty(1, dtype=torch.int64)
    dist.recv(ndim, src=src)
    shape_tensor = torch.empty(int(ndim.item()), dtype=torch.int64)
    dist.recv(shape_tensor, src=src)
    return tuple(int(v) for v in shape_tensor.tolist())


def send_tensor(tensor: torch.Tensor, dst: int, comm_dtype: torch.dtype) -> tuple[int, ...]:
    # 当前原型先统一走 CPU 浮点通信，先把 stage handoff 跑通，后面再扩多 tensor payload。
    payload = tensor.detach().to("cpu", dtype=comm_dtype).contiguous()
    _send_shape(payload.shape, dst)
    dist.send(payload, dst=dst)
    return tuple(payload.shape)


def recv_tensor(
    src: int,
    device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    shape = _recv_shape(src)
    payload = torch.empty(shape, dtype=comm_dtype)
    dist.recv(payload, src=src)
    return payload.to(device=device, dtype=target_dtype)


def send_hidden_states(hidden_states: torch.Tensor, dst: int, comm_dtype: torch.dtype) -> tuple[int, ...]:
    return send_tensor(hidden_states, dst, comm_dtype)


def recv_hidden_states(
    src: int,
    device: torch.device,
    hidden_dtype: torch.dtype,
    comm_dtype: torch.dtype,
) -> torch.Tensor:
    return recv_tensor(src, device, hidden_dtype, comm_dtype)
