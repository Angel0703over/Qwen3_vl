"""Small distributed runtime helpers for local prototype execution with gloo."""

from contextlib import contextmanager
import io
import os
import pickle
import socket
import time
from typing import Any

import torch
import torch.distributed as dist


_PICKLE_WIRE_FORMAT_MAGIC = b"HXPK1"
_TORCH_WIRE_FORMAT_MAGIC = b"HXTS1"
_STARTUP_TIMING_EVENTS: list[dict[str, Any]] = []
_TRANSPORT_PROFILE_EVENTS: list[dict[str, Any]] = []


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


def reset_startup_timing_events() -> None:
    _STARTUP_TIMING_EVENTS.clear()


def get_startup_timing_events() -> list[dict[str, Any]]:
    return [dict(event) for event in _STARTUP_TIMING_EVENTS]


def reset_transport_profile_events() -> None:
    _TRANSPORT_PROFILE_EVENTS.clear()


def get_transport_profile_events() -> list[dict[str, Any]]:
    return [dict(event) for event in _TRANSPORT_PROFILE_EVENTS]


def _shape_to_list(shape: Any) -> list[int] | None:
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _classify_transport_kind(channel: str, label: str | None) -> str:
    lowered = f"{channel} {label or ''}".lower()
    if "startup_contract" in lowered:
        return "startup_contract"
    if (
        "stage_scaffold" in lowered
        or "text_scaffold" in lowered
        or "runtime_inputs" in lowered
        or "stage_state" in lowered
    ):
        return "scaffold"
    if "stage_handoff" in lowered:
        return "stage_handoff"
    if "all_reduce" in lowered or "all_gather" in lowered or "broadcast_cpu" in lowered:
        return "tp_collective"
    return "other"


def _summary_to_profile_fields(summary: Any | None) -> dict[str, Any]:
    if summary is None:
        return {}
    tensor_shapes = getattr(summary, "tensor_shapes", {}) or {}
    tensor_dtypes = getattr(summary, "tensor_dtypes", {}) or {}
    tensor_numels = getattr(summary, "tensor_numels", {}) or {}
    tensor_bytes = getattr(summary, "tensor_bytes", {}) or {}
    return {
        "is_empty": bool(getattr(summary, "is_empty", False)),
        "num_tensors": int(getattr(summary, "num_tensors", 0)),
        "payload_keys": list(getattr(summary, "payload_keys", []) or []),
        "tensor_shapes": {
            key: _shape_to_list(value)
            for key, value in tensor_shapes.items()
        },
        "tensor_dtypes": {
            key: (None if value is None else str(value))
            for key, value in tensor_dtypes.items()
        },
        "tensor_numels": {
            key: int(value)
            for key, value in tensor_numels.items()
        },
        "tensor_bytes": {
            key: int(value)
            for key, value in tensor_bytes.items()
        },
        "total_tensor_bytes": int(getattr(summary, "total_tensor_bytes", 0) or 0),
    }


def _single_tensor_profile(tensor: torch.Tensor, *, dtype: torch.dtype) -> dict[str, Any]:
    shape = tuple(tensor.shape)
    numel = int(tensor.numel())
    element_size = torch.empty((), dtype=dtype).element_size()
    tensor_bytes = int(numel * element_size)
    return {
        "is_empty": False,
        "num_tensors": 1,
        "payload_keys": ["tensor"],
        "tensor_shapes": {"tensor": _shape_to_list(shape)},
        "tensor_dtypes": {"tensor": str(dtype)},
        "tensor_numels": {"tensor": numel},
        "tensor_bytes": {"tensor": tensor_bytes},
        "total_tensor_bytes": tensor_bytes,
    }


def record_transport_profile_event(
    *,
    channel: str,
    operation: str,
    label: str | None = None,
    peer: int | None = None,
    summary: Any | None = None,
    object_bytes: int | None = None,
    elapsed_seconds: float | None = None,
    status: str = "done",
    extra: dict[str, Any] | None = None,
) -> None:
    event = {
        "kind": _classify_transport_kind(channel, label),
        "channel": channel,
        "operation": operation,
        "label": label,
        "peer": peer,
        "status": status,
        "elapsed_seconds": None if elapsed_seconds is None else round(float(elapsed_seconds), 6),
        "object_bytes": object_bytes,
        **_summary_to_profile_fields(summary),
    }
    if extra:
        event.update(extra)
    _TRANSPORT_PROFILE_EVENTS.append(event)


def _record_startup_timing(
    component: str,
    message: str,
    elapsed_seconds: float,
    *,
    status: str = "done",
) -> None:
    _STARTUP_TIMING_EVENTS.append(
        {
            "component": component,
            "message": message,
            "status": status,
            "elapsed_seconds": round(float(elapsed_seconds), 6),
        }
    )


@contextmanager
def startup_timer(component: str, message: str):
    start = time.perf_counter()
    startup_log(component, f"begin {message}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        _record_startup_timing(component, message, elapsed, status="fail")
        startup_log(component, f"fail {message} after {elapsed:.2f}s: {exc!r}")
        raise
    else:
        elapsed = time.perf_counter() - start
        _record_startup_timing(component, message, elapsed)
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
    start = time.perf_counter()
    reduced = local_tensor.detach().to("cpu", dtype=comm_dtype)
    try:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=group)
    except Exception:
        record_transport_profile_event(
            channel="all_reduce_cpu",
            operation="all_reduce",
            label="tp_all_reduce",
            summary=None,
            elapsed_seconds=time.perf_counter() - start,
            status="fail",
            extra=_single_tensor_profile(local_tensor, dtype=comm_dtype),
        )
        raise
    record_transport_profile_event(
        channel="all_reduce_cpu",
        operation="all_reduce",
        label="tp_all_reduce",
        summary=None,
        elapsed_seconds=time.perf_counter() - start,
        extra=_single_tensor_profile(local_tensor, dtype=comm_dtype),
    )
    return reduced.to(device=target_device, dtype=target_dtype)


def all_gather_cpu(
    local_tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
    group=None,
) -> list[torch.Tensor]:
    start = time.perf_counter()
    payload = local_tensor.detach().to("cpu", dtype=comm_dtype).contiguous()
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(payload) for _ in range(world_size)]
    try:
        dist.all_gather(gathered, payload, group=group)
    except Exception:
        record_transport_profile_event(
            channel="all_gather_cpu",
            operation="all_gather",
            label="tp_all_gather",
            summary=None,
            elapsed_seconds=time.perf_counter() - start,
            status="fail",
            extra={
                **_single_tensor_profile(local_tensor, dtype=comm_dtype),
                "world_size": int(world_size),
                "total_tensor_bytes": int(payload.numel() * payload.element_size() * world_size),
            },
        )
        raise
    record_transport_profile_event(
        channel="all_gather_cpu",
        operation="all_gather",
        label="tp_all_gather",
        summary=None,
        elapsed_seconds=time.perf_counter() - start,
        extra={
            **_single_tensor_profile(local_tensor, dtype=comm_dtype),
            "world_size": int(world_size),
            "total_tensor_bytes": int(payload.numel() * payload.element_size() * world_size),
        },
    )
    return [tensor.to(device=target_device, dtype=target_dtype) for tensor in gathered]


def broadcast_cpu(
    reference_tensor: torch.Tensor,
    tensor: torch.Tensor | None,
    src: int,
    comm_dtype: torch.dtype,
    group=None,
) -> torch.Tensor:
    # stage leader 先拿到完整激活，再在组内广播给同 stage 的 TP rank。
    start = time.perf_counter()
    if tensor is None:
        payload = torch.empty(reference_tensor.shape, dtype=comm_dtype)
    else:
        payload = tensor.detach().to("cpu", dtype=comm_dtype).contiguous()
    try:
        dist.broadcast(payload, src=src, group=group)
    except Exception:
        record_transport_profile_event(
            channel="broadcast_cpu",
            operation="broadcast",
            label="tp_broadcast_cpu",
            peer=src,
            summary=None,
            elapsed_seconds=time.perf_counter() - start,
            status="fail",
            extra=_single_tensor_profile(reference_tensor, dtype=comm_dtype),
        )
        raise
    record_transport_profile_event(
        channel="broadcast_cpu",
        operation="broadcast",
        label="tp_broadcast_cpu",
        peer=src,
        summary=None,
        elapsed_seconds=time.perf_counter() - start,
        extra=_single_tensor_profile(reference_tensor, dtype=comm_dtype),
    )
    return payload.to(device=reference_tensor.device, dtype=reference_tensor.dtype)


def _payload_can_use_pickle_wire_format(payload: Any) -> bool:
    if torch.is_tensor(payload):
        return False
    if payload is None or isinstance(payload, (bool, int, float, str, bytes)):
        return True
    if isinstance(payload, (list, tuple)):
        return all(_payload_can_use_pickle_wire_format(item) for item in payload)
    if isinstance(payload, (set, frozenset)):
        return all(_payload_can_use_pickle_wire_format(item) for item in payload)
    if isinstance(payload, dict):
        return all(
            _payload_can_use_pickle_wire_format(key) and _payload_can_use_pickle_wire_format(value)
            for key, value in payload.items()
        )
    return False


def _serialize_object_to_uint8(payload: Any) -> torch.Tensor:
    if _payload_can_use_pickle_wire_format(payload):
        serialized = _PICKLE_WIRE_FORMAT_MAGIC + pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized = _TORCH_WIRE_FORMAT_MAGIC + buffer.getvalue()
    view = memoryview(bytearray(serialized))
    return torch.frombuffer(view, dtype=torch.uint8).clone()


def _deserialize_object_from_uint8(payload: torch.Tensor) -> Any:
    if payload.numel() == 0:
        return None
    raw_payload = payload.cpu().numpy().tobytes()
    if raw_payload.startswith(_PICKLE_WIRE_FORMAT_MAGIC):
        return pickle.loads(raw_payload[len(_PICKLE_WIRE_FORMAT_MAGIC) :])
    if raw_payload.startswith(_TORCH_WIRE_FORMAT_MAGIC):
        buffer = io.BytesIO(raw_payload[len(_TORCH_WIRE_FORMAT_MAGIC) :])
        return torch.load(buffer, map_location="cpu")

    # Backward compatibility for payloads produced before the wire-format header existed.
    buffer = io.BytesIO(raw_payload)
    return torch.load(buffer, map_location="cpu")


def send_object_cpu(payload: Any, dst: int, label: str | None = None) -> None:
    message = f"send {label or 'payload'} to dst={dst}"
    start = time.perf_counter()
    serialized = _serialize_object_to_uint8(payload)
    size = torch.tensor([serialized.numel()], dtype=torch.int64)
    startup_log(
        "object-send",
        f"sending {label or 'payload'} to dst={dst} bytes={int(size.item())}",
    )
    try:
        dist.send(size, dst=dst)
        if serialized.numel() > 0:
            dist.send(serialized.contiguous(), dst=dst)
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("object-send", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="object-send",
            operation="send",
            label=label,
            peer=dst,
            object_bytes=int(size.item()),
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("object-send", message, elapsed)
    record_transport_profile_event(
        channel="object-send",
        operation="send",
        label=label,
        peer=dst,
        object_bytes=int(size.item()),
        elapsed_seconds=elapsed,
    )
    startup_log(
        "object-send",
        f"sent {label or 'payload'} to dst={dst} bytes={int(size.item())}",
    )


def recv_object_cpu(src: int, label: str | None = None) -> Any:
    message = f"recv {label or 'payload'} from src={src}"
    start = time.perf_counter()
    startup_log("object-recv", f"waiting {label or 'payload'} from src={src}")
    size = torch.empty(1, dtype=torch.int64)
    try:
        dist.recv(size, src=src)
        payload = torch.empty(int(size.item()), dtype=torch.uint8)
        if payload.numel() > 0:
            dist.recv(payload, src=src)
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("object-recv", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="object-recv",
            operation="recv",
            label=label,
            peer=src,
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("object-recv", message, elapsed)
    record_transport_profile_event(
        channel="object-recv",
        operation="recv",
        label=label,
        peer=src,
        object_bytes=int(size.item()),
        elapsed_seconds=elapsed,
    )
    startup_log(
        "object-recv",
        f"received {label or 'payload'} from src={src} bytes={int(size.item())}",
    )
    return _deserialize_object_from_uint8(payload)


def send_tensor_payload_cpu(
    payload: dict[str, torch.Tensor | None] | None,
    dst: int,
    *,
    label: str | None = None,
    comm_dtype: torch.dtype | None = None,
):
    from .transport import send_payload

    message = f"send {label or 'payload'} to dst={dst}"
    start = time.perf_counter()
    startup_log(
        "tensor-send",
        f"sending {label or 'payload'} to dst={dst} tensors={0 if payload is None else len(payload)}",
    )
    try:
        summary = send_payload(payload, dst=dst, comm_dtype=comm_dtype)
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("tensor-send", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="tensor-send",
            operation="send",
            label=label,
            peer=dst,
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("tensor-send", message, elapsed)
    record_transport_profile_event(
        channel="tensor-send",
        operation="send",
        label=label,
        peer=dst,
        summary=summary,
        elapsed_seconds=elapsed,
    )
    startup_log(
        "tensor-send",
        f"sent {label or 'payload'} to dst={dst} tensors={summary.num_tensors}",
    )
    return summary


def recv_tensor_payload_cpu(
    src: int,
    *,
    label: str | None = None,
    target_dtypes: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor | None] | None:
    from .transport import recv_payload

    message = f"recv {label or 'payload'} from src={src}"
    start = time.perf_counter()
    startup_log("tensor-recv", f"waiting {label or 'payload'} from src={src}")
    try:
        payload = recv_payload(
            src=src,
            device=torch.device("cpu"),
            target_dtypes=target_dtypes,
        )
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("tensor-recv", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="tensor-recv",
            operation="recv",
            label=label,
            peer=src,
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("tensor-recv", message, elapsed)
    from .schema import PayloadSummary

    record_transport_profile_event(
        channel="tensor-recv",
        operation="recv",
        label=label,
        peer=src,
        summary=PayloadSummary.from_payload(payload),
        elapsed_seconds=elapsed,
    )
    startup_log(
        "tensor-recv",
        f"received {label or 'payload'} from src={src} tensors={0 if payload is None else len(payload)}",
    )
    return payload


def broadcast_tensor_payload_cpu(
    payload: dict[str, torch.Tensor | None] | None,
    *,
    src: int,
    group=None,
    label: str | None = None,
    target_dtypes: dict[str, torch.dtype] | None = None,
    comm_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | None] | None:
    from .transport import broadcast_payload

    message = f"broadcast {label or 'payload'} src={src}"
    start = time.perf_counter()
    startup_log(
        "tensor-bcast",
        f"broadcasting {label or 'payload'} src={src} local_has_payload={payload is not None}",
    )
    try:
        restored = broadcast_payload(
            payload,
            src=src,
            device=torch.device("cpu"),
            group=group,
            target_dtypes=target_dtypes,
            comm_dtype=comm_dtype,
        )
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("tensor-bcast", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="tensor-bcast",
            operation="broadcast",
            label=label,
            peer=src,
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("tensor-bcast", message, elapsed)
    from .schema import PayloadSummary

    record_transport_profile_event(
        channel="tensor-bcast",
        operation="broadcast",
        label=label,
        peer=src,
        summary=PayloadSummary.from_payload(restored),
        elapsed_seconds=elapsed,
    )
    startup_log(
        "tensor-bcast",
        f"broadcast done for {label or 'payload'} src={src} tensors={0 if restored is None else len(restored)}",
    )
    return restored


def broadcast_object_cpu(payload: Any | None, src: int, group=None, label: str | None = None) -> Any:
    message = f"broadcast {label or 'payload'} src={src}"
    start = time.perf_counter()
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
    try:
        dist.broadcast(size, src=src, group=group)

        if serialized is None:
            serialized = torch.empty(int(size.item()), dtype=torch.uint8)
        elif serialized.numel() != int(size.item()):
            raise RuntimeError(
                f"broadcast_object_cpu size mismatch: local={serialized.numel()} expected={int(size.item())}"
            )

        if serialized.numel() > 0:
            dist.broadcast(serialized, src=src, group=group)
    except Exception:
        elapsed = time.perf_counter() - start
        _record_startup_timing("object-bcast", message, elapsed, status="fail")
        record_transport_profile_event(
            channel="object-bcast",
            operation="broadcast",
            label=label,
            peer=src,
            elapsed_seconds=elapsed,
            status="fail",
        )
        raise
    elapsed = time.perf_counter() - start
    _record_startup_timing("object-bcast", message, elapsed)
    record_transport_profile_event(
        channel="object-bcast",
        operation="broadcast",
        label=label,
        peer=src,
        object_bytes=int(size.item()),
        elapsed_seconds=elapsed,
    )
    startup_log(
        "object-bcast",
        f"broadcast done for {label or 'payload'} src={src} bytes={int(size.item())}",
    )
    return _deserialize_object_from_uint8(serialized)
