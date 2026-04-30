"""Transport handlers and compatibility helpers for multimodal stage handoff."""

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.distributed as dist

from .distributed import record_transport_profile_event
from .schema import PayloadSummary, StageHandoffPayload
from .stage import build_stage_handoff_target_dtypes

TensorPayload = dict[str, torch.Tensor | None]

_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.bfloat16: 3,
    torch.uint8: 4,
    torch.int8: 5,
    torch.int16: 6,
    torch.int32: 7,
    torch.int64: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_SINGLE_TENSOR_KEY = "__tensor__"
_HIDDEN_STATES_KEY = "hidden_states"


@dataclass(slots=True)
class StageHandoffMessage:
    """Structured receive result for the stage-handoff communication channel."""

    handoff: StageHandoffPayload | None
    summary: PayloadSummary


class StageCommunicator:
    """Stage-to-stage communicator for structured runtime handoff payloads."""

    channel_name = "stage_handoff"

    def __init__(self, device: torch.device, comm_dtype: torch.dtype) -> None:
        self.device = device
        self.comm_dtype = comm_dtype

    def build_target_dtypes(self, stage_state: dict[str, Any]) -> dict[str, torch.dtype]:
        return build_stage_handoff_target_dtypes(stage_state)

    def send(self, handoff: StageHandoffPayload | None, dst: int) -> PayloadSummary:
        payload = None if handoff is None else handoff.to_transport_payload()
        start = time.perf_counter()
        try:
            summary = send_payload(payload, dst=dst, comm_dtype=self.comm_dtype)
        except Exception:
            record_transport_profile_event(
                channel=self.channel_name,
                operation="send",
                label="stage_handoff",
                peer=dst,
                elapsed_seconds=time.perf_counter() - start,
                status="fail",
            )
            raise
        record_transport_profile_event(
            channel=self.channel_name,
            operation="send",
            label="stage_handoff",
            peer=dst,
            summary=summary,
            elapsed_seconds=time.perf_counter() - start,
        )
        return summary

    def send_empty(self, dst: int) -> PayloadSummary:
        start = time.perf_counter()
        try:
            summary = send_payload(None, dst=dst, comm_dtype=self.comm_dtype)
        except Exception:
            record_transport_profile_event(
                channel=self.channel_name,
                operation="send_empty",
                label="stage_handoff",
                peer=dst,
                elapsed_seconds=time.perf_counter() - start,
                status="fail",
            )
            raise
        record_transport_profile_event(
            channel=self.channel_name,
            operation="send_empty",
            label="stage_handoff",
            peer=dst,
            summary=summary,
            elapsed_seconds=time.perf_counter() - start,
        )
        return summary

    def recv(
        self,
        src: int,
        stage_state: dict[str, Any] | None = None,
    ) -> StageHandoffMessage:
        if stage_state is None:
            raise ValueError("recv 需要 stage_state。")
        start = time.perf_counter()
        try:
            payload = recv_payload(
                src=src,
                device=self.device,
                target_dtypes=self.build_target_dtypes(stage_state),
            )
        except Exception:
            record_transport_profile_event(
                channel=self.channel_name,
                operation="recv",
                label="stage_handoff",
                peer=src,
                elapsed_seconds=time.perf_counter() - start,
                status="fail",
            )
            raise
        summary = PayloadSummary.from_payload(payload)
        record_transport_profile_event(
            channel=self.channel_name,
            operation="recv",
            label="stage_handoff",
            peer=src,
            summary=summary,
            elapsed_seconds=time.perf_counter() - start,
        )
        return StageHandoffMessage(
            handoff=StageHandoffPayload.from_transport_payload(payload),
            summary=summary,
        )


def _send_scalar(value: int, dst: int) -> None:
    dist.send(torch.tensor([value], dtype=torch.int64), dst=dst)


def _recv_scalar(src: int) -> int:
    value = torch.empty(1, dtype=torch.int64)
    dist.recv(value, src=src)
    return int(value.item())


def _send_shape(shape: torch.Size | tuple[int, ...], dst: int) -> None:
    _send_scalar(len(shape), dst)
    if shape:
        dist.send(torch.tensor(list(shape), dtype=torch.int64), dst=dst)


def _recv_shape(src: int) -> tuple[int, ...]:
    ndim = _recv_scalar(src)
    if ndim == 0:
        return ()

    shape_tensor = torch.empty(ndim, dtype=torch.int64)
    dist.recv(shape_tensor, src=src)
    return tuple(int(v) for v in shape_tensor.tolist())


def _send_string(value: str, dst: int) -> None:
    encoded = value.encode("utf-8")
    _send_scalar(len(encoded), dst)
    if encoded:
        dist.send(torch.tensor(list(encoded), dtype=torch.uint8), dst=dst)


def _recv_string(src: int) -> str:
    num_bytes = _recv_scalar(src)
    if num_bytes == 0:
        return ""

    payload = torch.empty(num_bytes, dtype=torch.uint8)
    dist.recv(payload, src=src)
    return bytes(payload.tolist()).decode("utf-8")


def _broadcast_scalar(
    value: int | None,
    *,
    src: int,
    group=None,
) -> int:
    tensor = (
        torch.tensor([int(value)], dtype=torch.int64)
        if value is not None
        else torch.empty(1, dtype=torch.int64)
    )
    dist.broadcast(tensor, src=src, group=group)
    return int(tensor.item())


def _broadcast_shape(
    shape: torch.Size | tuple[int, ...] | None,
    *,
    src: int,
    group=None,
) -> tuple[int, ...]:
    ndim = _broadcast_scalar(None if shape is None else len(shape), src=src, group=group)
    if ndim == 0:
        return ()
    if shape is not None:
        shape_tensor = torch.tensor(list(shape), dtype=torch.int64)
    else:
        shape_tensor = torch.empty(ndim, dtype=torch.int64)
    dist.broadcast(shape_tensor, src=src, group=group)
    return tuple(int(v) for v in shape_tensor.tolist())


def _broadcast_string(
    value: str | None,
    *,
    src: int,
    group=None,
) -> str:
    encoded = None if value is None else value.encode("utf-8")
    num_bytes = _broadcast_scalar(None if encoded is None else len(encoded), src=src, group=group)
    if num_bytes == 0:
        return ""
    if encoded is not None:
        payload = torch.tensor(list(encoded), dtype=torch.uint8)
    else:
        payload = torch.empty(num_bytes, dtype=torch.uint8)
    dist.broadcast(payload, src=src, group=group)
    return bytes(payload.tolist()).decode("utf-8")


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype not in _DTYPE_TO_CODE:
        raise ValueError(f"transport 暂不支持 dtype={dtype!r}。")
    return _DTYPE_TO_CODE[dtype]


def _code_to_dtype(code: int) -> torch.dtype:
    if code not in _CODE_TO_DTYPE:
        raise ValueError(f"transport 收到了不支持的 dtype code={code}。")
    return _CODE_TO_DTYPE[code]


def _resolve_wire_dtype(
    tensor: torch.Tensor,
    comm_dtype: torch.dtype | None,
) -> torch.dtype:
    if tensor.is_floating_point() and comm_dtype is not None:
        return comm_dtype
    return tensor.dtype


def send_payload(
    payload: TensorPayload | None,
    dst: int,
    comm_dtype: torch.dtype | None,
) -> PayloadSummary:
    # payload 为 None 表示“空通信占位”，用于和异构 PP 的 dummy send/recv 对齐。
    if payload is None:
        _send_scalar(1, dst)
        return PayloadSummary.empty()

    _send_scalar(0, dst)
    items = list(payload.items())
    _send_scalar(len(items), dst)

    tensor_shapes = {}
    tensor_dtypes = {}
    tensor_numels = {}
    tensor_bytes = {}
    for name, tensor in items:
        _send_string(name, dst)
        _send_scalar(0 if tensor is None else 1, dst)
        if tensor is None:
            tensor_shapes[name] = None
            tensor_dtypes[name] = None
            tensor_numels[name] = 0
            tensor_bytes[name] = 0
            continue

        wire_dtype = _resolve_wire_dtype(tensor, comm_dtype)
        _send_scalar(_dtype_to_code(tensor.dtype), dst)
        _send_scalar(_dtype_to_code(wire_dtype), dst)
        payload_cpu = tensor.detach().to("cpu", dtype=wire_dtype).contiguous()
        _send_shape(payload_cpu.shape, dst)
        dist.send(payload_cpu, dst=dst)
        tensor_shapes[name] = tuple(payload_cpu.shape)
        tensor_dtypes[name] = str(wire_dtype)
        tensor_numels[name] = int(payload_cpu.numel())
        tensor_bytes[name] = int(payload_cpu.numel() * payload_cpu.element_size())

    return PayloadSummary(
        is_empty=False,
        num_tensors=len(items),
        payload_keys=[name for name, _ in items],
        tensor_shapes=tensor_shapes,
        tensor_dtypes=tensor_dtypes,
        tensor_numels=tensor_numels,
        tensor_bytes=tensor_bytes,
        total_tensor_bytes=sum(tensor_bytes.values()),
    )


def recv_payload(
    src: int,
    device: torch.device,
    target_dtypes: dict[str, torch.dtype] | None = None,
) -> TensorPayload | None:
    is_empty = bool(_recv_scalar(src))
    if is_empty:
        return None

    num_tensors = _recv_scalar(src)
    payload = {}
    for _ in range(num_tensors):
        name = _recv_string(src)
        has_tensor = bool(_recv_scalar(src))
        if not has_tensor:
            payload[name] = None
            continue

        source_dtype = _code_to_dtype(_recv_scalar(src))
        wire_dtype = _code_to_dtype(_recv_scalar(src))
        shape = _recv_shape(src)
        tensor_cpu = torch.empty(shape, dtype=wire_dtype)
        dist.recv(tensor_cpu, src=src)

        target_dtype = source_dtype
        if target_dtypes is not None and name in target_dtypes:
            target_dtype = target_dtypes[name]
        payload[name] = tensor_cpu.to(device=device, dtype=target_dtype)
    return payload


def broadcast_payload(
    payload: TensorPayload | None,
    *,
    src: int,
    device: torch.device,
    group=None,
    target_dtypes: dict[str, torch.dtype] | None = None,
    comm_dtype: torch.dtype | None = None,
) -> TensorPayload | None:
    local_is_src = dist.get_rank() == src
    is_empty = bool(
        _broadcast_scalar(
            1 if local_is_src and payload is None else (0 if local_is_src else None),
            src=src,
            group=group,
        )
    )
    if is_empty:
        return None

    items = [] if payload is None else list(payload.items())
    num_tensors = _broadcast_scalar(
        len(items) if local_is_src else None,
        src=src,
        group=group,
    )
    restored: TensorPayload = {}
    for index in range(num_tensors):
        item = items[index] if payload is not None else None
        name = _broadcast_string(
            None if item is None else item[0],
            src=src,
            group=group,
        )
        tensor = None if item is None else item[1]
        has_tensor = bool(
            _broadcast_scalar(
                None if item is None else (0 if tensor is None else 1),
                src=src,
                group=group,
            )
        )
        if not has_tensor:
            restored[name] = None
            continue

        source_dtype = _code_to_dtype(
            _broadcast_scalar(
                None if tensor is None else _dtype_to_code(tensor.dtype),
                src=src,
                group=group,
            )
        )
        wire_dtype = _code_to_dtype(
            _broadcast_scalar(
                None
                if tensor is None
                else _dtype_to_code(_resolve_wire_dtype(tensor, comm_dtype)),
                src=src,
                group=group,
            )
        )
        if tensor is not None:
            tensor_cpu = tensor.detach().to("cpu", dtype=wire_dtype).contiguous()
            expected_shape = _broadcast_shape(tensor_cpu.shape, src=src, group=group)
            if tuple(tensor_cpu.shape) != tuple(expected_shape):
                raise RuntimeError(
                    f"broadcast_payload shape mismatch for {name}: "
                    f"local={tuple(tensor_cpu.shape)} expected={tuple(expected_shape)}"
                )
        else:
            expected_shape = _broadcast_shape(None, src=src, group=group)
            tensor_cpu = torch.empty(expected_shape, dtype=wire_dtype)
        if tensor_cpu.numel() > 0:
            dist.broadcast(tensor_cpu, src=src, group=group)

        target_dtype = source_dtype
        if target_dtypes is not None and name in target_dtypes:
            target_dtype = target_dtypes[name]
        restored[name] = tensor_cpu.to(device=device, dtype=target_dtype)
    return restored


def send_tensor(
    tensor: torch.Tensor | None,
    dst: int,
    comm_dtype: torch.dtype,
) -> tuple[int, ...] | None:
    payload = None if tensor is None else {_SINGLE_TENSOR_KEY: tensor}
    summary = send_payload(payload, dst=dst, comm_dtype=comm_dtype)
    return summary.tensor_shapes.get(_SINGLE_TENSOR_KEY)


def recv_tensor(
    src: int,
    device: torch.device,
    target_dtype: torch.dtype,
    comm_dtype: torch.dtype,
    allow_empty: bool = False,
) -> torch.Tensor | None:
    del comm_dtype
    payload = recv_payload(
        src=src,
        device=device,
        target_dtypes={_SINGLE_TENSOR_KEY: target_dtype},
    )
    if payload is None:
        if allow_empty:
            return None
        raise ValueError("recv_tensor 收到了空 payload。")

    tensor = payload.get(_SINGLE_TENSOR_KEY)
    if tensor is None and not allow_empty:
        raise ValueError("recv_tensor 收到的 payload 里没有有效 tensor。")
    return tensor


def send_hidden_states(
    hidden_states: torch.Tensor | None,
    dst: int,
    comm_dtype: torch.dtype,
) -> tuple[int, ...] | None:
    payload = None if hidden_states is None else {_HIDDEN_STATES_KEY: hidden_states}
    summary = send_payload(payload, dst=dst, comm_dtype=comm_dtype)
    return summary.tensor_shapes.get(_HIDDEN_STATES_KEY)


def recv_hidden_states(
    src: int,
    device: torch.device,
    hidden_dtype: torch.dtype,
    comm_dtype: torch.dtype,
    allow_empty: bool = False,
) -> torch.Tensor | None:
    del comm_dtype
    payload = recv_payload(
        src=src,
        device=device,
        target_dtypes={_HIDDEN_STATES_KEY: hidden_dtype},
    )
    if payload is None:
        if allow_empty:
            return None
        raise ValueError("recv_hidden_states 收到了空 payload。")

    hidden_states = payload.get(_HIDDEN_STATES_KEY)
    if hidden_states is None and not allow_empty:
        raise ValueError("recv_hidden_states 收到的 payload 里没有有效 hidden_states。")
    return hidden_states


__all__ = [
    "TensorPayload",
    "StageHandoffMessage",
    "StageCommunicator",
    "PayloadSummary",
    "broadcast_payload",
    "send_payload",
    "recv_payload",
    "send_tensor",
    "recv_tensor",
    "send_hidden_states",
    "recv_hidden_states",
]
