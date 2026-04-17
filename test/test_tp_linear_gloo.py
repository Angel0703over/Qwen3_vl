import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def all_gather_cat_cpu(local_tensor: torch.Tensor, world_size: int, target_device: torch.device) -> torch.Tensor:
    local_cpu = local_tensor.detach().to("cpu")
    gather_list = [torch.empty_like(local_cpu) for _ in range(world_size)]
    dist.all_gather(gather_list, local_cpu)
    gathered = torch.cat(gather_list, dim=-1)
    return gathered.to(target_device)


def all_reduce_sum_cpu(local_tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    reduced = local_tensor.detach().to("cpu")
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced.to(target_device)


def build_reference_tensors(
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
):
    torch.manual_seed(2026)
    x = torch.randn(batch_size, seq_len, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device)
    bias = torch.randn(out_features, dtype=dtype, device=device)
    return x, weight, bias


def run_column_parallel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
    world_size: int,
):
    out_features, in_features = weight.shape
    if out_features % world_size != 0:
        raise ValueError(f"out_features={out_features} 不能被 world_size={world_size} 整除。")

    shard_out = out_features // world_size
    start = rank * shard_out
    end = (rank + 1) * shard_out

    weight_shard = weight[start:end, :].contiguous()
    bias_shard = bias[start:end].contiguous()

    local_output = F.linear(x, weight_shard, bias_shard)
    gathered_output = all_gather_cat_cpu(local_output, world_size, x.device)
    reference_output = F.linear(x, weight, bias)

    max_diff = (gathered_output - reference_output).abs().max().item()
    mean_diff = (gathered_output - reference_output).abs().mean().item()

    return {
        "local_output_shape": tuple(local_output.shape),
        "gathered_output_shape": tuple(gathered_output.shape),
        "reference_output_shape": tuple(reference_output.shape),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def run_row_parallel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
    world_size: int,
):
    out_features, in_features = weight.shape
    if in_features % world_size != 0:
        raise ValueError(f"in_features={in_features} 不能被 world_size={world_size} 整除。")

    shard_in = in_features // world_size
    start = rank * shard_in
    end = (rank + 1) * shard_in

    x_shard = x[..., start:end].contiguous()
    weight_shard = weight[:, start:end].contiguous()

    local_partial = F.linear(x_shard, weight_shard, bias=None)
    reduced_output = all_reduce_sum_cpu(local_partial, x.device)
    if rank == 0:
        reduced_output = reduced_output + bias
    reduced_output_cpu = reduced_output.detach().to("cpu")
    dist.broadcast(reduced_output_cpu, src=0)
    reduced_output = reduced_output_cpu.to(x.device)

    reference_output = F.linear(x, weight, bias)
    max_diff = (reduced_output - reference_output).abs().max().item()
    mean_diff = (reduced_output - reference_output).abs().mean().item()

    return {
        "x_shard_shape": tuple(x_shard.shape),
        "local_partial_shape": tuple(local_partial.shape),
        "reduced_output_shape": tuple(reduced_output.shape),
        "reference_output_shape": tuple(reference_output.shape),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="最小手工 TP Linear 原型，通信层使用 gloo。")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=12)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = get_device(args.device)
    rank, world_size = init_dist()

    print(f"[config] rank={rank} device={device} dtype={dtype} world_size={world_size}")

    x, weight, bias = build_reference_tensors(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=dtype,
        device=device,
    )

    column_stats = run_column_parallel(x, weight, bias, rank, world_size)
    print(
        f"[column] rank={rank} local_output_shape={column_stats['local_output_shape']} "
        f"gathered_output_shape={column_stats['gathered_output_shape']} "
        f"max_diff={column_stats['max_diff']} mean_diff={column_stats['mean_diff']}"
    )

    row_stats = run_row_parallel(x, weight, bias, rank, world_size)
    print(
        f"[row] rank={rank} x_shard_shape={row_stats['x_shard_shape']} "
        f"local_partial_shape={row_stats['local_partial_shape']} "
        f"reduced_output_shape={row_stats['reduced_output_shape']} "
        f"max_diff={row_stats['max_diff']} mean_diff={row_stats['mean_diff']}"
    )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[done] rank={rank}")


if __name__ == "__main__":
    main()
