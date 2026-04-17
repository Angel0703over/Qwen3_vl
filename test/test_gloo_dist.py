import os
import socket

import torch
import torch.distributed as dist


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def main():
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

    print(f"[post-init] rank={rank} backend={dist.get_backend()}")

    local_value = torch.tensor([float(rank)], dtype=torch.float32)
    dist.all_reduce(local_value, op=dist.ReduceOp.SUM)
    print(f"[all_reduce] rank={rank} value={local_value.item()}")

    broadcast_value = torch.tensor([-1.0], dtype=torch.float32)
    if rank == 0:
        broadcast_value.fill_(123.0)
    dist.broadcast(broadcast_value, src=0)
    print(f"[broadcast] rank={rank} value={broadcast_value.item()}")

    if world_size >= 2:
        if rank == 0:
            send_tensor = torch.tensor([7.0], dtype=torch.float32)
            dist.send(send_tensor, dst=1)
            print(f"[send] rank=0 sent={send_tensor.item()} to rank=1")
        elif rank == 1:
            recv_tensor = torch.empty(1, dtype=torch.float32)
            dist.recv(recv_tensor, src=0)
            print(f"[recv] rank=1 received={recv_tensor.item()} from rank=0")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[done] rank={rank}")


if __name__ == "__main__":
    main()
