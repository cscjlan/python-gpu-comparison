import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import numpy as np


def run_ring(rank: int, world_size: int, topology: np.ndarray):
    loc = np.where(topology == rank)[0][0]
    send_rank = topology[(loc + 1) % len(topology)]
    recv_rank = topology[(loc - 1 + len(topology)) % len(topology)]

    send_data = torch.tensor(rank, device=torch.device("cuda"))
    recv_data = torch.tensor(0, device=torch.device("cuda"))

    print(
        f"Rank {rank} sending to {send_rank}, receiving from {recv_rank}",
        flush=True,
    )

    if loc & 1 == 0:
        recv_req = dist.irecv(tensor=recv_data, src=recv_rank)
        send_req = dist.isend(tensor=send_data, dst=send_rank)
    else:
        send_req = dist.isend(tensor=send_data, dst=send_rank)
        recv_req = dist.irecv(tensor=recv_data, src=recv_rank)

    print(f"Rank {rank} waiting recv_req", flush=True)
    recv_req.wait()
    print(f"Rank {rank} waiting send_req", flush=True)
    send_req.wait()

    print(f"Rank {rank} finished waiting", flush=True)

    assert send_data.item() == rank
    assert recv_data.item() == recv_rank


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            print(f"Group initialized? {dist.is_initialized()}", flush=True)

        topology = np.array([0, 1, 3, 2, 4, 5, 7, 6])
        run_ring(rank, world_size, topology)

    prof.export_chrome_trace("trace_" + str(rank) + ".json")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
