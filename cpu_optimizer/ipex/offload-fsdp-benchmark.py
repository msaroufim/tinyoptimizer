import time

from typing import Callable, List

import torch

torch.set_printoptions(threshold=10000)

# Llama-7B
SIZES = [torch.Size([32000, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([4096, 4096]), torch.Size([11008, 4096]), torch.Size([4096, 11008]), torch.Size([11008, 4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([4096]), torch.Size([32000, 4096])]
WORLD_SIZE = 8


def benchmark_time(
    benchmark_fn: Callable,
    *benchmark_fn_args,
    **benchmark_fn_kwargs,
) -> int:
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # torch.cuda.synchronize()
    # To test GPU time ignoring CPU-boundedness, add a sleep before recording
    from torch.testing._internal.common_utils import get_cycles_per_ms
    # torch.cuda._sleep(int(25 * get_cycles_per_ms()))
    # start.record()

    MEASURE_ITERS = 10
    for _ in range(MEASURE_ITERS):
        benchmark_fn(*benchmark_fn_args, **benchmark_fn_kwargs)

    # end.record()
    # torch.cuda.synchronize()
    # tot_time = start.elapsed_time(end)  # ms
    # iter_time = tot_time / MEASURE_ITERS

    # torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(MEASURE_ITERS):
        benchmark_fn(*benchmark_fn_args, **benchmark_fn_kwargs)
    end_time = time.time()
    cpu_time = (end_time - start_time) / MEASURE_ITERS  # s
    # torch.cuda.synchronize()

    return _, cpu_time * 1e3  # ms


def benchmark_with_profiler(
    benchmark_fn: Callable,
    *benchmark_fn_args,
    **benchmark_fn_kwargs,
) -> None:
    torch._C._profiler._set_cuda_sync_enabled_val(False)
    wait, warmup, active = 1, 1, 5
    num_steps = wait + warmup + active
    rank = 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./")
        if not rank  # only save on rank 0
        else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # incurs an additional overhead; disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models at the moment
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            benchmark_fn(*benchmark_fn_args, **benchmark_fn_kwargs)
            if rank is None or rank == 0:
                prof.step()  # notify the profiler at end of each step

def get_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    non_dim0_size = list(tensor_size[1:])
    if tensor_size[0] < dim0_factor:
        return torch.Size([dim0_factor] + non_dim0_size)
    elif tensor_size[0] % dim0_factor != 0:
        return torch.Size(
            [tensor_size[0] + dim0_factor - (tensor_size[0] % dim0_factor)]
            + non_dim0_size
        )
    else:
        return tensor_size


def shard_params(sizes: List[torch.Size], world_size: int):
    sharded_params = []
    for i, size in enumerate(sizes):
        padded_size = get_padded_size(size, world_size)
        padded_param = torch.empty(padded_size, device="cpu")
        chunks = torch.chunk(padded_param, world_size)
        sharded_param = chunks[0]
        # sharded_param.fill_(i)
        sharded_params.append(sharded_param.view(-1))  # assume precomputed this flattening
    return sharded_params


def fn(SIZES, WORLD_SIZE):
    sharded_params = shard_params(SIZES, WORLD_SIZE)

    padded_sharded_numel = sum(p.numel() for p in sharded_params)
    gb = padded_sharded_numel * 4 / 1e9
    print(f"Number of nn.Parameters: {len(SIZES)} | World size: {WORLD_SIZE} | Sharded Numel: {padded_sharded_numel} ({gb:.3f} GB)")

    optim1 = torch.optim.Adam(sharded_params, lr=1e-2)
    for param in sharded_params:
        param.grad = torch.empty_like(param)

    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.optim._optimizer_utils import optimizer_fusion
    optim2 = torch.optim.Adam(sharded_params, lr=1e-2)
    optim2 = optimizer_fusion(torch.optim.Adam(sharded_params, lr=1e-2), True, "cpu")

    def inner1():
        optim1.step()

    def inner2():
        optim2.step()

    print(benchmark_time(inner1))
    print(benchmark_time(inner2))
    # benchmark_with_profiler(inner1)
    # benchmark_with_profiler(inner2)


fn(SIZES, WORLD_SIZE)
