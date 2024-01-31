
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_ubuntu/gk/cgk5o3jgkkyyd5xap73vsouvd3wtwrhkoo3qhgi6lh3iotza3xns.py
# Source Nodes: [], Original ATen: []

triton_for_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import foreach
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
@foreach(num_warps=8, triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}, inductor_meta={'kernel_name': 'triton_for_fused_0'})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
    elif xpid >= 1 and xpid < 2:
        xpid_offset = xpid - 1
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp4 = tl.load(in_ptr1 + (0))
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
        tmp6 = 1.0
        tmp7 = tmp5 + tmp6
        tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)
    else:
        pass
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ubuntu/bz/cbz7dalc2xtnv6atar4lkzzvtzjknwqrhhpf3edmwphn6eynvwlq.py
# Source Nodes: [], Original ATen: []

triton_for_fused_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import foreach
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
@foreach(num_warps=8, triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}, inductor_meta={'kernel_name': 'triton_for_fused_1'})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, out_ptr2, out_ptr3, out_ptr5, out_ptr6, out_ptr7):
    xpid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    if xpid >= 0 and xpid < 1:
        xpid_offset = xpid - 0
        xnumel = 10
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp3 = tl.load(in_ptr1 + (x0), xmask)
        tmp8 = tl.load(in_ptr2 + (x0), xmask)
        tmp13 = tl.load(in_ptr3 + (x0), xmask)
        tmp15 = tl.load(in_ptr4 + (0))
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
        tmp1 = 0.999
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 * tmp3
        tmp5 = 0.0010000000000000009
        tmp6 = tmp4 * tmp5
        tmp7 = tmp2 + tmp6
        tmp9 = tmp3 - tmp8
        tmp10 = 0.09999999999999998
        tmp11 = tmp9 * tmp10
        tmp12 = tmp8 + tmp11
        tmp14 = tl.sqrt(tmp7)
        tmp17 = tl.math.pow(tmp1, tmp16)
        tmp18 = 1.0
        tmp19 = tmp17 - tmp18
        tmp20 = -tmp19
        tmp21 = tl.sqrt(tmp20)
        tmp22 = tmp14 / tmp21
        tmp23 = 1e-08
        tmp24 = tmp22 + tmp23
        tmp25 = 0.9
        tmp26 = tl.math.pow(tmp25, tmp16)
        tmp27 = tmp26 - tmp18
        tmp28 = 0.001
        tmp29 = tmp27 / tmp28
        tmp30 = 1 / tmp29
        tmp31 = tmp24 / tmp30
        tmp32 = tmp12 / tmp31
        tmp33 = tmp13 + tmp32
        tl.store(out_ptr1 + (x0), tmp12, xmask)
        tl.store(out_ptr2 + (x0), tmp33, xmask)
        tl.store(out_ptr3 + (x0), tmp7, xmask)
    elif xpid >= 1 and xpid < 2:
        xpid_offset = xpid - 1
        xnumel = 1
        xoffset = xpid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp34 = tl.load(in_ptr5 + (0))
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
        tmp38 = tl.load(in_ptr6 + (0))
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
        tmp44 = tl.load(in_ptr7 + (0))
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
        tmp50 = tl.load(in_ptr8 + (0))
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
        tmp53 = tl.load(in_ptr9 + (0))
        tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
        tmp36 = 0.999
        tmp37 = tmp35 * tmp36
        tmp40 = tmp39 * tmp39
        tmp41 = 0.0010000000000000009
        tmp42 = tmp40 * tmp41
        tmp43 = tmp37 + tmp42
        tmp46 = tmp39 - tmp45
        tmp47 = 0.09999999999999998
        tmp48 = tmp46 * tmp47
        tmp49 = tmp45 + tmp48
        tmp52 = tl.sqrt(tmp43)
        tmp55 = tl.math.pow(tmp36, tmp54)
        tmp56 = 1.0
        tmp57 = tmp55 - tmp56
        tmp58 = -tmp57
        tmp59 = tl.sqrt(tmp58)
        tmp60 = tmp52 / tmp59
        tmp61 = 1e-08
        tmp62 = tmp60 + tmp61
        tmp63 = 0.9
        tmp64 = tl.math.pow(tmp63, tmp54)
        tmp65 = tmp64 - tmp56
        tmp66 = 0.001
        tmp67 = tmp65 / tmp66
        tmp68 = 1 / tmp67
        tmp69 = tmp62 / tmp68
        tmp70 = tmp49 / tmp69
        tmp71 = tmp51 + tmp70
        tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp49, None)
        tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp71, None)
        tl.store(out_ptr7 + (tl.full([XBLOCK], 0, tl.int32)), tmp43, None)
    else:
        pass
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 10), (10, 1))
    assert_size_stride(arg1_1, (1, ), (1, ))
    assert_size_stride(arg2_1, (1, 10), (10, 1))
    assert_size_stride(arg3_1, (1, ), (1, ))
    assert_size_stride(arg4_1, (1, 10), (10, 1))
    assert_size_stride(arg5_1, (1, ), (1, ))
    assert_size_stride(arg6_1, (), ())
    assert_size_stride(arg7_1, (), ())
    assert_size_stride(arg8_1, (1, 10), (10, 1))
    assert_size_stride(arg9_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_for_fused_0.run(arg6_1, arg7_1, arg6_1, arg7_1, grid=((2, 1, 1)), stream=stream0)
        # Source Nodes: [], Original ATen: []
        triton_for_fused_1.run(arg4_1, arg8_1, arg2_1, arg0_1, arg6_1, arg5_1, arg9_1, arg3_1, arg1_1, arg7_1, arg2_1, arg0_1, arg4_1, arg3_1, arg1_1, arg5_1, grid=((2, 1, 1)), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
        return ()


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
