
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


cpp_fused_div_mul_0 = async_compile.cpp('''
#include "/tmp/torchinductor_ubuntu/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(268435456L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(0L)];
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(268435456.0);
                auto tmp2 = tmp0 / tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp2);
                auto tmp5 = tmp4 * tmp3;
                tmp5.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_ubuntu/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16384L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16384L*x1)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, sign, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (16384, 16384), (16384, 1))
    assert_size_stride(sign, (16384, 16384), (16384, 1))
    assert_size_stride(tangents_1, (), ())
    buf0 = empty((16384, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_div_mul_0(c_void_p(tangents_1.data_ptr()), c_void_p(sign.data_ptr()), c_void_p(buf0.data_ptr()))
    del sign
    del tangents_1
    buf1 = empty((16384, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf0, (16384, 16384), (1, 16384), 0), primals_3, out=buf1)
    del primals_3
    buf2 = empty((1, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_sum_1(c_void_p(buf0.data_ptr()), c_void_p(buf2.data_ptr()))
    return (reinterpret_tensor(buf1, (16384, 16384), (16384, 1), 0), reinterpret_tensor(buf2, (16384, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    sign = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_3, sign, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
