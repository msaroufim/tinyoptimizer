
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


cpp_fused_add_addcdiv_addcmul_div_lerp_mul_neg_pow_reciprocal_rsub_sqrt_0 = async_compile.cpp('''
#include "/tmp/torchinductor_ubuntu/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr15,
                       float* out_ptr17)
{
    #pragma omp parallel num_threads(24)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(268435456L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp10 = in_ptr2[static_cast<long>(x0)];
                auto tmp18 = in_ptr3[static_cast<long>(0L)];
                auto tmp37 = in_ptr4[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp3 = static_cast<float>(0.10000000149011612);
                auto tmp4 = decltype(tmp3)(tmp3 * tmp2);
                auto tmp5 = static_cast<bool>(0);
                auto tmp6 = tmp5 ? tmp0 : tmp1;
                auto tmp7 = decltype(tmp4)(tmp4 + tmp6);
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp11 = static_cast<float>(0.999);
                auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                auto tmp13 = static_cast<float>(0.0010000000000000009);
                auto tmp14 = decltype(tmp0)(tmp0 * tmp13);
                auto tmp15 = decltype(tmp14)(tmp14 * tmp0);
                auto tmp16 = decltype(tmp12)(tmp12 + tmp15);
                auto tmp17 = std::sqrt(tmp16);
                auto tmp19 = decltype(tmp18)(tmp18 + tmp8);
                auto tmp20 = std::pow(tmp11, tmp19);
                auto tmp21 = decltype(tmp8)(tmp8 - tmp20);
                auto tmp22 = std::sqrt(tmp21);
                auto tmp23 = static_cast<float>(0.9);
                auto tmp24 = std::pow(tmp23, tmp19);
                auto tmp25 = decltype(tmp8)(tmp8 - tmp24);
                auto tmp26 = 1 / tmp25;
                auto tmp27 = static_cast<float>(0.001);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp29 = decltype(tmp28)(-tmp28);
                auto tmp30 = decltype(tmp22)(tmp22 * tmp29);
                auto tmp31 = tmp17 / tmp30;
                auto tmp32 = 1 / tmp29;
                auto tmp33 = static_cast<float>(1e-08);
                auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                auto tmp35 = decltype(tmp31)(tmp31 + tmp34);
                auto tmp36 = tmp9 / tmp35;
                auto tmp38 = decltype(tmp37)(tmp37 + tmp36);
                out_ptr4[static_cast<long>(x0)] = tmp38;
                out_ptr5[static_cast<long>(x0)] = tmp7;
                out_ptr6[static_cast<long>(x0)] = tmp16;
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp10 = in_ptr7[static_cast<long>(x0)];
                    auto tmp18 = in_ptr8[static_cast<long>(0L)];
                    auto tmp37 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp3 = static_cast<float>(0.10000000149011612);
                    auto tmp4 = decltype(tmp3)(tmp3 * tmp2);
                    auto tmp5 = static_cast<bool>(0);
                    auto tmp6 = tmp5 ? tmp0 : tmp1;
                    auto tmp7 = decltype(tmp4)(tmp4 + tmp6);
                    auto tmp8 = static_cast<float>(1.0);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = static_cast<float>(0.999);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(0.0010000000000000009);
                    auto tmp14 = decltype(tmp0)(tmp0 * tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp0);
                    auto tmp16 = decltype(tmp12)(tmp12 + tmp15);
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp19 = decltype(tmp18)(tmp18 + tmp8);
                    auto tmp20 = std::pow(tmp11, tmp19);
                    auto tmp21 = decltype(tmp8)(tmp8 - tmp20);
                    auto tmp22 = std::sqrt(tmp21);
                    auto tmp23 = static_cast<float>(0.9);
                    auto tmp24 = std::pow(tmp23, tmp19);
                    auto tmp25 = decltype(tmp8)(tmp8 - tmp24);
                    auto tmp26 = 1 / tmp25;
                    auto tmp27 = static_cast<float>(0.001);
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp29 = decltype(tmp28)(-tmp28);
                    auto tmp30 = decltype(tmp22)(tmp22 * tmp29);
                    auto tmp31 = tmp17 / tmp30;
                    auto tmp32 = 1 / tmp29;
                    auto tmp33 = static_cast<float>(1e-08);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp35 = decltype(tmp31)(tmp31 + tmp34);
                    auto tmp36 = tmp9 / tmp35;
                    auto tmp38 = decltype(tmp37)(tmp37 + tmp36);
                    out_ptr11[static_cast<long>(x0)] = tmp38;
                    out_ptr12[static_cast<long>(x0)] = tmp7;
                    out_ptr13[static_cast<long>(x0)] = tmp16;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr3[static_cast<long>(0L)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr15[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr8[static_cast<long>(0L)];
                auto tmp1 = static_cast<float>(1.0);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr17[static_cast<long>(0L)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16384, 16384), (16384, 1))
    assert_size_stride(arg1_1, (16384, ), (1, ))
    assert_size_stride(arg2_1, (16384, 16384), (16384, 1))
    assert_size_stride(arg3_1, (16384, ), (1, ))
    assert_size_stride(arg4_1, (16384, 16384), (16384, 1))
    assert_size_stride(arg5_1, (16384, ), (1, ))
    assert_size_stride(arg6_1, (), ())
    assert_size_stride(arg7_1, (), ())
    assert_size_stride(arg8_1, (16384, 16384), (16384, 1))
    assert_size_stride(arg9_1, (16384, ), (1, ))
    cpp_fused_add_addcdiv_addcmul_div_lerp_mul_neg_pow_reciprocal_rsub_sqrt_0(c_void_p(arg8_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
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
    arg0_1 = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16384, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
