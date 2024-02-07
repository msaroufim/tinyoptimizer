# Borrowed from https://raw.githubusercontent.com/jllllll/exllama/master/setup.py
import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension
import platform
import os
import subprocess
import torch


import os

def set_ld_library_path(new_path):
    try:
        original_ld_path = os.environ['LD_LIBRARY_PATH']
    except KeyError:
        original_ld_path = ""

    # Append the new path
    os.environ['LD_LIBRARY_PATH'] = original_ld_path + ':' + new_path

# Should make this more generic
set_ld_library_path('/home/ubuntu/pytorch/torch/lib/libc10.so')

def get_cuda_version(cuda_home=os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))):
    if cuda_home == '' or not os.path.exists(os.path.join(cuda_home,"bin","nvcc.exe" if platform.system() == "Windows" else "nvcc")):
        return ''
    version_str = subprocess.check_output([os.path.join(cuda_home,"bin","nvcc"),"--version"]).decode('utf-8')
    idx = version_str.find("release")
    return version_str[idx+len("release "):idx+len("release ")+4]
    
CUDA_VERSION = "".join(get_cuda_version().split(".")) if not os.environ.get('ROCM_VERSION', False) else False
ROCM_VERSION = os.environ.get('ROCM_VERSION', False) if torch.version.hip else False

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],
}
if torch.version.hip:
    extra_compile_args["nvcc"].append("-U__HIP_NO_HALF_CONVERSIONS__")

version = "0.0.18" + (f"+cu{CUDA_VERSION}" if CUDA_VERSION else f"+rocm{ROCM_VERSION}" if ROCM_VERSION else "")
setup(
    name="square",
    version=version,
    install_requires=[
        "torch",
    ],
    packages=["inlined_kernel"],
    py_modules=["square"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "square",
            [
                "inlined_kernel/cuda.cu",
                "inlined_kernel/main.cpp",
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cublas"] if platform.system() == "Windows" else [],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)