import torch
from torch.utils.cpp_extension import load

# set cpp_source string by reading cpu_adam_impl.cpp file
cpp_source = ''
with open('cpu_adam_impl.cpp', 'r') as f:
    cpp_source = f.read()

cpu_adam = load(
    name='cpu_adam',
    sources=['cpu_adam_impl.cpp'],
    extra_cflags=['-I/home/ubuntu/tinyoptimizer/cpu_optimizer', '-std=c++2a'],

    verbose=True,
    build_directory='./tmp'
)

print(cpu_adam)