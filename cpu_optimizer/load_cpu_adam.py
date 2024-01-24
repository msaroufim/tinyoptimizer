import torch
from torch.utils.cpp_extension import load

# set cpp_source string by reading cpu_adam_impl.cpp file
cpp_source = ''
with open('cpu_adam_impl.cpp', 'r') as f:
    cpp_source = f.read()

cpu_adam = load(
    name='cpu_adam',
    sources=['cpu_adam_impl.cpp'],
    
    # This includes SIMD library and also cpu adam dot h
    # Make sure to git clone deepspeed in a top level directory
    extra_cflags=['-I/home/ubuntu/DeepSpeed/csrc/includes', '-std=c++17'], 

    verbose=True,
    build_directory='./tmp'
)

print(cpu_adam)

# This needs to be tested better
# import torch
# import torch.optim as optim
# import numpy as np

# def test_adam_optimizer():
#     # Initialize two identical sets of parameters
#     param1 = torch.randn(100, requires_grad=True)
#     param2 = param1.clone().detach().requires_grad_(True)

#     # Initialize your custom Adam optimizer and PyTorch's Adam optimizer
#     optimizer1 = optim.Adam([param1])
#     optimizer2 = cpu_adam.create_adam_optimizer(0, 0.001, 0.9, 0.999, 1e-8, 0, False, False)

#     for _ in range(100):
#         # Compute a dummy loss
#         loss1 = (param1 ** 2).sum()
#         loss2 = (param2 ** 2).sum()

#         # Backpropagate the gradients
#         loss1.backward()
#         loss2.backward()

#         # Perform a step of optimization
#         optimizer1.step()
#         optimizer2.ds_adam_step(0, _, 0.001, 0.9, 0.999, 1e-8, 0, True, param2, param2.grad, torch.zeros_like(param2), torch.zeros_like(param2))

#         # Zero the gradients
#         optimizer1.zero_grad()
#         param2.grad.zero_()

#     # Check if the parameters are close to each other
#     # assert torch.allclose(param1, param2, atol=1), "The parameters optimized by the two optimizers are not close to each other"

#     # Destroy the custom Adam optimizer
#     optimizer2.destroy_adam_optimizer(0)

# test_adam_optimizer()
