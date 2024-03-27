import torch

import intel_extension_for_pytorch as ipex

import torch
import torch.nn as nn
import torch.optim as optim
import time
import intel_extension_for_pytorch as ipex
from tqdm import tqdm

from intel_extension_for_pytorch.optim._functional import adam_step, adamw_step

# (base) ubuntu@ip-172-31-48-15:~/tinyoptimizer/cpu_optimizer/ipex$ conda activate fresh
# (fresh) ubuntu@ip-172-31-48-15:~/tinyoptimizer/cpu_optimizer/ipex$ python class.py 
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 304.98it/s]
# Fused Adam optimizer time using ipex_adam_step: 3.2803 seconds
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 362.54it/s]
# Fused Adam optimizer time using ipex.optimize: 2.7588 seconds
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 288.74it/s]
# Existing Adam optimizer time: 3.4637 seconds


class FusedCPUAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, maximize=False, foreach=None, fused=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach)
        super(FusedCPUAdam, self).__init__(params, defaults)
        self.fused = fused
        self.params_attr = {}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Call the original AdamW step function
        return adam_step(
            self=self, 
            closure=closure
        )


## The test case
# Define a simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)



# Create the model and input data
model = Net()
input_data = torch.randn(1000, 1000)


# Benchmark the fused Adam optimizer
optimizer = FusedCPUAdam(model.parameters(), fused=True)

start_time = time.time()
for _ in tqdm(range(1000)):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Fused Adam optimizer time using ipex_adam_step: {end_time - start_time:.4f} seconds")


# This works but unfortunately couples the optimization kernel to the nn module which in the case of distributed 
# will be annoying
_, optimizer = ipex.optimize(model = model, optimizer=torch.optim.Adam(model.parameters()))


# Benchmark the fused Adam optimizer
# optimizer = fused_cpu_adam(model.parameters(), fused=True)

start_time = time.time()
for _ in tqdm(range(1000)):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Fused Adam optimizer time using ipex.optimize: {end_time - start_time:.4f} seconds")


# Benchmark the existing Adam optimizer
optimizer = optim.Adam(model.parameters())

start_time = time.time()
for _ in tqdm(range(1000)):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Existing Adam optimizer time: {end_time - start_time:.4f} seconds")

