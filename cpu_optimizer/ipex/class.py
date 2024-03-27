import torch

import intel_extension_for_pytorch as ipex

import torch
import torch.nn as nn
import torch.optim as optim
import time
import intel_extension_for_pytorch as ipex

from intel_extension_for_pytorch.optim._functional import adamstep, adamw_step



class FusedCPUAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, maximize=False, foreach=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=foreach)
        super(FusedCPUAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Call the original AdamW step function
        return adamw_step(
            self=self, 
            closure=closure
        )


## The test case
# Define a simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10000, 10000)

    def forward(self, x):
        return self.fc(x)

# Create the model and input data
model = Net()
input_data = torch.randn(10000, 10000)

# Benchmark the existing Adam optimizer
optimizer = optim.Adam(model.parameters())

start_time = time.time()
for _ in range(10000):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Existing Adam optimizer time: {end_time - start_time:.4f} seconds")

# Benchmark the fused Adam optimizer
optimizer = FusedCPUAdam(model.parameters(), fused=True)

start_time = time.time()
for _ in range(10000):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Fused Adam optimizer time: {end_time - start_time:.4f} seconds")