import torch
import torch.nn as nn
import torch.optim as optim
import time
import intel_extension_for_pytorch as ipex


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

# Benchmark the existing Adam optimizer
optimizer = optim.Adam(model.parameters())

start_time = time.time()
for _ in range(1000):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Existing Adam optimizer time: {end_time - start_time:.4f} seconds")

# Benchmark the fused Adam optimizer
optimizer = ipex.optim._lamb.Lamb(model.parameters(), fused=True)

start_time = time.time()
for _ in range(1000):
    optimizer.zero_grad()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()
end_time = time.time()

print(f"Fused Adam optimizer time: {end_time - start_time:.4f} seconds")