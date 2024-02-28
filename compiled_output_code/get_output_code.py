# This has a few issues
# 1. `optimizer.step()` and `optimizer.zero_grad()` are not supported in fullgraph mode
# 2. `nn.MSELoss()` is not supported in fp16 on cpu
# loss.backwards() not supported in fullgraph

import torch
import torch.nn as nn
import torch.optim as optim
import os

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(16384, 16384)

    def forward(self, x):
        return self.fc(x)

# Put these outside of main() otherwise torch.compile() craps out
net = SimpleNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# criterion = nn.MSELoss() # Not supported in fp16 on cpu
criterion = nn.L1Loss()

def main(input):
    # Dummy input and target data
    # input = torch.randn(1, 10) 
    # for _ in range(128):
    target = torch.randn(1, 16384)
    output = net(input)

    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    # Step 7: Single optimizer step
    optimizer.step()


if __name__ == "__main__":
    main = torch.compile(main)
    
    main(torch.randn(1,16384))
