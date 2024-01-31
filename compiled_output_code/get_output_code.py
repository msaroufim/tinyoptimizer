import torch
import torch.nn as nn
import torch.optim as optim
import os

torch.set_default_device("cuda")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Put these outside of main() otherwise torch.compile() craps out
net = SimpleNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

def main(input):
    # Dummy input and target data
    # input = torch.randn(1, 10) 
    target = torch.randn(1, 1)
    output = net(input)

    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    # Step 7: Single optimizer step
    optimizer.step()


if __name__ == "__main__":
    main = torch.compile(main)
    main(torch.randn(1,10))
    # main()
