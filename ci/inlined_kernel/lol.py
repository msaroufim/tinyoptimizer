import torch 
torch.set_default_device("cuda")

a = torch.Tensor([1])
print(a.device) # prints CPU

a = torch.tensor([1])
print(a.device) # prints cuda:0

a = torch.randn(0)
print(a.device) # prints cuda:0

a = torch.empty(1)
print(a.device) # prints cuda:0


a = torch.tensor(())
print(a.device) # prints cuda:0