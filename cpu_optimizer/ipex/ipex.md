IPEX instructions https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/getting_started.html

Seem to indicate that you need to compile the model like so but I want is just to dispatch a specific kernel

```python
import torch
############## import ipex ###############
import intel_extension_for_pytorch as ipex
##########################################

model = Model()
model.eval()
data = ...

############## TorchDynamo ###############
model = ipex.optimize(model, weights_prepack=False)

model = torch.compile(model, backend="ipex")
with torch.no_grad():
  model(data)
##########################################
```

## Call a kernel directly

```python
optimizer = ipex.optim._lamb.Lamb(model.parameters(), fused=True)
```

This API does not unfortunately exist for ADAM thought but it could and we could also make this a public function