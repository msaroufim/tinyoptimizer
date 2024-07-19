compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    # Compute forward pass
    output = model(input)

event = torch.cuda.Event()
event.record(compute_stream)

with torch.cuda.stream(transfer_stream):
    event.wait()  # Wait for computation to finish
    # Offload activations to CPU
    cpu_activations = output.cpu()

# Later, when activations are needed again:
with torch.cuda.stream(transfer_stream):
    # Bring activations back to GPU
    gpu_activations = cpu_activations.cuda()

with torch.cuda.stream(compute_stream):
    # Use the activations for backward pass
    loss = criterion(gpu_activations, target)
    loss.backward()
