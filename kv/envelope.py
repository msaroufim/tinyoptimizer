import numpy as np
import matplotlib.pyplot as plt

# Given constants
layers = 32
num_attention_heads = 32
head_dim = 64
num_parameters = 405 * 10**9  # 8 billion parameters
batch_size = 1

# Bytes per element for different dtypes
bytes_per_element = {
    'int8': 1,
    'int4': 0.5,
    'int2': 0.25,
    'fp16': 2
}

# Sequence lengths: multiples of 2 from 128 to 128,000
sequence_lengths = [128 * (2 ** i) for i in range(13)]  # Adjusting range for proper multiples up to 128,000

# Calculate KV cache sizes including model parameters
kv_cache_sizes = {dtype: [] for dtype in bytes_per_element}

for seq_len in sequence_lengths:
    for dtype, byte_size in bytes_per_element.items():
        kv_cache_size = 2 * layers * num_attention_heads * head_dim * byte_size * batch_size * seq_len / (1024**3)
        model_size = num_parameters * byte_size / (1024**3)
        total_size = kv_cache_size + model_size
        kv_cache_sizes[dtype].append(total_size)

# Plotting the results
plt.figure(figsize=(10, 6))
markers = {'int8': 'o', 'int4': 's', 'int2': '^', 'fp16': 'x'}
for dtype, sizes in kv_cache_sizes.items():
    plt.plot(sequence_lengths, sizes, label=dtype, marker=markers[dtype])
plt.xlabel('Sequence Length')
plt.ylabel('Total Size (GB)')
plt.title('Llama 405b Total Size (Model Parameters + KV Cache) vs Sequence Length')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")

# Add dotted lines for thresholds
# thresholds_gb = [24, 40, 80]
thresholds_gb = [80]
for threshold in thresholds_gb:
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)
    plt.text(sequence_lengths[0], threshold, f'{threshold}GB', color='r', va='bottom')

# Manually setting y-ticks to avoid multiples of 10
plt.yticks([80, 100, 200, 500, 1000], ['80GB', '100GB', '200GB', '500GB', '1TB'])

# Format x-axis labels with commas
plt.xticks(sequence_lengths, labels=[f'{seq_len:,}' for seq_len in sequence_lengths], rotation=45)

# Save the plot to disk
plt.savefig('total_size_plot_with_model_parameters.png')

# Show the plot
plt.show()
