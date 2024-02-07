## Tutorial

## Author the kernel

Author the kernel and load it using `from torch.utils.cpp_extension import load_inline` https://github.com/cuda-mode/lectures/blob/main/lecture1/load_inline.py

```python
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)
```

The CUDA files with the right imports will then get written to `./load_inline_cuda` and you can copy paste them into your library

## Package the kernel

See `setup.py` for more info

```bash
pip install .
```

## Run the kernel

Then in a python repl

```python
import torch # otherwise you get a c++ symbol error when you import square
import square
square
dir(square)
```