```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh activation
hardtanh_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val < -1.0f ? -1.0f : (val > 1.0f ? 1.0f : val);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    hardtanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor input);"

# Compile the inline CUDA code for HardTanh
hardtanh_op = load_inline(
    name="hardtanh",
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_kernel_source,
    functions=["hardtanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh_op = hardtanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_op.hardtanh_cuda(x)
```