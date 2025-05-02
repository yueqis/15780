```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ELU activation
elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, alpha);

    return output;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor input, float alpha);"

# Compile the inline CUDA code for ELU
elu_op = load_inline(
    name="elu_cuda",
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=["elu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_op.elu_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda(x, self.alpha)
```