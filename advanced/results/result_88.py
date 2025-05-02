```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation function
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu(float x) {
    float sqrt2pi = 0.7978845608f; // sqrt(2.0f / pi)
    float beta = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(sqrt2pi * (x + beta * x * x * x)));
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu(input[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for GELU activation
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu_op

    def forward(self, x):
        return self.gelu.gelu_cuda(x)

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim).cuda()]

def get_init_inputs():
    return []
```