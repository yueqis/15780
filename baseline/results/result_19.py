```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for ReLU activation
relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code
relu_op = load_inline(
    name="relu_custom",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=["relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_op = relu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return self.relu_op.relu_cuda(x)
```