```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
mat_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mat_scalar_mul_kernel(const float* A, float s, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor mat_scalar_mul_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto C = torch::zeros_like(A);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mat_scalar_mul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), size);

    return C;
}
"""

mat_scalar_mul_cpp_source = (
    "torch::Tensor mat_scalar_mul_cuda(torch::Tensor A, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
mat_scalar_mul = load_inline(
    name="mat_scalar_mul",
    cpp_sources=mat_scalar_mul_cpp_source,
    cuda_sources=mat_scalar_mul_source,
    functions=["mat_scalar_mul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for matrix-scalar multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_scalar_op = mat_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a custom CUDA kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return self.matmul_scalar_op.mat_scalar_mul_cuda(A, s)
```