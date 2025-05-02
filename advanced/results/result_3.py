import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
bmm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void bmm_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int bid = blockIdx.z; // Batch index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        float a = A[bid * m * k + row * k + i];
        float b = B[bid * k * n + i * n + col];
        sum += a * b;
    }

    C[bid * m * n + row * n + col] = sum;
}

torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y, batch_size);

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                                batch_size, m, k, n);

    return C;
}
"""

bmm_cpp_source = "torch::Tensor bmm_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for batched matrix multiplication
bmm_module = load_inline(
    name="bmm_module",
    cpp_sources=bmm_cpp_source,
    cuda_sources=bmm_cuda_source,
    functions=["bmm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bmm_op = bmm_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bmm_op.bmm_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
