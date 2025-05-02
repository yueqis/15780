import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mul_kernel(const float* A, float s, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] * s;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    int M = A.size(0);
    int N = A.size(1);
    auto C = torch::zeros_like(A);

    dim3 block(32, 32);  // 32x32 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matrix_scalar_mul_kernel<<<grid, block>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), M, N);

    return C;
}
"""

matrix_scalar_mul_cpp_source = (
    "torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);"
)

# Compile the inline CUDA code for matrix-scalar multiplication
matrix_scalar_mul = load_inline(
    name="matrix_scalar_mul",
    cpp_sources=matrix_scalar_mul_cpp_source,
    cuda_sources=matrix_scalar_mul_source,
    functions=["matrix_scalar_mul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A, s)
