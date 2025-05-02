import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for lower triangular matrix multiplication
lt_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lt_matmul_kernel(
    const float* A, const float* B, float* C,
    int N, int strideA, int strideB, int strideC) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N || col > row)
        return;

    float sum = 0.0f;
    for (int k = 0; k <= min(row, col); ++k) {
        float a = A[row * strideA + k];
        float b = B[k * strideB + col];
        sum += a * b;
    }
    C[row * strideC + col] = sum;
}

torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    lt_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, A.stride(0), B.stride(0), C.stride(0));

    return C;
}
"""

lt_matmul_cpp_source = "torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
lt_matmul_op = load_inline(
    name="lt_matmul",
    cpp_sources=lt_matmul_cpp_source,
    cuda_sources=lt_matmul_cuda_source,
    functions=["lt_matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lt_matmul = lt_matmul_op

    def forward(self, A, B):
        return self.lt_matmul.lt_matmul_cuda(A, B)
