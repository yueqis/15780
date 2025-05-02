import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (GEMM)
gemm_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // Compute a single element of matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];  // Transpose handled by index swapping
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

gemm_cpp_source = "torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
gemm_module = load_inline(
    name="gemm_module",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_kernel_source,
    functions=["gemm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return gemm_module.gemm_cuda(A, B)
