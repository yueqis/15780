import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized matrix multiplication (GEMM)
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized matrix multiplication kernel for large K dimension
__global__ void matmul_kernel(
    const float* A, 
    const float* B, 
    float* C,
    int M, 
    int N, 
    int K) 
{
    // 2D grid of threads, each thread computes one element of output
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(16, 16);  // 16x16 thread blocks
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                   B.data_ptr<float>(), 
                                   C.data_ptr<float>(),
                                   M, N, K);
    
    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_op = load_inline(
    name="matmul_custom",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for matrix multiplication.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the same device
        if A.device != B.device:
            B = B.to(A.device)

        # Perform custom matrix multiplication
        return self.matmul_op.matmul_cuda(A, B)
