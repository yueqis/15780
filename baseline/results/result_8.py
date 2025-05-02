import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define a custom CUDA kernel for matrix multiplication (GEMM)
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple tiled GEMM kernel
template<int TILE_SIZE>
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE_SIZE) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        if (row < M && (t + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 tile(16, 16);
    dim3 grid((N + tile.x - 1) / tile.x, (M + tile.y - 1) / tile.y);

    matmul_kernel<16><<<grid, tile>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_extension = load_inline(
    name="matmul_extension",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for matrix multiplication.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_extension.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and on the same device
        A = A.contiguous()
        B = B.contiguous()

        # Move inputs to GPU if they aren't already
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()

        return self.matmul_op(A, B)
