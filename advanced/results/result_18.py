import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (C = A^T * B^T)
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Shared memory for A and B tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output element
    float value = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile of A into shared memory (A is transposed)
        int aRow = by * BLOCK_SIZE + ty;
        int aCol = t * BLOCK_SIZE + tx;
        if (aRow < M && aCol < K) {
            As[ty][tx] = A[aCol * M + aRow];  // Transpose A during load
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory (B is transposed)
        int bRow = t * BLOCK_SIZE + ty;
        int bCol = bx * BLOCK_SIZE + tx;
        if (bRow < K && bCol < N) {
            Bs[ty][tx] = B[bCol * K + bRow];  // Transpose B during load
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure data is loaded
        __syncthreads();

        // Multiply the current tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write output element
    int cRow = by * BLOCK_SIZE + ty;
    int cCol = bx * BLOCK_SIZE + tx;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N) {
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Launch kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int K, int N);
"""

# Compile the inline CUDA code
matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for matrix multiplication (C = A^T * B^T)
    """

    def __init__(self, M, K, N):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op
        self.M = M
        self.K = K
        self.N = N

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom CUDA kernel.
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).
        Returns:
            Output tensor of shape (M, N).
        """
        return self.matmul_op.matmul_cuda(A, B, self.M, self.K, self.N)
