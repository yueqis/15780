import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (no transpose version)
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // Thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory for sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Output element
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (K + BLOCK_SIZE - 1)/BLOCK_SIZE; ++t) {
        // Load tile of A into shared memory
        if (row < M && t * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B into shared memory
        if (t * BLOCK_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute dot product of row of As and column of Bs
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write output element
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function to call the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Launch kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                      C.data_ptr<float>(), M, N, K);

    return C;
}
"""

# C++ declaration for the matmul kernel
matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_op = load_inline(
    name="matmul_op",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for matrix multiplication.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom CUDA kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose B inside the forward method before passing to CUDA kernel
        B_T = B.T
        return self.matmul_op.matmul_cuda(A, B_T)
