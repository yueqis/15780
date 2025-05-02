import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for sub-matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the row and column index of the element in C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over all blocks of A and B
    for (int p = 0; p < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++p) {
        // Load the sub-matrix of A into shared memory
        if (row < M && (p * BLOCK_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + p * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load the sub-matrix of B into shared memory
        if ((p * BLOCK_SIZE + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(p * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads();

        // Multiply the sub-matrices
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize before loading next sub-matrices
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get the dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({M, N}, options);

    // Set block size
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid size
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_custom = load_inline(
    name="matmul_custom",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for matrix multiplication
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_custom

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using custom CUDA kernel.
        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).
        Returns:
            C: Output tensor with shape (M, N).
        """
        return self.matmul_op.matmul_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return torch.matmul(A, B)

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
