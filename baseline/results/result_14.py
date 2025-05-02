import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for upper triangular matrix multiplication
triu_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triu_matmul_kernel(
    const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <= col) {  // Only compute upper triangular part
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // Only use upper triangular parts of A and B
            if (row <= k && k <= col) {
                float a = A[row * N + k];
                float b = B[k * N + col];
                sum += a * b;
            }
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    auto C = torch::zeros({N, N}, A.options());

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    triu_matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

triu_matmul_cpp_source = (
    "torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for the custom operator
triu_matmul_op = load_inline(
    name="triu_matmul",
    cpp_sources=triu_matmul_cpp_source,
    cuda_sources=triu_matmul_source,
    functions=["triu_matmul_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication on upper triangular matrices using a custom CUDA kernel.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = triu_matmul_op

    def forward(self, A, B):
        """
        Performs optimized upper triangular matrix multiplication using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return self.triu_matmul.triu_matmul_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return torch.triu(torch.matmul(A, B))

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []
