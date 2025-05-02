import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for symmetric matrix multiplication
symm_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute C = A * B where A and B are symmetric matrices
__global__ void symm_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a = A[row * N + k];
            float b = B[k * N + col];
            value += a * b;
        }
        C[row * N + col] = value;
    }
}

torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    symm_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

symm_matmul_cpp_source = (
    "torch::Tensor symm_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
symm_matmul = load_inline(
    name="symm_matmul",
    cpp_sources=symm_matmul_cpp_source,
    cuda_sources=symm_matmul_source,
    functions=["symm_matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = symm_matmul

    def forward(self, A, B):
        return self.matmul_op.symm_matmul_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        return torch.matmul(A, B)

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.randn(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.randn(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []
