import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto C = torch::zeros({M}, options);

    dim3 blockSize(256);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x);

    matmul_kernel<<<gridSize, blockSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K
    );

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_extension = load_inline(
    name="matmul_extension",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_extension

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure input shapes are correct
        assert A.dim() == 2, "Input A must be 2D"
        assert B.dim() == 2, "Input B must be 2D"
        assert A.size(1) == B.size(0), "Incompatible dimensions for matmul"

        # Perform matrix-vector multiplication using the custom CUDA kernel
        return self.matmul_op.matmul_cuda(A, B).unsqueeze(-1)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)

M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
