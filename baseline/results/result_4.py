import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvec_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication (A: MxK, B: Kx1, out: Mx1)
__global__ void matvec_kernel(const float* A, const float* B, float* out, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k];
        }
        out[row] = sum;
    }
}

// CUDA C++ wrapper function
torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B, int M, int K) {
    auto out = torch::zeros({M, 1}, A.options());

    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;

    matvec_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, K);

    return out;
}
"""

matvec_cpp_source = """
torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B, int M, int K);
"""

# Compile the inline CUDA code
matvec_op = load_inline(
    name="matvec",
    cpp_sources=matvec_cpp_source,
    cuda_sources=matvec_cuda_source,
    functions=["matvec_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for matrix-vector multiplication.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec = matvec_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M = A.size(0)
        K = A.size(1)
        return self.matvec.matvec_cuda(A, B, M, K)


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
