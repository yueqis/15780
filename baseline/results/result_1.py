import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple CUDA kernel for matrix multiplication C = A * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);

    auto C = torch::zeros({N, N}, A.options());

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    matmul_kernel<<<grid_size, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_op = load_inline(
    name="matmul_cuda",
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
        self.matmul_op = matmul_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return self.matmul_op.matmul_cuda(A, B)
