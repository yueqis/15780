import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for diag(A) @ B optimization
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute diag(A) @ B directly without constructing the full diagonal matrix
__global__ void diag_matmul_kernel(const float* A, const float* B, float* out, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        out[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = B.size(1);

    auto out = torch::zeros_like(B);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    diag_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M);

    return out;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_op = diag_matmul

    def forward(self, A, B):
        return self.cuda_op.diag_matmul_cuda(A, B)
