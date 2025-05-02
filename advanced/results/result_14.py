import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul + triu fusion
matmul_triu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_triu_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <= col && row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else if (row > col) {
        // Lower triangle remains zero
        C[row * N + col] = 0.0f;
    }
}

torch::Tensor matmul_triu_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);

    matmul_triu_kernel<<<gridSize, blockSize>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_triu_cpp_source = (
    "torch::Tensor matmul_triu_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matmul_triu_op = load_inline(
    name="matmul_triu",
    cpp_sources=matmul_triu_cpp_source,
    cuda_sources=matmul_triu_source,
    functions=["matmul_triu_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_triu = matmul_triu_op

    def forward(self, A, B):
        return self.matmul_triu.matmul_triu_cuda(A, B)
