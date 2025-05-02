import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused transpose and matrix multiplication
matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// This kernel assumes A is (K x M), B is (K x N)
// We compute C[i][j] = sum_{p=0 to K-1} A[p][i] * B[p][j]
// which is equivalent to A^T (M x K) multiplied by B (K x N)

__global__ void matmul_transpose_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = A[k * M + row];  // A[k][row], equivalent to A^T[row][k]
            float b = B[k * N + col];  // B[k][col]
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(1);  // Columns of A (which is K x M)
    int K = A.size(0);  // Rows of A
    int N = B.size(1);  // Columns of B (which is K x N)

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    matmul_transpose_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_transpose_cpp_source = """
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_transpose_op = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transpose = matmul_transpose_op

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transpose.matmul_transpose_cuda(A, B)
