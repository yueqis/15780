```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed matmul (C = A^T * B^T)
matmul_transposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute C[i][j] = sum_{k} A[k][i] * B[j][k]
__global__ void matmul_transposed_kernel(
    const float* A, const float* B, float* C,
    int M, int K, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + i] * B[j * K + k];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor matmul_transposed_cuda(
    torch::Tensor A, torch::Tensor B) {
    
    int M = A.size(1);  // Columns of A (after transpose)
    int K = A.size(0);  // Rows of A (after transpose)
    int N = B.size(0);  // Rows of B (after transpose)

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y));

    matmul_transposed_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}
"""

matmul_transposed_cpp_source = """
torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_transposed_op = load_inline(
    name="matmul_transposed",
    cpp_sources=matmul_transposed_cpp_source,
    cuda_sources=matmul_transposed_source,
    functions=["matmul_transposed_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transposed = matmul_transposed_op.matmul_transposed_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_transposed(A, B)
```