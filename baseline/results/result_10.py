import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
matmul_3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_3d_kernel(const float* A, const float* B, float* out, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            for (int l = 0; l < L; l++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[n * M * K + m * K + k] * B[k * L + l];
                }
                out[n * M * L + m * L + l] = sum;
            }
        }
    }
}

torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto out = torch::zeros({N, M, L}, A.options());

    const int block_size = 256;
    const int num_blocks = (N * M * L + block_size - 1) / block_size;

    matmul_3d_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, K, L);

    return out;
}
"""

matmul_3d_cpp_source = "torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_3d_op = load_inline(
    name="matmul_3d",
    cpp_sources=matmul_3d_cpp_source,
    cuda_sources=matmul_3d_cuda_source,
    functions=["matmul_3d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_3d = matmul_3d_op

    def forward(self, A, B):
        return self.matmul_3d.matmul_3d_cuda(A, B)
