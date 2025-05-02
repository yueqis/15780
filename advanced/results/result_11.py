import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for 4D tensor-matrix multiplication
einsum_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel(const float* A, const float* B, float* C, int b, int i, int j, int l, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b * i * j * k) {
        // Compute the indices for output tensor C[b, i, j, k]
        int stride_k = 1;
        int stride_j = k;
        int stride_i = j * k;
        int stride_b = i * j * k;

        int batch = idx / stride_b;
        int rem = idx % stride_b;
        int ii = rem / stride_j;
        rem = rem % stride_j;
        int jj = rem / stride_k;
        int kk = rem % stride_k;

        // Perform contraction over dimension l
        float sum = 0.0f;
        for (int ll = 0; ll < l; ++ll) {
            float a_val = A[batch * i * j * l + ii * j * l + jj * l + ll];
            float b_val = B[ll * k + kk];
            sum += a_val * b_val;
        }
        C[idx] = sum;
    }
}

torch::Tensor einsum_cuda(torch::Tensor A, torch::Tensor B, int b, int i, int j, int l, int k) {
    auto C = torch::zeros({b, i, j, k}, A.options());

    const int threads_per_block = 256;
    const int num_blocks = (b * i * j * k + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "einsum_cuda", ([&] {
        using scalar_t = float;
        einsum_kernel<<<num_blocks, threads_per_block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            b, i, j, l, k);
    }));

    return C;
}
"""

einsum_cpp_source = "torch::Tensor einsum_cuda(torch::Tensor A, torch::Tensor B, int b, int i, int j, int l, int k);"

# Compile the inline CUDA code
einsum_op = load_inline(
    name="einsum",
    cpp_sources=einsum_cpp_source,
    cuda_sources=einsum_kernel_source,
    functions=["einsum_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.einsum_op = einsum_op

    def forward(self, A, B):
        b, i, j, l = A.shape
        l, k = B.shape
        return self.einsum_op.einsum_cuda(A, B, b, i, j, l, k)
