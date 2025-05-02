import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for lower triangular matrix multiplication
lt_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void lt_matmul_kernel(
    const float* A, const float* B, float* C,
    int N, int strideA, int strideB, int strideC) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N || col > row)
        return;

    float sum = 0.0f;
    for (int k = 0; k <= min(row, col); ++k) {
        float a = A[row * strideA + k];
        float b = B[k * strideB + col];
        sum += a * b;
    }
    C[row * strideC + col] = sum;
}

torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    lt_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, A.stride(0), B.stride(0), C.stride(0));

    return C;
}
"""

lt_matmul_cpp_source = "torch::Tensor lt_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
lt_matmul_op = load_inline(
    name="lt_matmul",
    cpp_sources=lt_matmul_cpp_source,
    cuda_sources=lt_matmul_cuda_source,
    functions=["lt_matmul_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lt_matmul = lt_matmul_op

    def forward(self, A, B):
        return self.lt_matmul.lt_matmul_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return torch.tril(torch.matmul(A, B))

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed


# correctness
assert len(get_init_inputs()) == 0
inputs = get_inputs()
inputs = [x.cuda() for x in inputs]
model_result = Model()(*inputs)
model_new_result = ModelNew()(*inputs)
assert torch.allclose(model_result.detach().cpu(), model_new_result.detach().cpu(), rtol=1e-02, atol=1e-03)

# profiling
import triton.profiler as proton
from triton.testing import do_bench

def bench(func, warmup=0, repeat=10, proton_name="kernel"):
    with proton.scope(proton_name, metrics={}):
        ms = do_bench(func, warmup=warmup, rep=repeat)
    return ms

func_model = lambda: Model()(*inputs)
func_model_new = lambda: ModelNew()(*inputs)
model_ms = bench(func_model, warmup=0, repeat=10, proton_name="Model")
model_new_ms = bench(func_model_new, warmup=0, repeat=10, proton_name="ModelNew")
print(f"Model: {model_ms} ms")
print(f"ModelNew: {model_new_ms} ms")
