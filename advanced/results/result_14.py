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


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return torch.triu(torch.matmul(A, B))

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []


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
