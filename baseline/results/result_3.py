import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for batched matrix multiplication (optimized custom BMM)
custom_bmm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for batched matrix multiplication
__global__ void custom_bmm_kernel(const float* A, const float* B, float* C,
                                  int batch_size, int m, int k, int n) {
    int bid = blockIdx.z;  // Batch index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (bid >= batch_size || row >= m || col >= n)
        return;

    // Compute offset for current batch
    const float* A_batch = A + bid * m * k;
    const float* B_batch = B + bid * k * n;
    float* C_batch = C + bid * m * n;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += A_batch[row * k + i] * B_batch[i * n + col];
    }
    C_batch[row * n + col] = sum;
}

torch::Tensor custom_bmm_cuda(torch::Tensor A, torch::Tensor B, int batch_size, int m, int k, int n) {
    auto options = A.options();
    torch::Tensor C = torch::zeros({batch_size, m, n}, options);

    dim3 threads(8, 8);
    dim3 blocks((m + 7) / 8, (n + 7) / 8, batch_size);

    custom_bmm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                          C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}
"""

custom_bmm_cpp_source = """
torch::Tensor custom_bmm_cuda(torch::Tensor A, torch::Tensor B, int batch_size, int m, int k, int n);
"""

# Compile the inline CUDA code
custom_bmm = load_inline(
    name="custom_bmm",
    cpp_sources=custom_bmm_cpp_source,
    cuda_sources=custom_bmm_source,
    functions=["custom_bmm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_bmm = custom_bmm
        # Store shape parameters for forward pass
        self.batch_size = 128
        self.m = 128
        self.k = 256
        self.n = 512

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication using a custom CUDA kernel.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return self.custom_bmm.custom_bmm_cuda(
            A, B, self.batch_size, self.m, self.k, self.n
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
