import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication (optimized for tall/skinny matrices)
matmul_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Thread indices within the thread block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Compute the row and column index of the element this thread computes in C
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    // Iterate over tiles of A and B to compute dot product
    for (int k = 0; k < K; k++) {
        // Load A[row][k] and B[k][col] into shared memory or registers
        float a = A[row * K + k];
        float b = B[k * M + col];

        // Accumulate the product
        sum += a * b;
    }

    // Write the computed value back to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function to call the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K) {
    // Create an output tensor with appropriate size
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({M, N}, options);

    // Define the number of threads per block and blocks per grid
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        return torch::Tensor();
    }

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, int M, int N, int K);
"""

# Compile the inline CUDA code
matmul_extension = load_inline(
    name="matmul_extension",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        # Get matrix dimensions
        M, K = A.shape
        _, N = B.shape

        # Ensure both inputs are on the same device
        assert A.device() == B.device(), "Input tensors must be on the same device"

        # Call the custom CUDA matmul implementation
        return matmul_extension.matmul_cuda(A, B, M, N, K)
