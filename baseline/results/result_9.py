import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication optimized for tall-and-skinny matrices
matmul_tall_skinny_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int TILE_WIDTH>
__global__ void matmul_tall_skinny_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aRow = row;
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int bRow = t * TILE_WIDTH + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch kernel with TILE_WIDTH = 16
    matmul_tall_skinny_kernel<16><<<gridSize, blockSize>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );

    return C;
}
"""

matmul_tall_skinny_cpp_source = """
torch::Tensor matmul_tall_skinny_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
matmul_tall_skinny_op = load_inline(
    name="matmul_tall_skinny",
    cpp_sources=matmul_tall_skinny_cpp_source,
    cuda_sources=matmul_tall_skinny_source,
    functions=["matmul_tall_skinny_cuda"],
    verbose=False,
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_op = matmul_tall_skinny_op

    def forward(self, A, B):
        # Ensure both tensors are on the same device
        device = A.device
        A = A.to(device)
        B = B.to(device)
        return self.matmul_op.matmul_tall_skinny_cuda(A, B)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
