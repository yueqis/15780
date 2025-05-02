import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for max reduction along a given dimension
max_reduction_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Kernel to compute max along dim=1 (reduction over features)
__global__ void max_reduce_dim1_kernel(const float* input, float* output, int batch_size, int dim1, int dim2) {
    extern __shared__ float s_data[];

    int b = blockIdx.y; // Batch index
    int row = b * blockDim.y + threadIdx.y; // Which row in the matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= dim2) return;

    int idx = row * dim1 * dim2 + threadIdx.x * dim2 + col;
    if (idx >= b * dim1 * dim2 + dim1 * dim2) return;

    s_data[threadIdx.x] = -FLT_MAX;
    for (int i = threadIdx.y; i < dim1; i += blockDim.y) {
        int local_idx = row * dim1 * dim2 + i * dim2 + col;
        s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], input[local_idx]);
    }
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] = fmaxf(s_data[threadIdx.x], s_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[b * dim2 + col] = s_data[0];
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor input, int dim) {
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);

    auto output = torch::zeros({batch_size, dim2}, input.options());

    dim3 threads(16, 16); // Thread block of 16x16
    dim3 blocks((dim2 + threads.x - 1) / threads.x, batch_size);

    int smem_size = threads.x * sizeof(float);

    max_reduce_dim1_kernel<<<blocks, threads, smem_size>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                                         batch_size, dim1, dim2);

    cudaDeviceSynchronize();
    return output;
}
"""

# C++ interface for the CUDA kernel
cpp_interface = """
torch::Tensor max_reduce_cuda(torch::Tensor input, int dim);
"""

# Compile the inline CUDA code
max_reduce_op = load_inline(
    name="max_reduce",
    cpp_sources=cpp_interface,
    cuda_sources=max_reduction_kernel,
    functions=["max_reduce_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduce = max_reduce_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduce.max_reduce_cuda(x, self.dim)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return torch.max(x, dim=self.dim)[0]

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension


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
