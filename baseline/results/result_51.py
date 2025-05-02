import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Argmax along dim=1 (for fixed optimization)
argmax_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, int64_t batch_size, int64_t dim1, int64_t dim2) {
    int64_t b = blockIdx.x;
    int64_t i2 = threadIdx.x;
    int64_t stride = blockDim.x;

    scalar_t max_val = -static_cast<scalar_t>(INFINITY);
    int64_t max_idx = -1;

    for (int64_t i1 = i2; i1 < dim1; i1 += stride) {
        scalar_t val = input[b * dim1 * dim2 + i1 * dim2 + i2];
        if (val > max_val) {
            max_val = val;
            max_idx = i1;
        }
    }

    __shared__ scalar_t shared_max_vals[32];
    __shared__ int64_t shared_max_indices[32];

    int tid = threadIdx.x;
    shared_max_vals[tid] = max_val;
    shared_max_indices[tid] = max_idx;

    __syncthreads();

    // Reduce within warp
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max_vals[tid] < shared_max_vals[tid + s]) {
                shared_max_vals[tid] = shared_max_vals[tid + s];
                shared_max_indices[tid] = shared_max_indices[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * dim2 + i2] = shared_max_indices[0];
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t batch_size, int64_t dim1, int64_t dim2) {
    auto output = torch::empty({batch_size, dim2}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    dim3 blocks(batch_size * dim2);
    dim3 threads(32);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));

    return output;
}
"""

argmax_cpp_binding = """
torch::Tensor argmax_cuda(torch::Tensor input, int64_t batch_size, int64_t dim1, int64_t dim2);
"""

# Compile the inline CUDA code
argmax_op = load_inline(
    name="argmax_cuda",
    cpp_sources=argmax_cpp_binding,
    cuda_sources=argmax_kernel_source,
    functions=["argmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel performing Argmax over a specified dimension.
    Currently supports dim=1 only for this optimized version.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_op.argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension (dim=1 only optimized in CUDA).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        if self.dim != 1:
            # Fallback to PyTorch implementation for other dimensions
            return torch.argmax(x, dim=self.dim)

        batch_size, dim1, dim2 = x.shape
        return self.argmax_cuda(x, batch_size, dim1, dim2)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return torch.argmax(x, dim=self.dim)

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]


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
