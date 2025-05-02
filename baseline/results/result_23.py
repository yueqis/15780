import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for online softmax (single-pass, numerically stable)
softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = gridDim.x;
    int col = tid;

    // Pointers to the current row
    float* row_input = input + row * cols;
    float* row_output = output + row * cols;

    // Find max element in the row for numerical stability
    float max_val = -FLT_MAX;
    while (col < cols) {
        max_val = fmaxf(row_input[col], max_val);
        col += blockDim.x;
    }

    // Broadcast max_val across threads using shuffle or shared memory
    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    // Compute sum of exponentials
    float sum_exp = 0.0f;
    col = threadIdx.x;
    while (col < cols) {
        float exp_val = expf(row_input[col] - max_val);
        sum_exp += exp_val;
        col += blockDim.x;
    }

    // Reduce sum across threads
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, s);
    }

    // Normalize and write outputs
    col = threadIdx.x;
    while (col < cols) {
        row_output[col] = expf(row_input[col] - max_val) / sum_exp;
        col += blockDim.x;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::zeros_like(input);

    dim3 blocks(rows);
    dim3 threads(1024);

    softmax_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor input);"

# Compile inline CUDA code
softmax_op = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_op = softmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_op.softmax_cuda(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

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
