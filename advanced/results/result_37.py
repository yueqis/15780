import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frobenius_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void frobenius_norm_kernel(const float* x, float* out, int size, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / norm;
    }
}

float compute_frobenius_norm(const float* x, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; ++i) {
        norm += x[i] * x[i];
    }
    return sqrtf(norm);
}

class FrobeniusNormCuda {
public:
    torch::Tensor forward(torch::Tensor x) {
        auto size = x.numel();
        auto out = torch::zeros_like(x);
        auto x_data = x.data_ptr<float>();
        float norm = compute_frobenius_norm(x_data, size);

        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        frobenius_norm_kernel<<<num_blocks, block_size>>>(x_data, out.data_ptr<float>(), size, norm);
        return out;
    }
};
"""

frobenius_norm_cpp_source = """
#include <torch/extension.h>
#include <vector>

class FrobeniusNormCuda {
public:
    torch::Tensor forward(torch::Tensor x);
};
TORCH_LIBRARY(my_ops, m) {
    m.def("frobenius_norm_cuda", []() { return std::make_shared<FrobeniusNormCuda>(); });
}
"""

# Compile the inline CUDA code for Frobenius norm normalization
frobenius_norm_op = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_cuda_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm_op.frobenius_norm_cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.forward(x)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        norm = torch.norm(x, p='fro')
        return x / norm

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
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
