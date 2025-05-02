import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rmsnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* x, float* out, int num_features, int size, float eps) {
    const int N = num_features;
    const int rest_size = size / N;

    int feature_idx = blockIdx.x;
    int rest_idx = blockIdx.y;

    int x_start = feature_idx + rest_idx * N;
    
    // Compute mean squared value for each sample
    float ms = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = x[i + rest_idx * N];
        ms += val * val;
    }
    ms /= N;
    ms += eps;
    float inv_rms = rsqrtf(ms);

    // Normalize
    for (int i = 0; i < N; i++) {
        out[i + rest_idx * N] = x[i + rest_idx * N] * inv_rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    auto size = x.numel();
    auto shape = x.sizes();
    auto out = torch::empty_like(x);
    
    int batch_dim = shape[0];
    int num_features = shape[1];
    dim3 blocks(num_features, batch_dim * (shape.size() > 2 ? shape[2] : 1) * (shape.size() > 3 ? shape[3] : 1));

    rmsnorm_kernel<<<blocks, 1>>>(x.data_ptr<float>(), out.data_ptr<float>(), num_features, size, eps);
    
    return out;
}
"""

rmsnorm_cpp_source = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

# Compile the inline CUDA code
rmsnorm_op = load_inline(
    name="rmsnorm",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_cuda_source,
    functions=["rmsnorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized RMSNorm layer using a custom CUDA kernel.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm_cuda = rmsnorm_op.rmsnorm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply custom CUDA RMS Normalization
        return self.rmsnorm_cuda(x, self.eps)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]


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
