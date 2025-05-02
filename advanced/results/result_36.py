import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for RMS Normalization
rms_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(const float* x, float* y, float eps, int num_features, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Compute mean of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < num_features; ++i) {
            float val = x[idx * num_features + i];
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / num_features;
        float inv_rms = rsqrtf(mean_sq + eps);

        // Normalize each feature
        for (int i = 0; i < num_features; ++i) {
            y[idx * num_features + i] = x[idx * num_features + i] * inv_rms;
        }
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps, int num_features) {
    int batch_size = x.size(0);
    int remaining_dims = x.numel() / (batch_size * num_features);
    auto size = batch_size * remaining_dims;

    auto out = torch::zeros_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    rms_norm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        eps, 
        num_features, 
        size
    );

    return out;
}
"""

# C++ declaration
rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps, int num_features);
"""

# Compile the inline CUDA code
rms_norm_extension = load_inline(
    name="rms_norm_extension",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_cuda_source,
    functions=["rms_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions after batch and features to treat them as "elements per batch"
        batch_size = x.size(0)
        original_shape = x.shape
        x = x.view(batch_size, self.num_features, -1)
        x = x.transpose(1, 2).contiguous()  # Shape: [batch_size, H*W, num_features]

        # Apply custom CUDA RMS Norm
        x = self.rms_norm.rms_norm_cuda(x, self.eps, self.num_features)

        # Restore original shape
        x = x.transpose(1, 2).contiguous()
        return x.view(original_shape)


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
