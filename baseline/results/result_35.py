import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GroupNorm
group_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int num_features,
    int height,
    int width,
    int group_size,
    float eps) 
{
    int b = blockIdx.x;
    int g = blockIdx.y;
    int fid = threadIdx.x;

    int spatial_size = height * width;
    int feature_per_group = num_features / group_size;

    float mean = 0.0f;
    float var = 0.0f;

    // Compute mean
    for (int f = 0; f < feature_per_group; ++f) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = (b * num_features + g * feature_per_group + f) * spatial_size + h * width + w;
                mean += input[idx];
            }
        }
    }

    mean /= (feature_per_group * spatial_size);

    // Compute variance
    for (int f = 0; f < feature_per_group; ++f) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = (b * num_features + g * feature_per_group + f) * spatial_size + h * width + w;
                float diff = input[idx] - mean;
                var += diff * diff;
            }
        }
    }

    var /= (feature_per_group * spatial_size);
    float inv_std = 1.0f / sqrt(var + eps);

    // Normalize and apply scale & shift
    for (int f = 0; f < feature_per_group; ++f) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = (b * num_features + g * feature_per_group + f) * spatial_size + h * width + w;
                float normalized = (input[idx] - mean) * inv_std;
                if (weight && bias) {
                    int widx = g * feature_per_group + f;
                    output[idx] = normalized * weight[widx] + bias[widx];
                } else {
                    output[idx] = normalized;
                }
            }
        }
    }
}

torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps) {
    int batch_size = input.size(0);
    int num_features = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int group_size = num_groups;
    int feature_per_group = num_features / group_size;

    auto options = input.options();
    torch::Tensor output = torch::empty_like(input, options);

    dim3 blocks(batch_size, group_size, 1);
    dim3 threads(feature_per_group, 1, 1);

    group_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size,
        num_features,
        height,
        width,
        group_size,
        eps
    );

    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);
"""

# Compile the inline CUDA code
group_norm_op = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_cuda_source,
    functions=["group_norm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        self.num_groups = num_groups
        self.eps = self.gn.eps
        self.register_buffer("weight", self.gn.weight)
        self.register_buffer("bias", self.gn.bias)
        self.group_norm_cuda = group_norm_op.group_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm_cuda(
            x, self.weight, self.bias, self.num_groups, self.eps
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)

batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features


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
