import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Group Normalization
group_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    float eps) {
    
    int channels_per_group = num_channels / num_groups;
    int hw = height * width;
    int total_elements = batch_size * num_channels * hw;
    int group_id = blockIdx.x;
    int channel_base = group_id * channels_per_group;

    // Shared memory for mean and variance computation
    extern __shared__ float shared_mem[];
    float* mean = shared_mem;
    float* inv_var = shared_mem + num_groups;

    // Compute mean
    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_per_group; ++c) {
            for (int i = 0; i < hw; ++i) {
                int idx = b * num_channels * hw + (channel_base + c) * hw + i;
                sum += input[idx];
            }
        }
    }
    mean[group_id] = sum / (batch_size * channels_per_group * hw);

    // Compute variance
    float var_sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_per_group; ++c) {
            for (int i = 0; i < hw; ++i) {
                int idx = b * num_channels * hw + (channel_base + c) * hw + i;
                float diff = input[idx] - mean[group_id];
                var_sum += diff * diff;
            }
        }
    }
    inv_var[group_id] = rsqrtf(var_sum / (batch_size * channels_per_group * hw) + eps);

    // Normalize
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels_per_group; ++c) {
            float w = (weight != nullptr) ? weight[channel_base + c] : 1.0f;
            float b_val = (bias != nullptr) ? bias[channel_base + c] : 0.0f;
            for (int i = 0; i < hw; ++i) {
                int idx = b * num_channels * hw + (channel_base + c) * hw + i;
                float normalized = (input[idx] - mean[group_id]) * inv_var[group_id];
                output[idx] = w * normalized + b_val;
            }
        }
    }
}

torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps) {
    
    int batch_size = input.size(0);
    int num_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::empty_like(input);

    dim3 block_size(num_groups);
    dim3 grid_size(num_groups);

    int smem_size = 2 * num_groups * sizeof(float);  // For mean and inv_var

    group_norm_kernel<<<grid_size, block_size, smem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        eps);

    return output;
}
"""

group_norm_cpp_source = """
torch::Tensor group_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps);
"""

# Compile the inline CUDA code
group_norm_op = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_cuda_source,
    functions=["group_norm_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.eps = 1e-5
        self.register_parameter("weight", nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_features)))
        self.group_norm_op = group_norm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm_op.group_norm_cuda(
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
