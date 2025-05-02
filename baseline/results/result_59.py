```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Conv3D with asymmetric kernel (kernel_size x kernel_size x 1)
conv3d_custom_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_custom_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int depth,
    int kernel_size, int stride, int padding, int dilation,
    int groups,
    const float* bias
) {
    int gid = blockIdx.z;
    int bid = blockIdx.y;
    int oid = threadIdx.z;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    int group_size = in_channels / groups;

    // Output dimensions
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Input pointers
    const float* input_group = input + (bid * in_channels + gid * group_size) * height * width * depth;
    const float* weight_group = weight + (gid * out_channels / groups) * group_size * kernel_size * kernel_size;

    float acc = (bias != nullptr) ? bias[gid * out_channels / groups + oid] : 0.0f;

    for (int ic = 0; ic < group_size; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * stride + dilation * kh - padding;
                int iw = ow * stride + dilation * kw - padding;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float in_val = input_group[(ic * height * width + ih * width + iw) * depth];
                    float w_val = weight_group[((oid * group_size + ic) * kernel_size + kh) * kernel_size + kw];
                    acc += in_val * w_val;
                }
            }
        }
    }

    output[(bid * out_channels + gid * out_channels / groups + oid) * out_h * out_w + oh * out_w + ow] = acc;
}

torch::Tensor conv3d_custom_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels, int out_channels,
    int kernel_size, int stride, int padding, int dilation,
    int groups
) {
    auto options = input.options();
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int depth = input.size(4);

    int out_h = (height + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    int out_d = depth;

    torch::Tensor output = torch::zeros({batch_size, out_channels, out_h, out_w, out_d}, options);

    dim3 blocks(groups, batch_size, 1);
    dim3 threads(16, 16, 16); // Adjust based on your GPU architecture

    conv3d_custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width, depth,
        kernel_size, stride, padding, dilation,
        groups,
        bias.data_ptr<float>()
    );

    return output;
}
"""

conv3d_custom_cpp_source = """
torch::Tensor conv3d_custom_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups);
"""

# Compile the inline CUDA code
custom_conv3d_op = load_inline(
    name="custom_conv3d",
    cpp_sources=conv3d_custom_cpp_source,
    cuda_sources=conv3d_custom_cuda_source,
    functions=["conv3d_custom_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Register learnable parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.bias)

        # Register custom CUDA op
        self.custom_conv3d = custom_conv3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_conv3d.conv3d_custom_cuda(
            x, self.weight, self.bias if self.use_bias else torch.tensor([]),
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.groups
        )
```