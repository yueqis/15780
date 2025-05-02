import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for asymmetric 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int depth,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_height,
    int out_width,
    int out_depth
) {
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x / out_width;
    int ow = blockIdx.x % out_width;

    int g = oc / (out_channels / groups);

    float acc = 0.0f;

    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = ih_start + kh * dilation;
            int iw = iw_start + kw * dilation;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                for (int ic = g * (in_channels / groups); ic < (g + 1) * (in_channels / groups); ++ic) {
                    for (int d = 0; d < depth; ++d) {
                        int idx_in = b * in_channels * height * width * depth + ic * height * width * depth + ih * width * depth + iw * depth + d;
                        int idx_weight = oc * in_channels / groups * kernel_size * kernel_size * depth + (ic - g * (in_channels / groups)) * kernel_size * kernel_size * depth + kh * kernel_size * depth + kw * depth + d;
                        acc += input[idx_in] * weight[idx_weight];
                    }
                }
            }
        }
    }

    int od = oc * out_height * out_width * out_depth + oh * out_width * out_depth + ow * out_depth;
    output[b * out_channels * out_height * out_width * out_depth + od] = acc;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto depth = input.size(4);

    auto out_channels = weight.size(0);
    auto out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_depth = depth;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width, out_depth}, input.options());

    dim3 blockSize(1);
    dim3 gridSize(out_width * out_height, out_channels, batch_size);

    conv3d_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        depth,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        out_height,
        out_width,
        out_depth
    );

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int stride, int padding, int dilation, int groups);
"""

# Compile the inline CUDA code for the custom 3D convolution
conv3d_op = load_inline(
    name="conv3d_op",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized 3D convolution using a custom CUDA kernel.
    Performs an asymmetric convolution where the kernel is square in spatial dimensions but not depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels // groups, kernel_size, kernel_size, 1
            )
        )
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv3d_op = conv3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d_op.conv3d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
