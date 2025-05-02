```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(const float* input, const float* weight, const float* bias, float* output,
                                        int batch_size, int in_channels, int out_channels, int height, int width,
                                        int kernel_h, int kernel_w, int stride_h, int stride_w,
                                        int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {
    // Simplified implementation for grouped transposed convolution
    int n = blockIdx.z;
    int oc = blockIdx.y;
    int ic = blockIdx.x * blockDim.x + threadIdx.x;

    if (ic >= in_channels || ic / (in_channels / groups) != oc / (out_channels / groups)) {
        return;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < height; ++oh) {
            for (int ow = 0; ow < width; ++ow) {
                float sum = 0.0f;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int ih = oh * stride_h - padding_h + kh * dilation_h;
                        int iw = ow * stride_w - padding_w + kw * dilation_w;
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int in_idx = b * in_channels * height * width + ic * height * width + ih * width + iw;
                            int w_idx = oc * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                int out_idx = b * out_channels * height * width + oc * height * width + oh * width + ow;
                output[out_idx] = bias ? sum + bias[oc] : sum;
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int64_t in_channels, int64_t out_channels, int64_t height, int64_t width,
                                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                   int64_t padding_h, int64_t padding_w, int64_t dilation_h, int64_t dilation_w, int64_t groups) {
    auto options = input.options().dtype(torch::kFloat32);
    auto output = torch::zeros({input.size(0), out_channels, height, width}, options);

    int threads_per_block = 256;
    int blocks_per_grid_x = (in_channels + threads_per_block - 1) / threads_per_block;
    dim3 blocks_per_grid(blocks_per_grid_x, out_channels, input.size(0));

    conv_transpose2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        input.size(0), in_channels, out_channels, height, width,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int64_t in_channels, int64_t out_channels, int64_t height, int64_t width,
                                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                   int64_t padding_h, int64_t padding_w, int64_t dilation_h, int64_t dilation_w, int64_t groups);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1),
                 padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        output_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        output_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]

        return conv_transpose2d_op.conv_transpose2d_cuda(
            x, self.weight, self.bias, self.in_channels, self.out_channels,
            output_height, output_width,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )
```