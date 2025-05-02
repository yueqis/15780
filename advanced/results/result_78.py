```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D transposed convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_height, 
    int input_width,
    int output_height, 
    int output_width,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ops = batch_size * out_channels * output_height * output_width;
    
    if (idx >= total_ops) return;

    int b = idx / (out_channels * output_height * output_width);
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    int c_in_start = 0;
    int c_in_end = in_channels;

    float sum = 0.0f;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh;
                int w_in = w_out * stride_w - padding_w + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    float input_val = input[b * in_channels * input_height * input_width + c_in * input_height * input_width + h_in * input_width + w_in];
                    float weight_val = weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[b * out_channels * output_height * output_width + c_out * output_height * output_width + h_out * output_width + w_out] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int weight_channels = weight.size(1); // Must be equal to in_channels
    int weight_kernel_h = weight.size(2);
    int weight_kernel_w = weight.size(3);

    // Validate that the kernel size matches the provided parameters
    TORCH_CHECK(weight_kernel_h == kernel_h && weight_kernel_w == kernel_w, "Weight kernel size does not match");

    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output_height * output_width + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        out_channels, 
        input_height, 
        input_width,
        output_height, 
        output_width,
        kernel_h, 
        kernel_w,
        stride_h, 
        stride_w,
        padding_h, 
        padding_w);

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, "
    "int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w);"
)

# Compile the inline CUDA code for the custom transposed convolution
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.conv_transpose2d_cuda = conv_transpose2d_op.conv_transpose2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d_cuda(
            x, 
            self.weight, 
            self.bias if self.bias is not None else torch.empty(0),
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1]
        )
```