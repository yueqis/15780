```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in,
    int kernel_size, int stride, int padding, int dilation) {
    
    // Compute output dimensions
    int height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height_out * width_out) return;

    int n = idx / (out_channels * height_out * width_out);
    int c_out = (idx / (height_out * width_out)) % out_channels;
    int h_out = (idx / width_out) % height_out;
    int w_out = idx % width_out;

    float acc = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - 2 * padding + dilation * kh;
                int w_in = w_out * stride - 2 * padding + dilation * kw;

                if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    float input_val = input[n * in_channels * height_in * width_in + c_in * height_in * width_in + h_in * width_in + w_in];
                    float weight_val = weight[c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw];
                    acc += input_val * weight_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c_out];
    }

    output[idx] = acc;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation) {

    int batch_size = input.size(0);
    int height_in = input.size(2);
    int width_in = input.size(3);

    // Compute output dimensions
    int height_out = (height_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int width_out = (width_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    int num_threads = 512;
    int num_blocks = (batch_size * out_channels * height_out * width_out + num_threads - 1) / num_threads;

    conv_transpose2d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height_in, width_in, kernel_size, stride, padding, dilation);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for ConvTranspose2d
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        self.conv_transpose2d_cuda = conv_transpose2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract weights and bias
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.use_bias else torch.tensor([]).to(x.device)

        # Call the custom CUDA implementation
        output = self.conv_transpose2d_cuda.conv_transpose2d_cuda(
            x, weight, bias,
            self.in_channels, self.out_channels,
            self.kernel_size, self.stride,
            self.padding, self.dilation
        )
        return output
```