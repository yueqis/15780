import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for 2D transposed convolution
conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int height_in, 
    int width_in,
    int height_out, 
    int width_out,
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation) {

    const int padded_kernel_size = (kernel_size - 1) * dilation + 1;
    
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int w = idx % width_out;
    idx /= width_out;
    int h = idx % height_out;
    idx /= height_out;
    int ic = idx;

    if (b >= batch_size || ic >= in_channels || h >= height_out || w >= width_out) return;

    float val = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h - (kh * dilation - padding);
            int w_in = w - (kw * dilation - padding);

            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                float w_val = weight[(oc * in_channels + ic) * kernel_size * kernel_size + kh * kernel_size + kw];
                val += input[(b * in_channels + ic) * height_in * width_in + h_in * width_in + w_in] * w_val;
            }
        }
    }

    output[(b * out_channels + oc) * height_out * width_out + h * width_out + w] = val;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation) {

    auto options = input.options();
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);

    int out_channels = weight.size(0);

    int padded_kernel_size = (kernel_size - 1) * dilation + 1;

    int height_out = (height_in - 1) * stride + padded_kernel_size - 2 * padding;
    int width_out = (width_in - 1) * stride + padded_kernel_size - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, options);

    dim3 blocks((in_channels * height_out * width_out + 255) / 256, out_channels, batch_size);
    dim3 threads(256);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height_in, width_in, 
        height_out, width_out, kernel_size, stride, padding, dilation);

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.cuda_op = conv_transpose2d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom CUDA implementation
        return self.cuda_op.conv_transpose2d_cuda(
            x,
            self.conv_transpose2d.weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
