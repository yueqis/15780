import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for depthwise 2D convolution
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int in_height, int in_width,
    int out_height, int out_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    const float* weight, const float* bias) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_height * out_width) return;

    const int w = idx % out_width;
    const int h = (idx / out_width) % out_height;
    const int c = (idx / (out_width * out_height)) % channels;
    const int b = idx / (channels * out_width * out_height);

    const int in_base_idx = b * channels * in_height * in_width;
    const int out_idx = b * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;

    float acc = 0.0f;

    const int weight_base_idx = c * kernel_h * kernel_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int h_in = -padding_h + h * stride_h + kh * dilation_h;
            const int w_in = -padding_w + w * stride_w + kw * dilation_w;

            if ((unsigned)h_in < (unsigned)in_height && (unsigned)w_in < (unsigned)in_width) {
                const int in_idx = in_base_idx + c * in_height * in_width + h_in * in_width + w_in;
                const int w_idx = weight_base_idx + kh * kernel_w + kw;
                acc += input[in_idx] * weight[w_idx];
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[c];
    }

    output[out_idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    const int total_threads = batch_size * channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr);

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w

        # Initialize weights and optional bias
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, 1, kernel_size_h, kernel_size_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=0.1)
        if bias:
            nn.init.zeros_(self.bias)

        # Bind compiled CUDA function
        self.depthwise_conv2d_func = depthwise_conv2d.depthwise_conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv2d_func(
            x,
            self.weight,
            self.bias,
            self.kernel_size_h,
            self.kernel_size_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )
