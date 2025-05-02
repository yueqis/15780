import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Depthwise 2D Convolution
depthwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int channels, 
    int in_height, 
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = out_height * out_width * channels * batch_size;

    if (idx >= total_threads) return;

    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    int in_start_h = h * stride - padding;
    int in_start_w = w * stride - padding;

    float acc = 0.0f;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = in_start_h + kh;
            int in_w = in_start_w + kw;

            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                int in_idx = b * channels * in_height * in_width + c * in_height * in_width + in_h * in_width + in_w;
                int weight_idx = c * kernel_size * kernel_size + kh * kernel_size + kw;
                acc += input[in_idx] * weight[weight_idx];
            }
        }
    }

    int out_idx = b * channels * out_height * out_width + c * out_height * out_width + h * out_width + w;
    output[out_idx] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int stride,
    int padding) {

    auto in_channels = input.size(1);
    auto batch_size = input.size(0);
    auto in_height = input.size(2);
    auto in_width = input.size(3);

    auto kernel_size = weight.size(2);
    auto out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    auto out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());

    const int num_threads = 512;
    const int64_t total_elements = output.numel();
    const int num_blocks = (total_elements + num_threads - 1) / num_threads;

    depthwise_conv2d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width
    );

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_cuda_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, 1, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter("bias", None)
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.reset_parameters()

        # Load custom CUDA operator
        self.depthwise_conv2d_op = depthwise_conv2d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom CUDA implementation for depthwise convolution
        out = self.depthwise_conv2d_op.depthwise_conv2d_cuda(
            x, self.weight, self.stride, self.padding
        )

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out
