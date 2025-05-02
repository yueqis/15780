import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for depthwise 2D convolution with asymmetric kernel (kernel_size x 1)
depthwise_conv2d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) 
{
    const int hwid = blockIdx.x * blockDim.x + threadIdx.x;
    const int chid = blockIdx.y;
    const int batid = blockIdx.z;

    if (hwid >= out_height * out_width || chid >= channels || batid >= batch_size) {
        return;
    }

    const int hout = hwid / out_width;
    const int wout = hwid % out_width;
    const int hstart = hout * stride - padding;
    const int wstart = wout * stride - padding;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        int hin = hstart + i * dilation;
        if (hin < 0 || hin >= in_height) continue;
        int win = wstart;
        if (win < 0 || win >= in_width) continue;
        const float* input_ptr = input + batid * channels * in_height * in_width +
                                 chid * in_height * in_width +
                                 hin * in_width +
                                 win;
        const float* weight_ptr = weight + chid * kernel_size + i;
        sum += *input_ptr * *weight_ptr;
    }
    output[batid * channels * out_height * out_width +
           chid * out_height * out_width +
           hout * out_width +
           wout] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (in_width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    dim3 blocks((out_height * out_width + 255) / 256, channels, batch_size);
    dim3 threads(256);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation);

    return output;
}
"""

depthwise_conv2d_cpp = """
torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=[depthwise_conv2d_cpp],
    cuda_sources=[depthwise_conv2d_kernel],
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
    with_cuda=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.Tensor(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only support no bias for now, can be extended
        assert self.bias is None, "Bias is not supported in the custom CUDA kernel yet"
        return depthwise_conv2d.depthwise_conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding, self.dilation
        )
