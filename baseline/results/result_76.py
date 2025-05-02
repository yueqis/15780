import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized 1D convolution
conv1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int dilation) {

    int b = blockIdx.z;
    int oc = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_length) return;

    int offset = b * in_channels * input_length;
    float* out_ptr = &output[b * out_channels * output_length + oc * output_length + i];

    for (int ic = 0; ic < in_channels; ++ic) {
        const float* in_channel = &input[offset + ic * input_length];
        const float* w_channel = &weight[oc * in_channels * kernel_size + ic * kernel_size];

        for (int k = 0; k < kernel_size; ++k) {
            int j = i * stride + k * dilation;
            if (j < input_length) {
                *out_ptr += in_channel[j] * w_channel[k];
            }
        }
    }
}

torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int output_length = (input_length + stride - 1) / stride;  // Simplified shape calculation

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 blocks((output_length + 255) / 256, out_channels, batch_size);
    dim3 threads(256);

    conv1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        dilation);

    return output;
}
"""

conv1d_cpp_source = """
torch::Tensor conv1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int dilation);
"""

# Compile the inline CUDA code for custom Conv1d
conv1d_op = load_inline(
    name="conv1d_cuda",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_cuda_source,
    functions=["conv1d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        self.stride = stride
        self.dilation = dilation
        self.conv1d_cuda = conv1d_op.conv1d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom CUDA convolution kernel
        output = self.conv1d_cuda(x, self.weight, self.stride, self.dilation)

        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        return output
