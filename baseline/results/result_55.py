import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2d with asymmetric input optimization
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, float* output,
                              int batch_size, int in_channels, int out_channels,
                              int in_height, int in_width,
                              int out_height, int out_width,
                              int kernel_size, int stride, int padding) {
    int n = blockIdx.z;
    int oh = blockIdx.y;
    int ow = blockIdx.x;
    int oc = threadIdx.z;
    int ic = threadIdx.y;
    int k = threadIdx.x;

    int kh = k / kernel_size;
    int kw = k % kernel_size;

    int ih_start = oh * stride - padding + kh;
    int iw_start = ow * stride - padding + kw;

    if (ih_start >= 0 && ih_start < in_height && iw_start >= 0 && iw_start < in_width) {
        int input_idx = n * in_channels * in_height * in_width + ic * in_height * in_width + ih_start * in_width + iw_start;
        int weight_idx = oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
        atomicAdd(&output[n * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow],
                  input[input_idx] * weight[weight_idx]);
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int in_channels, int out_channels,
                          int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 blocks(out_width, out_height, batch_size);
    dim3 threads(in_channels, 1, out_channels);

    conv2d_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
                                      batch_size, in_channels, out_channels,
                                      in_height, in_width,
                                      out_height, out_width,
                                      kernel_size, stride, padding);

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int in_channels, int out_channels,
                          int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for custom Conv2d
conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
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
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == 1, "Custom CUDA kernel currently supports only groups=1"
        assert dilation == 1, "Custom CUDA kernel currently supports only dilation=1"
        assert not bias, "Custom CUDA kernel currently supports only bias=False"

        self.conv2d_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights using Kaiming normal initialization
        torch.nn.init.kaiming_normal_(self.conv2d_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_op.conv2d_cuda(
            x,
            self.conv2d_weight,
            x.size(1),
            self.conv2d_weight.size(0),
            self.kernel_size,
            self.stride,
            self.padding,
        )
