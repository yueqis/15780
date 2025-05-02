import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D Average Pooling
avg_pool_1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_1d_kernel(const float* input, float* output,
                                   int batch_size, int channels, int input_length, int output_length,
                                   int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_length) return;

    int b = idx / (channels * output_length);
    int c = (idx / output_length) % channels;
    int l = idx % output_length;

    int input_l_start = l * stride - padding;
    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = input_l_start + k;
        if (in_idx >= 0 && in_idx < input_length) {
            sum += input[b * channels * input_length + c * input_length + in_idx];
            ++count;
        }
    }

    output[idx] = (count > 0) ? sum / count : 0.0f;
}

torch::Tensor avg_pool_1d_cuda(torch::Tensor input,
                              int kernel_size, int stride, int padding,
                              int output_length) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int input_length = sizes[2];

    auto output = torch::zeros({batch_size, channels, output_length}, input.options());

    const int num_threads = 256;
    const int num_blocks = (batch_size * channels * output_length + num_threads - 1) / num_threads;

    avg_pool_1d_kernel<<<num_blocks, num_threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_length, output_length,
        kernel_size, stride, padding
    );

    return output;
}
"""

avg_pool_1d_cpp_source = """
torch::Tensor avg_pool_1d_cuda(torch::Tensor input,
                              int kernel_size, int stride, int padding,
                              int output_length);
"""

# Compile the inline CUDA code
avg_pool_1d_op = load_inline(
    name="avg_pool_1d",
    cpp_sources=avg_pool_1d_cpp_source,
    cuda_sources=avg_pool_1d_cuda_source,
    functions=["avg_pool_1d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool_cuda = avg_pool_1d_op.avg_pool_1d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output length manually like PyTorch's AvgPool1d
        input_length = x.size(2)
        output_length = (
            input_length + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        return self.avg_pool_cuda(
            x, self.kernel_size, self.stride, self.padding, output_length
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)

batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
