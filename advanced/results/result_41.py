import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 1D
maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_length)
        return;

    int b = idx / (channels * out_length);
    int c = (idx / out_length) % channels;
    int l = idx % out_length;

    int start = l * stride - padding;
    float max_val = -1e20f;

    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = start + k * dilation;
        if (in_pos >= 0 && in_pos < in_length) {
            float val = input[b * channels * in_length + c * in_length + in_pos];
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    output[idx] = max_val;
}

torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_length = input.size(2);
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_length}, input.options());

    const int threads_per_block = 256;
    const int blocks = (batch_size * channels * out_length + threads_per_block - 1) / threads_per_block;

    maxpool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        in_length,
        out_length,
        kernel_size,
        stride,
        padding,
        dilation);

    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
maxpool1d_op = load_inline(
    name="maxpool1d",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=["maxpool1d_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel for Max Pooling 1D.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool_op = maxpool1d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA operator
        return self.maxpool_op.maxpool1d_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return self.maxpool(x)

batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
