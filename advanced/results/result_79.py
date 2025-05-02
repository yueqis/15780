import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for transposed 1D convolution
conv1d_transpose_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void conv1d_transpose_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_length, 
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool use_bias) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_length * out_channels * batch_size) return;

    int o_ch = tid % out_channels;
    int temp = tid / out_channels;
    int length = temp % output_length;
    int batch = temp / output_length;

    float acc = 0.0f;

    for (int i_ch = 0; i_ch < in_channels; i_ch++) {
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = ((batch * in_channels + i_ch) * input_length);
            
            // Compute the position in the input that contributes to the current output position
            int d = (length - (k * dilation)) / stride;
            if ((length - (k * dilation)) % stride != 0 || d < 0 || d >= input_length) continue;
            
            acc += input[in_pos + d] * weight[((o_ch * in_channels + i_ch) * kernel_size + k)];
        }
    }

    if (use_bias) {
        acc += bias[o_ch];
    }

    int out_idx = ((batch * out_channels + o_ch) * output_length + length);
    output[out_idx] = acc;
}

torch::Tensor conv1d_transpose_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int dilation) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    bool use_bias = bias.numel() > 0;

    const int threads_per_block = 256;
    const int num_blocks = (output_length * out_channels * batch_size + threads_per_block - 1) / threads_per_block;

    conv1d_transpose_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_length, output_length,
        kernel_size, stride, padding, dilation, use_bias);

    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv1d_transpose_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
conv1d_transpose_op = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_cuda_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Create the weight parameter
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))

        # Initialize the weight using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity="leaky_relu")

        # Create the bias parameter if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Register the custom CUDA op
        self.conv1d_transpose = conv1d_transpose_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_transpose.conv1d_transpose_cuda(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 128
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
