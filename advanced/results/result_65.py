import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_height, 
    int input_width,
    int output_height, 
    int output_width,
    int kernel_h, 
    int kernel_w,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int output_padding_h, 
    int output_padding_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int b = idx / (out_channels * output_height * output_width);
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int h_out = (idx / output_width) % output_height;
    int w_out = idx % output_width;

    float acc = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh;
                int w_in = w_out * stride_w - padding_w + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    float input_val = input[b * in_channels * input_height * input_width + c_in * input_height * input_width + h_in * input_width + w_in];
                    float weight_val = weight[c_out * in_channels * kernel_h * kernel_w + c_in * kernel_h * kernel_w + kh * kernel_w + kw];
                    acc += input_val * weight_val;
                }
            }
        }
    }

    output[b * out_channels * output_height * output_width + c_out * output_height * output_width + h_out * output_width + w_out] = acc;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    torch::IntArrayRef kernel_size,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int output_padding_h, 
    int output_padding_w) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);
    auto kernel_h = kernel_size[0];
    auto kernel_w = kernel_size[1];

    auto output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    auto output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int threads_per_block = 256;
    const int num_blocks = (batch_size * out_channels * output_height * output_width + threads_per_block - 1) / threads_per_block;

    transposed_conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        out_channels, 
        input_height, 
        input_width,
        output_height, 
        output_width,
        kernel_h, 
        kernel_w,
        stride_h, 
        stride_w,
        padding_h, 
        padding_w,
        output_padding_h, 
        output_padding_w);

    return output;
}
"""

transposed_conv2d_cpp_source = """
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    torch::IntArrayRef kernel_size,
    int stride_h, 
    int stride_w,
    int padding_h, 
    int padding_w,
    int output_padding_h, 
    int output_padding_w);
"""

# Compile the inline CUDA code for transposed convolution
transposed_conv2d = load_inline(
    name="transposed_conv2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Register weight as a parameter
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels // groups, *kernel_size)
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")

        # Register the custom CUDA function
        self.transposed_conv2d = transposed_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transposed_conv2d.transposed_conv2d_cuda(
            x,
            self.weight,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.output_padding,
            self.output_padding,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization


# correctness
assert len(get_init_inputs()) == 0
inputs = get_inputs()
inputs = [x.cuda() for x in inputs]
model_result = Model()(*inputs)
model_new_result = ModelNew()(*inputs)
assert torch.allclose(model_result.detach().cpu(), model_new_result.detach().cpu(), rtol=1e-02, atol=1e-03)

# profiling
import triton.profiler as proton
from triton.testing import do_bench

def bench(func, warmup=0, repeat=10, proton_name="kernel"):
    with proton.scope(proton_name, metrics={}):
        ms = do_bench(func, warmup=warmup, rep=repeat)
    return ms

func_model = lambda: Model()(*inputs)
func_model_new = lambda: ModelNew()(*inputs)
model_ms = bench(func_model, warmup=0, repeat=10, proton_name="Model")
model_new_ms = bench(func_model_new, warmup=0, repeat=10, proton_name="ModelNew")
print(f"Model: {model_ms} ms")
print(f"ModelNew: {model_new_ms} ms")
