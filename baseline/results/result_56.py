import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D convolution with asymmetric kernel support
conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels,
    int input_height, 
    int input_width,
    int kernel_height, 
    int kernel_width,
    int stride_height, 
    int stride_width,
    int padding_height, 
    int padding_width,
    int dilation_height, 
    int dilation_width
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    int b  = blockIdx.w;

    if (ox >= (input_width  + 2 * padding_width  - kernel_width) / stride_width + 1 ||
        oy >= (input_height + 2 * padding_height - kernel_height) / stride_height + 1 ||
        oc >= out_channels || b >= batch_size) {
        return;
    }

    int ix_base = b * in_channels * input_height * input_width;
    int w_base = oc * in_channels * kernel_height * kernel_width;
    int oy_out = oy * stride_height - padding_height;
    int ox_out = ox * stride_width - padding_width;

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int iy = oy_out + ky * dilation_height;
                int ix = ox_out + kx * dilation_width;

                if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
                    float inp = input[ix_base + ic * input_height * input_width + iy * input_width + ix];
                    float wgt = weight[w_base + ic * kernel_height * kernel_width + ky * kernel_width + kx];
                    sum += inp * wgt;
                }
            }
        }
    }

    output[b * out_channels * ((input_height + 2 * padding_height - kernel_height) / stride_height + 1) * 
           ((input_width + 2 * padding_width - kernel_width) / stride_width + 1) +
           oc * ((input_height + 2 * padding_height - kernel_height) / stride_height + 1) * 
           ((input_width + 2 * padding_width - kernel_width) / stride_width + 1) +
           oy * ((input_width + 2 * padding_width - kernel_width) / stride_width + 1) + ox] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();

    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_height = input_sizes[2];
    int input_width = input_sizes[3];
    int out_channels = weight_sizes[0];
    int kernel_height = weight_sizes[2];
    int kernel_width = weight_sizes[3];

    int out_height = (input_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int out_width = (input_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((out_width + 15) / 16, (out_height + 15) / 16, out_channels, batch_size);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w);
"""

# Compile the inline CUDA code
conv2d_cuda_op = load_inline(
    name="conv2d_cuda",
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
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()

        if groups != 1:
            raise ValueError(
                "This custom implementation currently only supports groups=1"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define weights and bias similar to Conv2d
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Register the custom CUDA op
        self.conv2d_cuda = conv2d_cuda_op

    def reset_parameters(self):
        # Initialize weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our custom CUDA convolution
        output = self.conv2d_cuda.conv2d_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
        )

        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions

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
