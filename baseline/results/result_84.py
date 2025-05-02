import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise 2D convolution with square kernel
depthwise_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size, 
    int channels, 
    int height_in, 
    int width_in, 
    int height_out, 
    int width_out, 
    int kernel_size, 
    int stride, 
    int padding) 
{
    int b = blockIdx.z;
    int c = blockIdx.y;
    int ox = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ox >= width_out * height_out) return;

    int oy = ox / width_out;
    int ox_rem = ox % width_out;

    int px_start = ox_rem * stride - padding;
    int py_start = oy * stride - padding;

    float acc = 0.0f;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int px = px_start + kx;
            int py = py_start + ky;

            if (px >= 0 && px < width_in && py >= 0 && py < height_in) {
                float in_val = input[b * channels * height_in * width_in + c * height_in * width_in + py * width_in + px];
                float w_val = weight[c * kernel_size * kernel_size + ky * kernel_size + kx];
                acc += in_val * w_val;
            }
        }
    }

    output[b * channels * height_out * width_out + c * height_out * width_out + oy * width_out + ox_rem] = acc;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_size, 
    int stride, 
    int padding) 
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);
    
    int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, height_out, width_out}, input.options());

    dim3 blocks((width_out * height_out + 255) / 256, channels, batch_size);
    dim3 threads(256);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, 
        channels, 
        height_in, 
        width_in, 
        height_out, 
        width_out, 
        kernel_size, 
        stride, 
        padding
    );

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_size, 
    int stride, 
    int padding);
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
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_weight = nn.Parameter(
            torch.Tensor(out_channels, 1, kernel_size, kernel_size)
        )
        if bias:
            self.conv_bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("conv_bias", None)

        # Initialize weights like PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.conv_weight, nonlinearity="relu")
        if bias:
            nn.init.zeros_(self.conv_bias)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply custom CUDA convolution
        out = self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, self.conv_weight, self.kernel_size, self.stride, self.padding
        )

        # Add bias if enabled
        if self.bias:
            out = out + self.conv_bias.view(1, -1, 1, 1)

        return out


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]


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
