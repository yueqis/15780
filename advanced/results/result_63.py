import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for 2D convolution
conv2d_cuda_code = """
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
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width
) {
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int ic = blockIdx.x;

    for (int oh = threadIdx.y; oh < output_height; oh += blockDim.y) {
        for (int ow = threadIdx.x; ow < output_width; ow += blockDim.x) {
            float acc = 0.0f;
            int ih_base = oh * stride - padding;
            int iw_base = ow * stride - padding;

            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = ih_base + kh;
                    int iw = iw_base + kw;

                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        acc += input[b * in_channels * input_height * input_width + ic * input_height * input_width + ih * input_width + iw] *
                               weight[oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw];
                    }
                }
            }

            atomicAdd(&output[b * out_channels * output_height * output_width + oc * output_height * output_width + oh * output_width + ow], acc);
        }
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);

    auto out_channels = weight.size(0);
    auto output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    auto output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 blocks(in_channels, out_channels, batch_size);
    dim3 threads(8, 8);  // Tune this block size for best performance

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_height,
        output_width
    );

    return output;
}
"""

# C++ declaration for the CUDA function
conv2d_cpp_decl = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code
conv2d_op = load_inline(
    name="conv2d_cuda",
    cpp_sources=conv2d_cpp_decl,
    cuda_sources=conv2d_cuda_code,
    functions=["conv2d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized 2D convolution using a custom CUDA kernel.
    """

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
        assert groups == 1, "Only support groups == 1"
        assert dilation == 1, "Only support dilation == 1"

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Register the custom CUDA op
        self.conv2d_cuda = conv2d_op.conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
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
kernel_size = 3
width = 256
height = 256

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
