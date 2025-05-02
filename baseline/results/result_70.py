import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for ConvTranspose3d
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive implementation of transposed 3D convolution (for illustration)
__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation,
    bool use_bias) {

    int b = blockIdx.z;
    int oc = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int out_w = idx % out_width;
    idx /= out_width;
    int out_h = idx % out_height;
    idx /= out_height;
    int out_d = idx % out_depth;

    if (b >= batch_size || oc >= out_channels || out_d >= out_depth || out_h >= out_height || out_w >= out_width) return;

    float val = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = out_d * stride - 2 * padding + kd * dilation;
                    int in_h = out_h * stride - 2 * padding + kh * dilation;
                    int in_w = out_w * stride - 2 * padding + kw * dilation;

                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {

                        float w = weight[oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                         ic * (kernel_size * kernel_size * kernel_size) +
                                         kd * (kernel_size * kernel_size) +
                                         kh * kernel_size +
                                         kw];

                        float v = input[b * (in_channels * in_depth * in_height * in_width) +
                                        ic * (in_depth * in_height * in_width) +
                                        in_d * (in_height * in_width) +
                                        in_h * in_width +
                                        in_w];

                        val += v * w;
                    }
                }
            }
        }
    }

    if (use_bias) {
        val += bias[oc];
    }

    output[b * (out_channels * out_depth * out_height * out_width) +
           oc * (out_depth * out_height * out_width) +
           out_d * (out_height * out_width) +
           out_h * out_width +
           out_w] = val;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size, int stride, int padding, int dilation,
    int groups) {

    auto in_sizes = input.sizes();
    int batch_size = in_sizes[0];
    int in_channels = in_sizes[1];
    int in_depth = in_sizes[2];
    int in_height = in_sizes[3];
    int in_width = in_sizes[4];

    int out_channels = weight.sizes()[0];
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding, dilation,
        !bias.is_empty());

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size, int stride, int padding, int dilation,
    int groups);
"""

# Compile the inline CUDA code
custom_conv_transpose3d = load_inline(
    name="custom_conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
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
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Learnable parameters
        self.weight = nn.Parameter(
            torch.Tensor(
                in_channels,
                out_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Register custom CUDA function
        self.custom_conv_transpose3d = custom_conv_transpose3d.conv_transpose3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_conv_transpose3d(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = 3
depth = 16
height = 32
width = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
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
