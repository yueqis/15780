import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed convolution with asymmetric kernel size (3x5)
transposed_conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive implementation of 2D transposed convolution for asymmetric kernel (3x5)
__global__ void transposed_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height_in, int width_in,
    int height_out, int width_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height_out * width_out;
    
    if (idx >= total_elements) return;

    int w = idx % width_out;
    int h = (idx / width_out) % height_out;
    int c_out = (idx / (width_out * height_out)) % out_channels;
    int b = idx / (width_out * height_out * out_channels);

    float val = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h - padding_h + kh * dilation_h;
                int w_in = w - padding_w + kw * dilation_w;

                if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    int in_idx = b * in_channels * height_in * width_in +
                                 c_in * height_in * width_in +
                                 h_in * width_in + w_in;

                    int weight_idx = c_out * in_channels * kernel_h * kernel_w +
                                     c_in * kernel_h * kernel_w +
                                     kh * kernel_w + kw;

                    val += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    int out_idx = b * out_channels * height_out * width_out +
                  c_out * height_out * width_out +
                  h * width_out + w;

    output[out_idx] = val;
}

torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool with_bias
) {
    // Only support groups == 1
    TORCH_CHECK(groups == 1, "Only support groups=1");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height_in = input.size(2);
    int width_in = input.size(3);

    int out_channels = weight.size(0);
    int weight_in_channels = weight.size(1);
    TORCH_CHECK(in_channels == weight_in_channels, "Mismatch between input channels and weight shape");

    int height_out = (height_in - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1 + output_padding_h;
    int width_out = (width_in - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1 + output_padding_w;

    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    transposed_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height_in, width_in,
        height_out, width_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w
    );

    if (with_bias) {
        auto bias_expanded = bias.view({1, out_channels, 1, 1});
        output += bias_expanded;
    }

    return output;
}
"""

transposed_conv2d_cpp_source = """
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool with_bias
);
"""

# Compile the inline CUDA code
conv_transpose2d_op = load_inline(
    name="conv_transpose2d",
    cpp_sources=transposed_conv2d_cpp_source,
    cuda_sources=transposed_conv2d_cuda_source,
    functions=["transposed_conv2d_cuda"],
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
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
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
        self.dilation = dilation
        self.groups = groups

        # Register learnable parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="leaky_relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # CUDA operator
        self.transposed_conv2d = conv_transpose2d_op.transposed_conv2d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transposed_conv2d(
            x,
            self.weight,
            self.bias,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.output_padding[0],
            self.output_padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.bias is not None,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
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
