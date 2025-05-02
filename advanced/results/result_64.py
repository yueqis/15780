import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for transposed 1D convolution
conv1d_transpose_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be on CUDA")
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
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool use_bias) {
    
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int i = threadIdx.x;

    int group_id = c_out / (out_channels / groups);
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;

    int weight_offset = group_id * group_out_channels * group_in_channels * kernel_size +
                        (c_out % group_out_channels) * group_in_channels * kernel_size;

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    if (n * c_out * i >= batch_size * out_channels * output_length) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < group_in_channels; ++c_in) {
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = n * in_channels * input_length + (group_id * group_in_channels + c_in) * input_length + 
                         (i - padding + k * stride);
            int weight_pos = weight_offset + c_in * kernel_size + k;
            if ((i - padding + k * stride) >= 0 && (i - padding + k * stride) < input_length) {
                sum += input[in_pos] * weight[weight_pos];
            }
        }
    }

    if (use_bias) {
        output[n * out_channels * output_length + c_out * output_length + i] = sum + bias[c_out];
    } else {
        output[n * out_channels * output_length + c_out * output_length + i] = sum;
    }
}

torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 blocks(batch_size, out_channels);
    dim3 threads(input_length);

    bool use_bias = bias.numel() > 0;

    conv1d_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        use_bias);

    return output;
}
"""

conv1d_transpose_cpp_source = """
torch::Tensor conv1d_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups);
"""

# Compile the inline CUDA code
conv1d_transpose_op = load_inline(
    name="conv1d_transpose",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_cuda_source,
    functions=["conv1d_transpose_cuda"],
    verbose=True,
    extra_cflags=["-O2"],
    with_cuda=True,
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
        self.use_bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

        # Register custom CUDA op
        self.conv1d_transpose_cuda = conv1d_transpose_op.conv1d_transpose_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_transpose_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.tensor([]),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
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
in_channels = 64
out_channels = 3
kernel_size = 3
length = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
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
