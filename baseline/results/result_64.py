import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Transposed 1D Convolution
conv1d_transpose_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

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
    int groups) {

    // Compute output length manually inside the kernel
    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    int n = blockIdx.x; // Batch index
    int oc = blockIdx.y; // Output channel
    int i = threadIdx.x; // Position in output

    if (n >= batch_size || oc >= out_channels || i >= output_length) return;

    int group_id = oc / (out_channels / groups);
    int weight_offset = group_id * (in_channels / groups) * kernel_size;

    float sum = 0.0f;
    for (int ic = group_id * (in_channels / groups); ic < (group_id + 1) * (in_channels / groups); ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = n * in_channels * input_length + ic * input_length + ((i - k + padding) / stride);
            int weight_pos = oc * in_channels * kernel_size + ic * kernel_size + k;
            if ((i - k + padding) % stride == 0 && ((i - k + padding) / stride) >= 0 && ((i - k + padding) / stride) < input_length) {
                sum += input[input_pos] * weight[weight_pos];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[n * out_channels * output_length + oc * output_length + i] = sum;
}

std::vector<torch::Tensor> conv1d_transpose_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

    dim3 grid(batch_size, out_channels, 1);
    dim3 block(output_length, 1, 1);

    conv1d_transpose_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups);

    return {output};
}
"""

conv1d_transpose_cpp_source = """
#include <torch/extension.h>

std::vector<torch::Tensor> conv1d_transpose_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups);
"""

# Compile inline CUDA code
conv1d_transpose_cuda = load_inline(
    name="conv1d_transpose_cuda",
    cpp_sources=conv1d_transpose_cpp_source,
    cuda_sources=conv1d_transpose_cuda_source,
    functions=["conv1d_transpose_cuda_forward"],
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

        # Initialize weights and bias similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="leaky_relu", a=0.2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_transpose_cuda.conv1d_transpose_cuda_forward(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )[0]


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
