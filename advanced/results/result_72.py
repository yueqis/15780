import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(const float* input, const float* weight, float* output,
                                        int batch_size, int in_channels, int out_channels,
                                        int depth, int height, int width,
                                        int k_depth, int k_height, int k_width,
                                        int stride_depth, int stride_height, int stride_width,
                                        int padding_depth, int padding_height, int padding_width,
                                        int output_padding_depth, int output_padding_height, int output_padding_width,
                                        int groups) {
    // Compute output dimensions
    int o_depth = (depth - 1) * stride_depth - 2 * padding_depth + k_depth + output_padding_depth;
    int o_height = (height - 1) * stride_height - 2 * padding_height + k_height + output_padding_height;
    int o_width = (width - 1) * stride_width - 2 * padding_width + k_width + output_padding_width;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int index = gid; index < batch_size * out_channels * o_depth * o_height * o_width; index += num_threads) {
        int b = index / (out_channels * o_depth * o_height * o_width);
        int rest = index % (out_channels * o_depth * o_height * o_width);
        int oc = rest / (o_depth * o_height * o_width);
        rest = rest % (o_depth * o_height * o_width);
        int od = rest / (o_height * o_width);
        int oh = (rest / o_width) % o_height;
        int ow = rest % o_width;

        int group = oc / (out_channels / groups);

        // Adjust pointers for input and weight based on group
        int in_group_channels = in_channels / groups;
        int out_group_channels = out_channels / groups;

        const float* input_ptr = input + b * in_channels * depth * height * width + group * in_group_channels * depth * height * width;
        const float* weight_ptr = weight + group * out_group_channels * in_group_channels * k_depth * k_height * k_width + oc % out_group_channels * in_group_channels * k_depth * k_height * k_width;
        float* output_ptr = output + index;

        *output_ptr = 0;

        for (int ic = 0; ic < in_group_channels; ++ic) {
            for (int kd = 0; kd < k_depth; ++kd) {
                for (int kh = 0; kh < k_height; ++kh) {
                    for (int kw = 0; kw < k_width; ++kw) {
                        int id = od * stride_depth - padding_depth + kd;
                        int ih = oh * stride_height - padding_height + kh;
                        int iw = ow * stride_width - padding_width + kw;

                        if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int input_idx = ic * depth * height * width + id * height * width + ih * width + iw;
                            int weight_idx = ic * k_depth * k_height * k_width + kd * k_height * k_width + kh * k_width + kw;
                            *output_ptr += input_ptr[input_idx] * weight_ptr[weight_idx];
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int64_t batch_size, int64_t in_channels, int64_t out_channels,
                                   int64_t depth, int64_t height, int64_t width,
                                   int64_t k_depth, int64_t k_height, int64_t k_width,
                                   int64_t stride_depth, int64_t stride_height, int64_t stride_width,
                                   int64_t padding_depth, int64_t padding_height, int64_t padding_width,
                                   int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width,
                                   int64_t groups) {
    auto output = torch::zeros({batch_size, out_channels,
                                (depth - 1) * stride_depth - 2 * padding_depth + k_depth + output_padding_depth,
                                (height - 1) * stride_height - 2 * padding_height + k_height + output_padding_height,
                                (width - 1) * stride_width - 2 * padding_width + k_width + output_padding_width},
                               input.options());

    int threads = 1024;
    int blocks = (output.numel() + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth, height, width,
        k_depth, k_height, k_width,
        stride_depth, stride_height, stride_width,
        padding_depth, padding_height, padding_width,
        output_padding_depth, output_padding_height, output_padding_width,
        groups);

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, "
    "int64_t batch_size, int64_t in_channels, int64_t out_channels, "
    "int64_t depth, int64_t height, int64_t width, "
    "int64_t k_depth, int64_t k_height, int64_t k_width, "
    "int64_t stride_depth, int64_t stride_height, int64_t stride_width, "
    "int64_t padding_depth, int64_t padding_height, int64_t padding_width, "
    "int64_t output_padding_depth, int64_t output_padding_height, int64_t output_padding_width, "
    "int64_t groups);"
)

# Compile the inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom CUDA kernel for 3D transposed convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
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

        # Create weight parameter
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )

        # Initialize weights (similar to PyTorch's ConvTranspose3d)
        nn.init.kaiming_uniform_(self.weight, nonlinearity="leaky_relu", param=0.2)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.conv_transpose3d = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized 3D transposed convolution.
        """
        batch_size, in_channels, depth, height, width = x.shape
        _, out_channels, k_depth, k_height, k_width = self.weight.shape

        # Call the custom CUDA implementation
        output = self.conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            batch_size,
            in_channels,
            out_channels,
            depth,
            height,
            width,
            k_depth,
            k_height,
            k_width,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
        )

        # Add bias if enabled
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple of ints): Size of the convolution kernel in the form (kernel_size_depth, kernel_size_height, kernel_size_width).
        stride (tuple of ints, optional): Stride of the convolution in the form (stride_depth, stride_height, stride_width). Defaults to (1, 1, 1).
        padding (tuple of ints, optional): Padding applied to the input in the form (padding_depth, padding_height, padding_width). Defaults to (0, 0, 0).
        output_padding (tuple of ints, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]


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
