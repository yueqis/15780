import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int in_channels, int depth, int height, int width,
    int out_channels, int kernel_size,
    int stride, int padding, int dilation,
    int out_depth, int out_height, int out_width) {

    int oc = blockIdx.z;
    int b = blockIdx.y;
    int idx = threadIdx.x;

    int total_threads = blockDim.x;
    for (int oh = blockIdx.x * blockDim.x + idx; oh < out_depth * out_height * out_width; oh += gridDim.x * blockDim.x) {
        int d = oh / (out_height * out_width);
        int h = (oh / out_width) % out_height;
        int w = oh % out_width;

        int in_d_start = d * stride - padding;
        int in_h_start = h * stride - padding;
        int in_w_start = w * stride - padding;

        float val = 0.0f;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int in_d = in_d_start + kd * dilation;
                        int in_h = in_h_start + kh * dilation;
                        int in_w = in_w_start + kw * dilation;

                        if (in_d >= 0 && in_d < depth &&
                            in_h >= 0 && in_h < height &&
                            in_w >= 0 && in_w < width) {
                            float input_val = input[b * in_channels * depth * height * width +
                                                    ic * depth * height * width +
                                                    in_d * height * width +
                                                    in_h * width +
                                                    in_w];
                            float weight_val = weight[oc * in_channels * kernel_size * kernel_size * kernel_size +
                                                      ic * kernel_size * kernel_size * kernel_size +
                                                      kd * kernel_size * kernel_size +
                                                      kh * kernel_size +
                                                      kw];
                            val += input_val * weight_val;
                        }
                    }
                }
            }
        }

        output[b * out_channels * out_depth * out_height * out_width +
               oc * out_depth * out_height * out_width +
               d * out_height * out_width +
               h * out_width +
               w] = val;
    }
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);

    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 blocks(out_depth * out_height * out_width, batch_size, 1);
    dim3 threads(256);

    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, depth, height, width,
        out_channels, kernel_size,
        stride, padding, dilation,
        out_depth, out_height, out_width);

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code
conv3d_op = load_inline(
    name="conv3d_op",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
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
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == 1, "Groups > 1 not supported in custom kernel"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.randn(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

        # Register the custom op
        self.conv3d_cuda = conv3d_op.conv3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            x += self.bias.view(1, -1, 1, 1, 1)
        return x


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with square input and square kernel.

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
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
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
