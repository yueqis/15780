import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D Transposed Convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Naive implementation of 3D transposed convolution (for demonstration)
__global__ void conv_transpose3d_kernel(
    const float* input, float* output,
    const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_size, int stride, int padding, int dilation,
    bool use_bias
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_work = batch_size * out_channels * out_depth * out_height * out_width;
    
    for (int i = tid; i < total_work; i += blockDim.x * gridDim.x) {
        // Unpack index
        int w = i % out_width;
        int h = (i / out_width) % out_height;
        int d = (i / (out_width * out_height)) % out_depth;
        int oc = (i / (out_width * out_height * out_depth)) % out_channels;
        int b = i / (out_width * out_height * out_depth * out_channels);

        float val = use_bias ? bias[oc] : 0.0f;

        // Loop over input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Loop over kernel elements
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Compute input coordinates
                        int id = d * stride - padding + kd * dilation;
                        int ih = h * stride - padding + kh * dilation;
                        int iw = w * stride - padding + kw * dilation;

                        if (id >= 0 && id < in_depth &&
                            ih >= 0 && ih < in_height &&
                            iw >= 0 && iw < in_width) {
                            
                            // Input and weight indices
                            int in_idx = b * in_channels * in_depth * in_height * in_width +
                                         ic * in_depth * in_height * in_width +
                                         id * in_height * in_width +
                                         ih * in_width +
                                         iw;

                            int weight_idx = oc * in_channels * kernel_size * kernel_size * kernel_size +
                                             ic * kernel_size * kernel_size * kernel_size +
                                             kd * kernel_size * kernel_size +
                                             kh * kernel_size +
                                             kw;

                            val += input[in_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        int out_idx = b * out_channels * out_depth * out_height * out_width +
                      oc * out_depth * out_height * out_width +
                      d * out_height * out_width +
                      h * out_width +
                      w;

        output[out_idx] = val;
    }
}

torch::Tensor custom_conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, bool use_bias) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int kernel_size = weight.size(2);  // Assuming square kernel
    int out_channels = weight.size(0);
    
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int grid_size = (output.numel() + block_size - 1) / block_size;
    
    conv_transpose3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding, dilation,
        use_bias
    );
    
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor custom_conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation, bool use_bias);
"""

# Compile the inline CUDA code
custom_ops = load_inline(
    name="custom_conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["custom_conv_transpose3d_cuda"],
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
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()

        # Initialize weights similar to PyTorch ConvTranspose3d
        self.weight = nn.Parameter(
            torch.Tensor(
                in_channels, out_channels, kernel_size, kernel_size, kernel_size
            )
        )
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights using Kaiming normal initialization
        nn.init.kaiming_normal_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.cuda_op = custom_ops.custom_conv_transpose3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make sure inputs are on the GPU
        x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda() if self.bias is not None else self.bias

        return self.cuda_op(
            x, weight, bias, self.stride, self.padding, self.dilation, self.use_bias
        )


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square kernel, so only one value needed).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)

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
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]


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
