```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Transposed 3D Convolution
conv_transpose3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Simplified transposed convolution kernel (1x1x1 kernel for demonstration)
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int depth_in, int height_in, int width_in,
    int depth_out, int height_out, int width_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int output_pad_d, int output_pad_h, int output_pad_w) {

    // Linear index for the output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * out_channels * depth_out * height_out * width_out; i += stride) {
        // Compute multi-dimensional indices
        int n = i / (out_channels * depth_out * height_out * width_out);
        int c_out = (i / (depth_out * height_out * width_out)) % out_channels;
        int d_out = (i / (height_out * width_out)) % depth_out;
        int h_out = (i / width_out) % height_out;
        int w_out = i % width_out;

        // Initialize output value
        float val = 0;

        // Loop over input channels and kernel dimensions
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kd = 0; kd < kernel_d; ++kd) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        // Calculate input position
                        int d_in = d_out / stride_d - pad_d + kd;
                        int h_in = h_out / stride_h - pad_h + kh;
                        int w_in = w_out / stride_w - pad_w + kw;

                        if (d_in >= 0 && d_in < depth_in &&
                            h_in >= 0 && h_in < height_in &&
                            w_in >= 0 && w_in < width_in) {
                            
                            // Load input and weight
                            float input_val = input[n * in_channels * depth_in * height_in * width_in +
                                                    c_in * depth_in * height_in * width_in +
                                                    d_in * height_in * width_in +
                                                    h_in * width_in + w_in];

                            float weight_val = weight[c_out * in_channels * kernel_d * kernel_h * kernel_w +
                                                      c_in * kernel_d * kernel_h * kernel_w +
                                                      kd * kernel_h * kernel_w +
                                                      kh * kernel_w + kw];

                            val += input_val * weight_val;
                        }
                    }
                }
            }
        }

        // Store result
        output[i] = val;
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int output_pad_d, int output_pad_h, int output_pad_w,
    int groups) {

    // We ignore groups for this simplified implementation
    TORCH_CHECK(groups == 1, "Only groups=1 is supported in custom kernel");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);

    // Calculate output dimensions
    int depth_out = (depth_in - 1) * stride_d - 2 * pad_d + kernel_d + output_pad_d;
    int height_out = (height_in - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int width_out = (width_in - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        output_pad_d, output_pad_h, output_pad_w);

    // Add bias if provided
    if (bias.defined()) {
        output = output + bias.view({1, out_channels, 1, 1, 1});
    }

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int output_pad_d, int output_pad_h, int output_pad_w,
    int groups);
"""

# Compile inline CUDA code
conv_transpose3d_op = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Define weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        
        # Define bias
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
        # Register custom op
        self.conv_transpose3d = conv_transpose3d_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, self.bias if self.bias is not None else torch.tensor([]),
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups
        )
```