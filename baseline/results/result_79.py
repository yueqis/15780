import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_1d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias,
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int input_length, 
    int output_length,
    int kernel_size,
    int stride, 
    int padding, 
    int dilation,
    bool use_bias) {

    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the output tensor
    int output_elements = batch_size * out_channels * output_length;
    if (tid >= output_elements) return;

    // Compute indices
    int out_idx = tid;
    int channel = out_idx / (batch_size * output_length);
    out_idx %= batch_size * output_length;
    int batch = out_idx / output_length;
    int out_pos = out_idx % output_length;

    // Calculate receptive field for transposed convolution
    int input_start = out_pos * stride - padding;
    
    // Accumulate result
    float acc = 0.0f;

    // Loop over kernel positions
    for (int k = 0; k < kernel_size; ++k) {
        // Compute input position with dilation
        int input_pos = input_start + k * dilation;

        // Check bounds
        if (input_pos >= 0 && input_pos < input_length) {
            // Input index
            int in_idx = batch * in_channels * input_length + channel * input_length + input_pos;
            
            // Weight index (output channels first)
            int weight_idx = channel * out_channels * kernel_size + k * out_channels + channel;
            
            // Multiply and accumulate
            acc += input[in_idx] * weight[weight_idx];
        }
    }

    // Apply bias if needed
    if (use_bias) {
        acc += bias[channel];
    }

    // Store result
    output[tid] = acc;
}

torch::Tensor conv_transpose_1d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, 
    int padding, 
    int dilation,
    bool use_bias) {
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    
    // Calculate output length using standard formula
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, in_channels, output_length}, input.options());
    
    // Launch kernel
    const int block_size = 256;
    const int output_elements = batch_size * in_channels * output_length;
    const int num_blocks = (output_elements + block_size - 1) / block_size;

    conv_transpose_1d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, 
        in_channels, 
        in_channels,  // Same as in_channels since we're doing grouped convolution
        input_length, 
        output_length,
        weight.size(2),  // kernel_size
        stride, 
        padding, 
        dilation,
        use_bias);

    return output;
}
"""

conv_transpose_1d_cpp_source = """
torch::Tensor conv_transpose_1d_cuda(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias,
    int stride, 
    int padding, 
    int dilation,
    bool use_bias);
"""

# Compile the inline CUDA code for custom transposed convolution
conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_cuda_source,
    functions=["conv_transpose_1d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    """
    Custom implementation of transposed 1D convolution using a CUDA kernel.
    """

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

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Create weight parameter
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))

        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

        # Register the CUDA convolution function
        self.conv_transpose_1d_func = conv_transpose_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using our custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Expand weight to match expected dimensions (out_channels comes first)
        expanded_weight = self.weight.permute(1, 0, 2).contiguous()

        # Call our custom CUDA convolution
        return self.conv_transpose_1d_func.conv_transpose_1d_cuda(
            x,
            expanded_weight,
            (
                self.bias
                if self.bias is not None
                else torch.zeros(self.out_channels, device=x.device)
            ),
            self.stride,
            self.padding,
            self.dilation,
            self.use_bias,
        )
