import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax along a specified dimension
argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, int64_t size, int64_t dim, int64_t stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int64_t offset = idx * stride;
        scalar_t max_val = input[offset];
        int64_t max_index = 0;
        for (int64_t i = 1; i < dim; ++i) {
            scalar_t val = input[offset + i];
            if (val > max_val) {
                max_val = val;
                max_index = i;
            }
        }
        output[idx] = max_index;
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim) {
    auto input_size = input.sizes();
    int64_t num_dims = input.dim();
    
    // Ensure dim is within valid range
    if (dim < 0) {
        dim += num_dims;
    }
    
    // Compute output shape: remove the specified dimension
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < num_dims; ++i) {
        if (i != dim) {
            output_shape.push_back(input_size[i]);
        }
    }
    
    auto output = torch::empty(output_shape, input.options().dtype(torch::kInt64));
    
    int64_t total_size = 1;
    for (int64_t i = 0; i < num_dims; ++i) {
        if (i != dim) {
            total_size *= input_size[i];
        }
    }
    
    int64_t block_size = 256;
    int64_t num_blocks = (total_size + block_size - 1) / block_size;
    
    int64_t stride = input.stride(dim);
    int64_t dim_size = input_size[dim];
    
    // Reshape input to have all dimensions except 'dim' flattened
    std::vector<int64_t> perm(num_dims);
    int64_t j = 0;
    for (int64_t i = 0; i < num_dims; ++i) {
        if (i != dim) {
            perm[j++] = i;
        }
    }
    perm[j] = dim;
    
    auto perm_tensor = torch::from_blob(perm.data(), {num_dims}, torch::kInt64);
    auto input_permuted = input.permute(perm_tensor);
    
    std::vector<int64_t> reshape_dims(num_dims);
    for (int64_t i = 0; i < num_dims - 1; ++i) {
        reshape_dims[i] = input_permuted.size(i);
    }
    reshape_dims[num_dims - 1] = input_permuted.size(num_dims - 1);
    
    auto input_reshaped = input_permuted.reshape({total_size, dim_size});
    
    AT_DISPATCH_FLOATING_TYPES(input_reshaped.scalar_type(), "argmax_cuda", ([&] {
        using scalar_t = scalar_t;
        const scalar_t* input_data = input_reshaped.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmax_kernel<scalar_t><<<num_blocks, block_size>>>(input_data, output_data, total_size, dim_size, stride);
    }));
    
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
argmax_op = load_inline(
    name="argmax_op",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_cuda_source,
    functions=["argmax_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_op.argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_cuda(x, self.dim)
