import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for KL Divergence computation (log(predictions) + multiplication with targets)
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void kl_div_kernel(const float* pred, const float* target, float* out, int size, int dim_size, int batch_size) {
    extern __shared__ float s_sum[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    int offset = batch_idx * dim_size + (idx % dim_size);
    if (offset < (batch_idx + 1) * dim_size && offset < size) {
        float log_pred = logf(fmaxf(pred[offset], 1e-8)); // Avoid log(0)
        out[offset] = -target[offset] * log_pred;
    }

    __syncthreads();

    // Sum over the dimensions per batch element
    if (threadIdx.x < dim_size) {
        s_sum[threadIdx.x] = (idx < size && offset < size) ? out[offset] : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float batch_sum = 0.0f;
        for (int i = 0; i < dim_size; ++i) {
            batch_sum += s_sum[i];
        }
        out[batch_idx] = batch_sum / batch_size; // Average across the batch
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto batch_size = predictions.size(0);
    auto dim_size = predictions.size(1);

    auto out = torch::zeros(size + batch_size, predictions.options().dtype(torch::kFloat));

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x, batch_size);

    kl_div_kernel<<<grid, block, dim_size * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        dim_size,
        batch_size
    );

    return out.slice(0, 0, batch_size);
}
"""

kl_div_cpp_source = (
    "torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile inline CUDA code
kl_div_op = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div_op.kl_div_cuda

    def forward(self, predictions, targets):
        return self.kl_div(predictions, targets).sum()


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []



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
