import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Triplet Margin Loss
triplet_margin_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void triplet_margin_loss_kernel(
    const float* anchor, 
    const float* positive, 
    const float* negative, 
    float* output, 
    int64_t batch_size, 
    int64_t embedding_dim,
    float margin) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float d_pos = 0.0f;
    float d_neg = 0.0f;

    for (int64_t j = 0; j < embedding_dim; ++j) {
        float a_j = anchor[idx * embedding_dim + j];
        float p_j = positive[idx * embedding_dim + j];
        float n_j = negative[idx * embedding_dim + j];

        float diff_pos = a_j - p_j;
        float diff_neg = a_j - n_j;

        d_pos += diff_pos * diff_pos;
        d_neg += diff_neg * diff_neg;
    }

    float loss = fmaxf(margin + d_pos - d_neg, 0.0f);
    atomicAdd(output, loss);
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor, 
    torch::Tensor positive, 
    torch::Tensor negative, 
    float margin) {

    auto batch_size = anchor.size(0);
    auto embedding_dim = anchor.size(1);

    auto output = torch::zeros({}, anchor.options());

    int64_t threads_per_block = 256;
    int64_t blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    triplet_margin_loss_kernel<<<blocks, threads_per_block>>>(
        anchor.data_ptr<float>(), 
        positive.data_ptr<float>(), 
        negative.data_ptr<float>(),
        output.data_ptr<float>(), 
        batch_size, 
        embedding_dim, 
        margin);

    output /= batch_size;

    return output;
}
"""

triplet_margin_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the inline CUDA code
triplet_margin_loss_op = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_cuda_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.loss_fn = triplet_margin_loss_op.triplet_margin_loss_cuda

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative, self.margin)


import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin



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
