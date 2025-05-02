import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triplet_margin_loss_kernel(
    const float* anchor, 
    const float* positive, 
    const float* negative, 
    float* out, 
    int batch_size, 
    int vec_size,
    float margin) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float d_pos = 0.0f;
        float d_neg = 0.0f;

        // Compute squared L2 distances
        for (int j = 0; j < vec_size; ++j) {
            float a_p = anchor[i * vec_size + j] - positive[i * vec_size + j];
            float a_n = anchor[i * vec_size + j] - negative[i * vec_size + j];
            d_pos += a_p * a_p;
            d_neg += a_n * a_n;
        }

        // Hinge loss: max(0, d_pos - d_neg + margin)
        out[i] = fmaxf(0.0f, d_pos - d_neg + margin);
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor, 
    torch::Tensor positive, 
    torch::Tensor negative, 
    float margin) 
{
    auto size = anchor.size(0);
    auto out = torch::zeros({size}, anchor.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    triplet_margin_loss_kernel<<<num_blocks, block_size>>>(
        anchor.data_ptr<float>(), 
        positive.data_ptr<float>(), 
        negative.data_ptr<float>(), 
        out.data_ptr<float>(), 
        size, 
        anchor.size(1),
        margin);

    return out.mean();  // Return mean loss
}
"""

triplet_margin_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the inline CUDA code
triplet_margin_loss_op = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using a custom CUDA kernel to compute Triplet Margin Loss.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """

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
