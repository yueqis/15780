# append profiling script
for i in range(100):
    try:
        file_name = f"result_{i+1}.py"

        code_str = f"""\n
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
    with proton.scope(proton_name, metrics={{}}):
        ms = do_bench(func, warmup=warmup, rep=repeat)
    return ms

func_model = lambda: Model()(*inputs)
func_model_new = lambda: ModelNew()(*inputs)
model_ms = bench(func_model, warmup=0, repeat=10, proton_name="Model")
model_new_ms = bench(func_model_new, warmup=0, repeat=10, proton_name="ModelNew")
print(f"Model: {{model_ms}} ms")
print(f"ModelNew: {{model_new_ms}} ms")
"""

        with open(file_name, "a", encoding="latin1") as f:
            f.write(code_str)

        print(f"Successfully post-processed {file_name}")

    except Exception as e:
        print(f"Error post-processing {i+1}: {str(e)}")
