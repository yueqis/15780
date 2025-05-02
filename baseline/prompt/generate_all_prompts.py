import os
from datasets import load_dataset

ds = load_dataset("ScalingIntelligence/KernelBench")
ds = ds["level_1"].to_dict()

problem_ids = ds["problem_id"]
names = ds["name"]
code = ds["code"]

sorted_list = sorted(zip(problem_ids, names, code), key=lambda x: x[0])
problem_ids, names, code = zip(*sorted_list)

PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""
PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

example_arch_src = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
"""


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""

    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


example_new_arch_src = read_file(os.path.join(os.getcwd(), "example_new_arch_src.py"))


def prompt_generate_custom_cuda(arc_src: str) -> str:
    prompt = PROBLEM_STATEMENT

    prompt += f"""
    Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
    ``` \n
    {example_arch_src}
    ``` \n
    The example new arch with custom CUDA kernels looks like this: 
    ```
    {example_new_arch_src}
    ``` \n
    """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt


prompt1 = prompt_generate_custom_cuda(code[0])

print(prompt1)
