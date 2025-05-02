from datasets import load_dataset

ds = load_dataset("ScalingIntelligence/KernelBench")
ds = ds["level_1"].to_dict()

problem_ids = ds["problem_id"]
names = ds["name"]
code = ds["code"]

sorted_list = sorted(zip(problem_ids, names, code), key=lambda x: x[0])
problem_ids, names, code = zip(*sorted_list)

# append Model, get_inputs, get_init_inputs to the code
for i in range(100):
    try:
        file_name = f"result_{i+1}.py"
        code_str = f"""\n\n{code[i]}\n"""

        with open(file_name, "a", encoding="latin1") as f:
            f.write(code_str)

        print(f"Successfully post-processed {file_name}")

    except Exception as e:
        print(f"Error post-processing {i+1}: {str(e)}")
