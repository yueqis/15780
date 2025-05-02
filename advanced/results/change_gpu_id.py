# change all cuda() to cuda(device_id)

device_id = 1

for i in range(100):
    try:
        file_name = f"result_{i+1}.py"

        with open(file_name, "r", encoding="latin1") as f:
            src = f.read()

        replaced = src.replace("cuda()", f"cuda({device_id})")

        with open(file_name, "w", encoding="latin1") as f:
            f.write(replaced)

        print(f"Successfully post-processed {file_name}")

    except Exception as e:
        print(f"Error post-processing {i+1}: {str(e)}")
