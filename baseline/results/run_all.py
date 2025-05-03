import glob
import os
import subprocess
import sys
import json
from tqdm import tqdm


def main():
    py_files = [
        f for f in glob.glob("*.py") if os.path.basename(f).startswith("result_")
    ]
    py_files.sort()

    if len(py_files) != 100:
        print(f"Warning: find {len(py_files)} python files, but expected 100.")
    else:
        print("Find 100 python files, start running ...")

    results = []
    for task_id, script in tqdm(enumerate(py_files, start=1)):
        print(f">>> Task {task_id}: Running `{script}`")
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as e:
            results.append(
                {
                    "task_id": task_id,
                    "script": script,
                    "status": "timeout",
                    "error": str(e),
                }
            )
            continue

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            if "AssertionError" in err or "allclose" in err:
                status = "incorrect"
            else:
                status = "compile_error"
            results.append(
                {"task_id": task_id, "script": script, "status": status, "error": err}
            )
        else:
            model_time = None
            modelnew_time = None
            for line in proc.stdout.splitlines():
                if line.startswith("Model:"):
                    model_time = line.split("Model:", 1)[1].strip()
                elif line.startswith("ModelNew:"):
                    modelnew_time = line.split("ModelNew:", 1)[1].strip()
            results.append(
                {
                    "task_id": task_id,
                    "script": script,
                    "status": "success",
                    "Model": model_time,
                    "ModelNew": modelnew_time,
                }
            )

        with open("results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nAll tasks complete, save results to results.json")


if __name__ == "__main__":
    main()
