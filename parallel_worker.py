"""Run one notebook experiment on the single GPU exposed to this process."""

import ast
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
VAL_DEBUG_SPLITS = "work/val_subset/splits"


def notebook_cell(notebook, cell_id):
    for cell in notebook["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise KeyError(f"Notebook cell {cell_id} is missing")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"--check", "e1", "e2", "e3"}:
        raise SystemExit("Usage: python parallel_worker.py --check|e1|e2|e3")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_device or "," in visible_device:
        raise RuntimeError("Worker must be assigned exactly one Kaggle GPU")
    if not (Path(VAL_DEBUG_SPLITS) / "val.jsonl").is_file():
        raise FileNotFoundError("Prepare the validation subset before starting workers")

    notebook = json.loads((PROJECT_ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
    e0_tree = ast.parse(notebook_cell(notebook, "e1f61f8b"))
    prompt_node = next(
        node.value
        for node in e0_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets)
    )
    globals()["prompt"] = ast.literal_eval(prompt_node)
    globals()["config"] = yaml.safe_load(Path("configs/e0.yaml").read_text(encoding="utf-8"))
    globals()["model_id"] = config["model_id"]

    for cell_id in ("8dcfba5e", "49fb342a", "598719fa", "18a942da"):
        exec(compile(notebook_cell(notebook, cell_id), f"cell:{cell_id}", "exec"), globals())

    if sys.argv[1] == "--check":
        print("Parallel worker bootstrap OK", flush=True)
        return

    experiment = sys.argv[1]
    result = run_experiment(experiment)
    result_path = Path("work") / f"{experiment}_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{experiment.upper()} complete: {result_path}", flush=True)


if __name__ == "__main__":
    main()
