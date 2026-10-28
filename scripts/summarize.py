"""Write an honest Actions summary from completed stage records only."""
import argparse
import json
import os
from pathlib import Path


def summarize(root):
    root.mkdir(parents=True, exist_ok=True)
    lines = ["# LACToS paper verification", "", "Datasets execute only on an ephemeral GitHub Actions runner.",
             "No dataset, reconstruction, model weight or raw map is uploaded as an artifact.", "",
             "| Stage | Observed status | Scope |", "|---|---|---|"]
    for stage in ["reference", "xray", "ffhq"]:
        path = root / stage / "verification.json"
        if not path.exists():
            lines.append(f"| {stage} | No completed result | Not run or failed; inspect step logs |")
            continue
        value = json.loads(path.read_text())
        lines.append(f"| {stage} | {value['status']} | {value['scope']} |")
    lines += ["", "Reference-map table recalculation is distinct from fresh scoring.",
              "Fresh scoring uses the authors' precomputed diffusion reconstructions; diffusion sampling and training are not rerun.",
              "FFHQ checks two images and is not a full-benchmark performance result."]
    content = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(content)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(content)
    print(content)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    summarize(p.parse_args().results)
