"""Experimental NVFP4 entry point; Qwen4 architecture support is unverified."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODELOPT_COMMIT = "2758c046fbe041c5a645b360c71c384e227a8308"


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def validate_inputs(args):
    if not re.fullmatch(r"Qwen/[A-Za-z0-9][A-Za-z0-9._-]*", args.model_id):
        raise ValueError("Use a confirmed official model ID in the Qwen namespace")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Source revision must be a full 40-character commit SHA")
    script = args.modelopt_dir / "examples" / "hf_ptq" / "hf_ptq.py"
    if not script.is_file():
        raise ValueError("The Model Optimizer checkout must contain examples/hf_ptq/hf_ptq.py")
    commit = subprocess.check_output(
        ["git", "-C", str(args.modelopt_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != MODELOPT_COMMIT:
        raise ValueError(f"Expected Model Optimizer commit {MODELOPT_COMMIT}; got {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(args.modelopt_dir), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("Model Optimizer has tracked changes; use an unmodified checkout")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("Output directory must be absent or empty")
    return script


def build_command(args, script, source_path):
    return [
        sys.executable, str(script),
        "--pyt_ckpt_path", str(source_path),
        "--export_path", str(args.output_dir),
        "--qformat", "nvfp4",
        "--kv_cache_qformat", "none",
        "--dataset", args.dataset,
        "--calib_size", str(args.calib_size),
        "--calib_seq", str(args.calib_seq),
        "--batch_size", "1",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--modelopt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Reviewed upstream calibration dataset name")
    parser.add_argument("--calib-size", type=positive_int, default=512)
    parser.add_argument("--calib-seq", type=positive_int, default=2048)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloads or GPU use")
    args = parser.parse_args()
    args.modelopt_dir = args.modelopt_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    try:
        script = validate_inputs(args)
        if args.dry_run:
            print(json.dumps({"source_model": args.model_id, "source_revision": args.revision,
                              "modelopt_commit": MODELOPT_COMMIT, "status": "plan_only"}, indent=2))
            print(shlex.join(build_command(args, script, "<pinned-HF-snapshot>")))
            return

        import torch
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for quantization")
        source_path = Path(snapshot_download(repo_id=args.model_id, revision=args.revision))
        config_path = source_path / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("quantization_config"):
            raise ValueError("Start from the original unquantized model checkpoint")
        command = build_command(args, script, source_path)
        subprocess.run(command, cwd=script.parent, check=True)
        report = {
            "status": "exported_unvalidated",
            "source_model": args.model_id,
            "source_revision": args.revision,
            "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "modelopt_commit": MODELOPT_COMMIT,
            "command": command,
            "calibration_dataset": args.dataset,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "nvidia-modelopt", "huggingface-hub")},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_dir / "quantization-run.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print("Export complete. GPU inference and quality evaluation are still required.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Quantization stopped: {error}\n")


if __name__ == "__main__":
    main()
