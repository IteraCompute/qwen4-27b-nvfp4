# Release checklist

There is no model release yet. Work through these checks when the official source is available. Do not publish a model card that implies usable weights while this repository is in preparation.

## 1. Confirm the source and environment

- Confirm the official Qwen model ID, parameter count, architecture, tokenizer/chat template, and license. Pin the exact source commit SHA.
- Check Transformers, Model Optimizer, and the intended inference engine against the actual architecture. Update the pinned tool commit in a PR if required.
- Record the GPU, driver, CUDA, Python, and package versions. Reserve disk and GPU memory for the unquantized checkpoint and calibration.
- Review quantized layer coverage, sensitive projections, and any multimodal components. If the model cannot use the generic NVFP4 path, implement and test an architecture-specific recipe before running it.

## 2. Calibrate and evaluate

- Review calibration dataset licenses, domains, revisions, sample selection, and length distribution. Record dataset revisions and sample hashes alongside the run report; the wrapper records only the upstream dataset name.
- Start with a small smoke run, then calibrate the intended model. Retain logs and `quantization-run.json`.
- Run the unquantized baseline and NVFP4 checkpoint on the same frozen evaluation data and settings. Record quality deltas and agree acceptance thresholds before publishing.
- On compatible Blackwell hardware, test the selected engine, chat template, deterministic short generation, long-context behavior, peak memory, latency, and throughput.
- Publish measured values with hardware, engine versions, context lengths, concurrency, and sampling settings. Include applicable language and multimodal benchmarks. Do not infer VRAM or speed from weight bit width.
- Run `python scripts/validate_artifact.py <export-directory>`. This checks structure and creates checksums; it does not replace GPU or quality evaluation.

## 3. Package and publish

- Copy `templates/MODEL_CARD.md` to the export directory as `README.md`. Replace every placeholder with verified source/license information, supported engine commands, and measured results. Retain all required upstream license/notice files.
- Add evaluation results, environment details, and run logs. Rerun structural preflight after finalizing all files so `SHA256SUMS` describes the package.
- Confirm the HF organization namespace and write access. Authenticate locally with `hf auth login`; never put tokens into source control, issue bodies, or shell command arguments.
- Upload only the validated export directory. The intended destination below must be confirmed to be writable before use:

```bash
hf auth whoami
hf repos create IteraCompute/Qwen4-27B-NVFP4 --type model --private
hf upload IteraCompute/Qwen4-27B-NVFP4 artifacts/Qwen4-27B-NVFP4 . \
  --type model --commit-message 'Add validated NVFP4 checkpoint and evaluation report'
```

- Download the uploaded revision to a fresh directory with `hf download ... --revision <HF_COMMIT_SHA> --local-dir <directory>`, verify the original package checksums, and repeat engine loading/generation from that directory.
- After the checks pass, publish with `hf repos settings IteraCompute/Qwen4-27B-NVFP4 --type model --public`.
- Create a GitHub release pointing to the exact HF commit and include evaluation reports and checksums. Store large weights on HF; keep GitHub focused on code and reproducibility.

HF repository creation and uploads are manual. No credentials, scheduled checks, automatic publication, or GPU spending are configured by the preparation repository.
