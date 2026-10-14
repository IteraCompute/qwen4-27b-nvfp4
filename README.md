# Qwen4-27B NVFP4

[![CI](https://github.com/IteraCompute/qwen4-27b-nvfp4/actions/workflows/ci.yml/badge.svg)](https://github.com/IteraCompute/qwen4-27b-nvfp4/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

An independent community project by [IteraCompute](https://iteracompute.com) preparing a reproducible NVFP4 quantization and release workflow for the anticipated Qwen4-27B model.

**Status: preparation only.** There are no Qwen4 model weights, benchmark results, or validated Qwen4 compatibility in this repository. The final model ID, architecture, license, and supported inference engines must be confirmed after the official release. This project is not affiliated with Qwen or NVIDIA.

**中文：** 本仓库提前准备 Qwen4-27B 的 NVFP4 量化与发布流程。当前只有工具和文档，没有模型权重或已验证的性能结果。模型正式发布后，先确认架构与许可证，再进行校准、Blackwell 推理验证和质量评测。本项目独立维护，不代表 Qwen 官方。当前不启用定时模型检测。

## What is ready

- A small wrapper around NVIDIA's official Model Optimizer PTQ example, with a pinned upstream commit, explicit source revision, dry-run mode, and run provenance.
- A CPU-only artifact preflight that checks NVFP4 metadata, packed weight headers, tokenizer files, and shard completeness, then generates SHA-256 checksums.
- CI, automatic assignment of new issues to the maintainer, and a pull request template.
- A [release checklist](docs/RELEASE.md) and a [Hugging Face model card template](templates/MODEL_CARD.md).

Model Optimizer supports NVFP4 export for several existing Qwen architectures. Qwen4 support is still unverified; see the [official support matrix](https://github.com/NVIDIA/Model-Optimizer/tree/2758c046fbe041c5a645b360c71c384e227a8308/examples/hf_ptq). NVFP4 inference requires compatible NVIDIA Blackwell hardware and engine support. Calibration memory must be budgeted for the original model, activations, and export buffers, not just the eventual 4-bit checkpoint.

## Prepare a GPU environment

The toolkit itself uses Python 3.11+ and the standard library. GPU quantization uses the dependencies of the pinned upstream checkout. Run the following in a Linux CUDA environment, after checking its PyTorch/CUDA requirements:

```bash
git clone https://github.com/IteraCompute/qwen4-27b-nvfp4.git
cd qwen4-27b-nvfp4
git clone https://github.com/NVIDIA/Model-Optimizer.git ../Model-Optimizer
git -C ../Model-Optimizer checkout 2758c046fbe041c5a645b360c71c384e227a8308
python -m pip install '../Model-Optimizer[hf]'
```

The pinned version is a starting point. If Qwen4 needs new upstream support, update the pin and review the wrapper in a PR before using it. No remote model code is enabled by this wrapper.

## Quantize after the official release

Replace both placeholders with the confirmed official repository and its full 40-character commit SHA. These are placeholders, not a claim that this model exists:

```bash
python scripts/quantize.py \
  --model-id 'Qwen/REPLACE_WITH_OFFICIAL_MODEL_ID' \
  --revision 'REPLACE_WITH_40_CHARACTER_SOURCE_COMMIT' \
  --modelopt-dir ../Model-Optimizer \
  --output-dir artifacts/Qwen4-27B-NVFP4 \
  --dataset cnn_dailymail \
  --dry-run
```

Remove `--dry-run` to download the pinned source and run quantization. The initial recipe requests NVFP4 weights and activations, leaving the KV cache at its original precision. It uses 512 calibration samples, sequence length 2048, and batch size 1. These are initial experiment settings, not an accuracy guarantee. Review quantized layer coverage and calibration data against the actual architecture before a real run.

After GPU and quality evaluation, run:

```bash
python scripts/validate_artifact.py artifacts/Qwen4-27B-NVFP4
```

This is a structural preflight; it cannot establish numerical accuracy or inference compatibility. Follow [the release checklist](docs/RELEASE.md) before uploading to HF or creating a GitHub model release. No automatic uploads, scheduled model checks, or paid GPU jobs are enabled.

## Development

```bash
python -m unittest discover -s tests -v
```

Contribute through a PR; `main` requires the `tests` CI check. Open an [issue](https://github.com/IteraCompute/qwen4-27b-nvfp4/issues) for architecture support, calibration, evaluation, or packaging work.

## License

The toolkit code is [MIT licensed](LICENSE). Model weights, tokenizers, and calibration datasets keep their own upstream licenses. Confirm the official Qwen4 terms before redistributing any derived checkpoint.
