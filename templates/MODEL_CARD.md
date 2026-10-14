---
license: other
license_name: replace-with-official-upstream-license
license_link: REPLACE_WITH_OFFICIAL_LICENSE_URL
base_model: Qwen/REPLACE_WITH_OFFICIAL_MODEL_ID
library_name: modelopt
tags:
  - nvfp4
  - quantized
  - modelopt
---

# Qwen4-27B NVFP4 — release template

**UNPUBLISHED TEMPLATE. Replace all placeholders and remove this notice only after GPU and quality validation. This file does not announce available weights.**

Independent community quantization by [IteraCompute](https://github.com/IteraCompute/qwen4-27b-nvfp4), not affiliated with Qwen or NVIDIA.

## Source and recipe

- Official source: `REPLACE_WITH_MODEL_ID`
- Source revision: `REPLACE_WITH_SOURCE_COMMIT_SHA`
- Upstream model license: `REPLACE_WITH_VERIFIED_LICENSE`
- Model Optimizer commit and package versions: `REPLACE_WITH_RUN_REPORT`
- Actual quantized layer coverage and exclusions: `REPLACE_WITH_REVIEWED_COVERAGE`
- Calibration dataset revisions, samples, lengths, and seed: `REPLACE_WITH_CALIBRATION_REPORT`
- KV cache precision: `REPLACE_WITH_ACTUAL_PRECISION`

## Evaluation

`REPLACE_WITH_BASELINE_AND_NVFP4_RESULTS_AND_ACCEPTANCE_THRESHOLDS`

Include quality deltas, hardware, engine versions, measured peak memory, context length, concurrency, latency, and throughput. Separate measured values from estimates. State unsupported use cases and observed regressions.

## Usage

`REPLACE_WITH_ENGINE_COMMANDS_TESTED_ON_THE_PUBLISHED_HF_REVISION`

NVFP4 inference requires compatible Blackwell hardware and support for this exact architecture/format in the selected engine. Do not assume support from the engine name alone.

## Reproducibility and limitations

`REPLACE_WITH_GITHUB_RELEASE_AND_HF_COMMIT_LINKS`

Include run provenance, checksums, evaluation commands, and original license/notice files. Explain any calibration-data, long-context, multilingual, or multimodal limitations observed during testing.
