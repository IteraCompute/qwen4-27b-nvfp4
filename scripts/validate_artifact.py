"""Structural NVFP4 preflight only; does not validate accuracy or engine support."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def read_tensor_header(path):
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 100_000_000 or length + 8 > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        header = json.loads(handle.read(length))
    data_size = path.stat().st_size - length - 8
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"No tensors in {path.name}")
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        if not 0 <= start < end <= data_size:
            raise ValueError(f"Invalid tensor data range: {name}")
    return tensors


def validate_artifact(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt" or quant.get("quant_algo") != "NVFP4":
        raise ValueError("Expected Model Optimizer NVFP4 quantization metadata")
    for name in ("tokenizer_config.json", "quantization-run.json"):
        json.loads((directory / name).read_text(encoding="utf-8"))
    if not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Missing tokenizer.json or tokenizer.model")
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ValueError("No safetensors weight shards")
    tensors = {}
    for shard in shards:
        for name, tensor in read_tensor_header(shard).items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            tensors[name] = (shard.name, tensor)
    index_path = directory / "model.safetensors.index.json"
    if len(shards) > 1 and not index_path.is_file():
        raise ValueError("Multiple shards require model.safetensors.index.json")
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if not weight_map or set(weight_map) != set(tensors):
            raise ValueError("Shard index does not match exported tensors")
        if any(tensors[name][0] != shard for name, shard in weight_map.items()):
            raise ValueError("Shard index references the wrong or missing file")
    packed = [name for name, (_, tensor) in tensors.items()
              if name.endswith(".weight") and tensor.get("dtype") == "U8"]
    if not packed:
        raise ValueError("No packed uint8 weights found; metadata alone does not establish NVFP4 export")
    for name in packed:
        scale_prefix = name.removesuffix(".weight") + ".weight_scale"
        if not any(key.startswith(scale_prefix) for key in tensors):
            raise ValueError(f"Missing quantization scales for {name}")
    return {"status": "structural_preflight_passed", "weight_shards": len(shards),
            "packed_weight_tensors": len(packed), "gpu_and_quality_validation": "still_required"}


def write_checksums(directory):
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            lines.append(f"{digest.hexdigest()}  {path.relative_to(directory).as_posix()}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = validate_artifact(args.directory)
        write_checksums(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Artifact preflight failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
