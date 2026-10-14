import argparse
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.quantize import MODELOPT_COMMIT, build_command, positive_int, validate_inputs
from scripts.validate_artifact import validate_artifact, write_checksums


def write_shard(path, tensors):
    header = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x01\x02")


class QuantizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        script = root / "modelopt/examples/hf_ptq/hf_ptq.py"
        script.parent.mkdir(parents=True)
        script.touch()
        self.args = argparse.Namespace(
            model_id="Qwen/Test-Model", revision="a" * 40,
            modelopt_dir=root / "modelopt", output_dir=root / "output",
            dataset="cnn_dailymail", calib_size=512, calib_seq=2048,
        )

    def test_reject_nonofficial_model(self):
        self.args.model_id = "someone/Qwen4-27B"
        with self.assertRaisesRegex(ValueError, "official"):
            validate_inputs(self.args)

    def test_reject_mutable_source_revision(self):
        self.args.revision = "main"
        with self.assertRaisesRegex(ValueError, "40-character"):
            validate_inputs(self.args)

    def test_reject_missing_upstream_script(self):
        self.args.modelopt_dir = Path(self.temp.name) / "missing"
        with self.assertRaisesRegex(ValueError, "checkout"):
            validate_inputs(self.args)

    @patch("scripts.quantize.subprocess.check_output", return_value="b" * 40)
    def test_reject_wrong_tool_version(self, _):
        with self.assertRaisesRegex(ValueError, "Expected Model Optimizer"):
            validate_inputs(self.args)

    @patch("scripts.quantize.subprocess.check_output", side_effect=[MODELOPT_COMMIT, " M file.py"])
    def test_reject_modified_tool_checkout(self, _):
        with self.assertRaisesRegex(ValueError, "tracked changes"):
            validate_inputs(self.args)

    @patch("scripts.quantize.subprocess.check_output", side_effect=[MODELOPT_COMMIT, ""])
    def test_reject_existing_output(self, _):
        self.args.output_dir.mkdir()
        (self.args.output_dir / "checkpoint").touch()
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_inputs(self.args)

    @patch("scripts.quantize.subprocess.check_output", side_effect=[MODELOPT_COMMIT, ""])
    def test_pinned_command_uses_local_snapshot_and_nvfp4(self, _):
        script = validate_inputs(self.args)
        command = build_command(self.args, script, "pinned-snapshot")
        self.assertEqual(command[command.index("--pyt_ckpt_path") + 1], "pinned-snapshot")
        self.assertEqual(command[command.index("--qformat") + 1], "nvfp4")
        self.assertEqual(command[command.index("--kv_cache_qformat") + 1], "none")
        self.assertNotIn("--trust_remote_code", command)

    def test_reject_nonpositive_calibration(self):
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_int(value)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
        self.save_config()
        for name in ("tokenizer_config.json", "tokenizer.json", "quantization-run.json"):
            (self.root / name).write_text("{}", encoding="utf-8")
        self.tensors = {
            "model.layer.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
            "model.layer.weight_scale": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
        }
        write_shard(self.root / "model.safetensors", self.tensors)

    def save_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def test_valid_structure_and_checksums(self):
        result = validate_artifact(self.root)
        self.assertEqual(result["packed_weight_tensors"], 1)
        self.assertEqual(result["gpu_and_quality_validation"], "still_required")
        write_checksums(self.root)
        checksums = (self.root / "SHA256SUMS").read_text()
        digest = hashlib.sha256((self.root / "model.safetensors").read_bytes()).hexdigest()
        self.assertIn(f"{digest}  model.safetensors", checksums)
        self.assertNotIn("  SHA256SUMS", checksums)

    def test_reject_non_nvfp4_metadata(self):
        self.config["quantization_config"]["quant_algo"] = "FP8"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "NVFP4"):
            validate_artifact(self.root)

    def test_reject_missing_tokenizer(self):
        (self.root / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            validate_artifact(self.root)

    def test_reject_missing_weights(self):
        (self.root / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weight shards"):
            validate_artifact(self.root)

    def test_reject_fake_quantized_float_weights(self):
        self.tensors["model.layer.weight"]["dtype"] = "F16"
        write_shard(self.root / "model.safetensors", self.tensors)
        with self.assertRaisesRegex(ValueError, "packed uint8"):
            validate_artifact(self.root)

    def test_reject_missing_scales(self):
        del self.tensors["model.layer.weight_scale"]
        write_shard(self.root / "model.safetensors", self.tensors)
        with self.assertRaisesRegex(ValueError, "scales"):
            validate_artifact(self.root)

    def test_reject_truncated_tensor_data(self):
        self.tensors["model.layer.weight"]["data_offsets"] = [0, 100]
        write_shard(self.root / "model.safetensors", self.tensors)
        with self.assertRaisesRegex(ValueError, "data range"):
            validate_artifact(self.root)

    def test_reject_missing_shard_in_index(self):
        index = {"weight_map": {name: "missing.safetensors" for name in self.tensors}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing file"):
            validate_artifact(self.root)

    def test_reject_multiple_shards_without_index(self):
        write_shard(self.root / "second.safetensors", {
            "other.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        })
        with self.assertRaisesRegex(ValueError, "Multiple shards"):
            validate_artifact(self.root)


if __name__ == "__main__":
    unittest.main()
