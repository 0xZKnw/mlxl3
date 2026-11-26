import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Resources/Image21/edit_vision.py"
SPEC = importlib.util.spec_from_file_location("edit_vision", SOURCE)
vision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vision)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def raw_shard(header, payload):
    raw = json.dumps(header, separators=(",", ":")).encode("ascii")
    return len(raw).to_bytes(8, "little") + raw, payload


def tensor(first, last):
    return {"dtype": "U8", "shape": [last - first], "data_offsets": [first, last]}


class Response(io.BytesIO):
    def __init__(self, status, content_range, body):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}


class VisionDownloadTest(unittest.TestCase):
    def setUp(self):
        self.constants = SimpleNamespace(HF_HUB_OFFLINE=False, HF_ASSETS_CACHE="/fixture")
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "huggingface_hub": SimpleNamespace(constants=self.constants),
                    "huggingface_hub.constants": self.constants,
                },
            )
        )

    def test_pinned_https_range_writes_exact_bytes(self):
        with patch.object(
            vision, "urlopen", return_value=Response(206, "bytes 2-4/7", b"CDE")
        ) as open_:
            out = io.BytesIO()
            vision._get_range("text_encoder/1.safetensors", 7, 2, 5, out)
        self.assertEqual(out.getvalue(), b"CDE")
        req = open_.call_args.args[0]
        self.assertEqual(req.full_url, vision.URL + "text_encoder/1.safetensors")
        self.assertEqual(req.get_header("Range"), "bytes=2-4")
        self.assertIsNone(req.get_header("Authorization"))

    def test_range_fails_on_status_range_and_body_mismatches(self):
        for status, origin, payload in (
            (200, "bytes 2-4/7", b"CDE"),
            (206, "bytes 3-5/7", b"CDE"),
            (206, "bytes 2-4/8", b"CDE"),
            (206, "bytes 2-4/7", b"CD"),
            (206, "bytes 2-4/7", b"CDEF"),
        ):
            with (
                self.subTest(status=status, origin=origin, payload=payload),
                patch.object(vision, "urlopen", return_value=Response(status, origin, payload)),
                self.assertRaises(OSError),
            ):
                vision._get_range("text_encoder/1.safetensors", 7, 2, 5, io.BytesIO())

    def test_packing_selects_only_vision_and_preserves_payload_bytes(self):
        shard_a = {
            "visual.b": tensor(5, 7),
            "language_model.a": tensor(2, 5),
            "visual.a": tensor(0, 2),
            "visual.c": tensor(7, 8),
        }
        shard_b = {"lm_head.weight": tensor(0, 3), "visual.d": tensor(3, 7)}
        origin = {
            "first": raw_shard(shard_a, b"abX!?cde"),
            "second": raw_shard(shard_b, b"???fghi"),
        }
        pins = {
            name: (len(head) + len(body), len(head), sha256(head))
            for name, (head, body) in origin.items()
        }

        def fetch(name, size, start, stop, out):
            raw = b"".join(origin[name])
            self.assertEqual(size, len(raw))
            out.write(raw[start:stop])

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.multiple(vision, SHARDS=pins, VISION_BYTES=9, VISION_TENSORS=4),
            patch.object(vision, "_get_range", side_effect=fetch) as ranges,
        ):
            path = Path(tmp) / "result.safetensors"
            vision._extract_vision(path)
            raw = path.read_bytes()
        header_size = int.from_bytes(raw[:8], "little")
        self.assertEqual(header_size % 8, 0)
        self.assertEqual(
            json.loads(raw[8 : 8 + header_size]),
            {
                "visual.a": tensor(0, 2),
                "visual.b": tensor(2, 4),
                "visual.c": tensor(4, 5),
                "visual.d": tensor(5, 9),
            },
        )
        self.assertEqual(raw[8 + header_size :], b"abcdefghi")
        self.assertEqual(ranges.call_count, 5)  # Two headers and three combined ranges.

    def test_negative_and_overlapping_offsets_and_duplicates_rejected(self):
        for shards in (
            [("a", 12, 4, {"visual.bad": tensor(-1, 2)})],
            [("a", 12, 4, {"visual.bad": tensor(4, 9)})],
            [("a", 12, 4, {"visual.x": tensor(0, 4), "visual.y": tensor(3, 7)})],
            [("a", 12, 4, {"visual.x": tensor(0, 4)}), ("b", 12, 4, {"visual.x": tensor(0, 4)})],
        ):
            with self.subTest(shards=shards), self.assertRaises(ValueError):
                vision._plan_vision(shards)

    def test_reuse_and_repair_preserve_previous_valid_assets(self):
        config = {"processor/preprocessor_config.json": b"metadata"}
        weights = b"lens"
        expected_configs = {
            name: (len(payload), sha256(payload)) for name, payload in config.items()
        }

        def fetch_config(name, size, digest, destination):
            self.assertEqual(expected_configs[name], (size, digest))
            destination.write_bytes(config[name])

        def fetch_weights(destination):
            destination.write_bytes(weights)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.multiple(
                vision,
                CONFIGS=expected_configs,
                VISION_FILE_BYTES=len(weights),
                VISION_SHA256=sha256(weights),
            ),
            patch.object(vision, "_download_config", side_effect=fetch_config) as cfg,
            patch.object(vision, "_extract_vision", side_effect=fetch_weights) as download,
        ):
            result = vision.prepare_edit_vision(tmp)
            self.assertEqual(result, vision.prepare_edit_vision(tmp))
            download.assert_called_once()
            cfg.assert_called_once()
            (result / vision.VISION_FILE).write_bytes(b"fake")  # Right size, wrong digest.
            with (
                patch.object(self.constants, "HF_HUB_OFFLINE", True),
                self.assertRaises(FileNotFoundError),
            ):
                vision.prepare_edit_vision(tmp)
            self.assertEqual(result, vision.prepare_edit_vision(tmp))
            self.assertEqual((result / vision.VISION_FILE).read_bytes(), weights)
            self.assertEqual(download.call_count, 2)
            cfg.assert_called_once()
            with patch.object(self.constants, "HF_HUB_OFFLINE", True):
                self.assertEqual(vision.prepare_edit_vision(tmp), result)

    def test_invalid_payload_leaves_neither_cache_entry_nor_staging(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.multiple(
                vision,
                CONFIGS={},
                VISION_FILE_BYTES=4,
                VISION_SHA256=sha256(b"real"),
            ),
            patch.object(vision, "_extract_vision", side_effect=lambda p: p.write_bytes(b"fake")),
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                vision.prepare_edit_vision(tmp)
            cache = Path(tmp) / "mlxl3-image21-vision" / vision.REVISION
            self.assertFalse((cache / ".download").exists())
            self.assertFalse((cache / vision.VISION_FILE).exists())


if __name__ == "__main__":
    unittest.main()
