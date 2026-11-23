"""Exercise the actual worker CLI without Metal, weights or downloads."""

import contextlib
import importlib.util
import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ImageWorkerTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Resources/Image21/render.py"
        spec = importlib.util.spec_from_file_location("studio_render", path)
        self.worker = importlib.util.module_from_spec(spec)
        self.core = Mock()
        self.core.device_info.return_value = {"device_name": "Apple M5"}
        self.core.isfinite.side_effect = lambda tensor: [math.isfinite(x) for x in tensor.values]
        self.core.all.side_effect = lambda values: SimpleNamespace(item=lambda: all(values))
        self.saver = Mock()
        modules = {
            "mlx": SimpleNamespace(core=self.core, nn=SimpleNamespace(Module=object)),
            "mlx.core": self.core,
            "mlx.utils": SimpleNamespace(tree_flatten=Mock()),
            "mflux.callbacks.instances.memory_saver": SimpleNamespace(MemorySaver=self.saver),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        spec.loader.exec_module(self.worker)
        self.enterContext(
            patch.object(self.worker.platform, "mac_ver", return_value=("26.3", (), ""))
        )
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("MLXL3_DISABLE_TENSOR_OPS", None)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model = Mock()
        self.build = self.enterContext(patch.object(self.worker, "build", return_value=self.model))
        self.image = Mock(size=(512, 512))
        self.latent = self.tensor([0.25, -1.0])
        self.pixels = self.tensor([0.0, 0.75])
        self.decode = Mock(side_effect=lambda _: self.pixels)
        self.model.vae.decode = self.decode

        def generate(**kwargs):
            self.saver.assert_called_once_with(
                self.model, keep_transformer=False, cache_limit_bytes=None
            )
            self.model.callbacks.register.assert_called_once_with(self.saver.return_value)
            self.assertIs(self.model.vae.decode(self.latent), self.pixels)
            return SimpleNamespace(image=self.image)

        self.model.generate_image.side_effect = generate

    @staticmethod
    def tensor(values):
        return SimpleNamespace(size=len(values), values=values)

    def run_worker(self, *extra):
        output = self.root / "result.png"
        args = [
            "render",
            "--model",
            str(self.root),
            "--prompt",
            "oak",
            "--seed",
            "371",
            "--output",
            str(output),
        ]
        with patch.object(sys, "argv", args + list(extra)):
            self.worker.main()
        return output

    def test_registers_phase_cleanup_before_generating_one_image(self):
        self.image.save.assert_not_called()
        output = self.run_worker()
        self.model.generate_image.assert_called_once_with(
            seed=371, prompt="oak", width=512, height=512, num_inference_steps=40
        )
        self.image.save.assert_called_once_with(output, format="PNG")

    def test_two_steps(self):
        self.run_worker("--steps", "2")
        self.assertEqual(self.model.generate_image.call_args.kwargs["num_inference_steps"], 2)

    def test_fifty_steps(self):
        self.run_worker("--steps", "50")
        self.assertEqual(self.model.generate_image.call_args.kwargs["num_inference_steps"], 50)

    def test_invalid_steps_rejected_before_setup(self):
        for value in ("0", "1", "51"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as cm,
            ):
                self.run_worker("--steps", value)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--steps", err.getvalue())
        self.build.assert_not_called()
        self.core.device_info.assert_not_called()

    def test_rejects_unsupported_hardware_before_loading(self):
        for version, gpu, metal, disabled in (
            ("26.3", "Apple M4 Max", True, False),
            ("26.3", "Apple M50", True, False),
            ("26.2.9", "Apple M5", True, False),
            ("", "", False, False),
            ("27.0", "Apple M5", False, False),
            ("27.0", "Apple M5", True, True),
        ):
            with (
                self.subTest(version=version, gpu=gpu),
                patch.dict(os.environ, {"MLXL3_DISABLE_TENSOR_OPS": "1"} if disabled else {}),
            ):
                self.worker.platform.mac_ver.return_value = (version, (), "")
                self.core.device_info.return_value = {"device_name": gpu}
                self.core.metal.is_available.return_value = metal
                with self.assertRaisesRegex(RuntimeError, "M5 GPU and macOS 26.3"):
                    self.run_worker()
        self.build.assert_not_called()

    def test_accepts_m5_family(self):
        self.core.device_info.return_value = {"device_name": "Apple M5 Max"}
        self.run_worker()
        self.image.save.assert_called_once()

    def test_rejects_invalid_latent_before_decode(self):
        for values in ([0.1, float("nan")], [float("inf")], [-float("inf")], []):
            with self.subTest(values=values):
                self.latent = self.tensor(values)
                self.saver.reset_mock()
                self.model.callbacks.reset_mock()
                self.model.vae.decode = self.decode
                with self.assertRaisesRegex(ValueError, "latents"):
                    self.run_worker()
        self.decode.assert_not_called()
        self.image.save.assert_not_called()

    def test_rejects_invalid_pixels_before_png(self):
        for values in ([0, float("nan")], [float("inf")], [-float("inf")], []):
            with self.subTest(values=values):
                self.pixels = self.tensor(values)
                self.saver.reset_mock()
                self.model.callbacks.reset_mock()
                self.model.vae.decode = self.decode
                with self.assertRaisesRegex(ValueError, "pixels"):
                    self.run_worker()
        self.assertEqual(self.decode.call_count, 4)
        self.image.save.assert_not_called()
