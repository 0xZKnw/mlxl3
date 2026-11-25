"""Exercise the actual worker CLI without Metal, weights or downloads."""

import contextlib
import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


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

    def test_matmul_depth_for_image_projection_shapes(self):
        for inputs, outputs, bits, depth in (
            (4096, 4096, 3, 64),
            (4096, 4096, 4, 64),
            (4096, 12288, 3, 64),
            (4096, 12288, 4, 32),
            (12288, 4096, 3, 32),
            (12288, 4096, 4, 64),
            (4096, 4096, 5, 16),
            (128, 256, 3, 16),
        ):
            for block_rows, group, expected in ((64, 4, depth), (32, 1, 16)):
                with self.subTest(inputs=inputs, outputs=outputs, bits=bits, group=group):
                    self.worker.qmm(inputs, outputs, bits, block_rows, group)
                    args = self.core.fast.metal_kernel.call_args.kwargs
                    self.assertIn(f"#define OP_K {expected}u", args["header"].splitlines())

    def test_native_window_depth_and_invalid_overrides(self):
        self.worker.qmm(4096, 4096, 3, 64, 4)
        shader = self.core.fast.metal_kernel.call_args.kwargs["source"]

        def preprocess(step_tiles, inputs, depth=None):
            command = ["cc", "-E", "-P", "-x", "c", "-DBK=16u"]
            command += [f"-DSTEP_TILES={step_tiles}u", f"-DINPUT_DIMS={inputs}u"]
            if depth is not None:
                command.append(f"-DOP_K={depth}u")
            return subprocess.run(
                [*command, "-"],
                input=shader,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        for tiles in (1, 2, 4, 8):
            with self.subTest(tiles=tiles):
                default = preprocess(tiles, 4112)
                old_depth = preprocess(tiles, 4112, 16)
                self.assertEqual(default.returncode, 0, default.stderr)
                self.assertEqual(old_depth.returncode, 0, old_depth.stderr)
                self.assertEqual(default.stdout, old_depth.stdout)
        for tiles, inputs, depth in ((1, 4096, 32), (4, 4112, 64), (4, 4096, 24)):
            with self.subTest(tiles=tiles, inputs=inputs, depth=depth):
                self.assertNotEqual(preprocess(tiles, inputs, depth).returncode, 0)

    def test_limits_allocator_cache_before_loading_at_512px(self):
        order = Mock()
        order.attach_mock(self.core.set_cache_limit, "limit")
        order.attach_mock(self.build, "build")
        self.run_worker("--size", "512")
        self.assertEqual(order.method_calls, [call.limit(512 * 1024**2), call.build(self.root)])

    def test_preserves_allocator_cache_at_larger_sizes(self):
        for size in (768, 1024):
            with self.subTest(size=size), patch.object(self.model.vae, "decode", self.decode):
                self.saver.reset_mock()
                self.model.callbacks.reset_mock()
                self.image.size = (size, size)
                self.run_worker("--size", str(size))
                self.core.set_cache_limit.assert_not_called()
                self.assertEqual(self.model.generate_image.call_args.kwargs["width"], size)
                self.assertEqual(self.model.generate_image.call_args.kwargs["height"], size)

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
        self.core.set_cache_limit.assert_not_called()
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
        self.core.set_cache_limit.assert_not_called()

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
