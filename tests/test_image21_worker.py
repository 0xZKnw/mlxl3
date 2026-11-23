"""Exercise the single-image CLI without loading Metal or model weights."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class ImageWorkerTest(unittest.TestCase):
    def test_registers_phase_cleanup_before_generating_one_image(self):
        path = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Resources/Image21/render.py"
        spec = importlib.util.spec_from_file_location("studio_render", path)
        worker = importlib.util.module_from_spec(spec)
        core = Mock()
        core.device_info.return_value = {"device_name": "test"}
        saver = Mock()
        modules = {
            "mlx": SimpleNamespace(core=core, nn=SimpleNamespace(Module=object)),
            "mlx.core": core,
            "mlx.utils": SimpleNamespace(tree_flatten=Mock()),
            "mflux.callbacks.instances.memory_saver": SimpleNamespace(MemorySaver=saver),
        }
        model = Mock()
        png = Mock(size=(512, 512))

        def generate(**kwargs):
            saver.assert_called_once_with(model, keep_transformer=False, cache_limit_bytes=None)
            model.callbacks.register.assert_called_once_with(saver.return_value)
            return SimpleNamespace(image=png)

        model.generate_image.side_effect = generate
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            spec.loader.exec_module(worker)
            output = Path(tmp) / "result.png"
            args = [
                "render",
                "--model",
                tmp,
                "--prompt",
                "oak",
                "--seed",
                "371",
                "--output",
                str(output),
            ]
            with patch.object(worker, "build", return_value=model), patch.object(sys, "argv", args):
                worker.main()
            model.generate_image.assert_called_once_with(
                seed=371, prompt="oak", width=512, height=512, num_inference_steps=40
            )
            png.save.assert_called_once_with(output, format="PNG")
