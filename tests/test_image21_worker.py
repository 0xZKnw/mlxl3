"""Exercise the actual worker CLI without Metal, weights or downloads."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

# CLI spellings and dimensions for the four squares offered by Studio.
IMAGE_SIZE_CASES = (
    ("512", (512, 512)),
    ("512x512", (512, 512)),
    ("768", (768, 768)),
    ("768x768", (768, 768)),
    ("1024", (1024, 1024)),
    ("1024x1024", (1024, 1024)),
    ("2048", (2048, 2048)),
    ("2048x2048", (2048, 2048)),
)


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
        self.actual_build = self.worker.build
        self.build = self.enterContext(patch.object(self.worker, "build", return_value=self.model))
        self.build_editor = self.enterContext(
            patch.object(self.worker, "build_editor", return_value=self.model)
        )
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

    def test_turbo_uses_eight_steps_and_an_explicit_variant(self):
        self.run_worker("--variant", "turbo")
        self.build.assert_called_once_with(self.root, "turbo")
        self.model.generate_image.assert_called_once_with(
            seed=371, prompt="oak", width=512, height=512, num_inference_steps=8
        )

    def test_turbo_rejects_base_step_count_before_setup(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.run_worker("--variant", "turbo", "--steps", "40")
        self.assertIn("exactly 8 steps", err.getvalue())
        self.build.assert_not_called()
        self.core.device_info.assert_not_called()

    def test_edit_routes_one_reference_to_turbo_512_pipeline(self):
        reference = self.root / "User image.png"
        source = object()
        self.core.device_info.return_value["memory_size"] = 24 * 1024**3
        with patch.object(self.worker, "load_reference", return_value=source) as load:
            output = self.run_worker(
                "--variant",
                "turbo",
                "--size",
                "512x512",
                "--low-memory",
                "--reference-image",
                str(reference),
            )
        load.assert_called_once_with(reference)
        self.build_editor.assert_called_once_with(self.root)
        self.build.assert_not_called()
        self.model.generate_image.assert_called_once_with(
            seed=371,
            prompt="oak",
            width=512,
            height=512,
            num_inference_steps=8,
            output_resolution=512,
            image_paths=[source],
        )
        self.image.save.assert_called_once_with(output, format="PNG")
        self.core.set_cache_limit.assert_called_once_with(512 * 1024**2)

    def test_edit_rejects_wrong_variant_and_size_before_setup(self):
        for variant, size in (
            ("base", "512"),
            *(
                ("turbo", value)
                for value, dimensions in IMAGE_SIZE_CASES
                if dimensions != (512, 512)
            ),
        ):
            with self.subTest(variant=variant, size=size):
                with (
                    patch.object(
                        self.worker,
                        "load_reference",
                        side_effect=AssertionError("Unsupported mode opened the reference"),
                    ) as load,
                    contextlib.redirect_stderr(io.StringIO()) as err,
                    self.assertRaises(SystemExit) as stop,
                ):
                    self.run_worker(
                        "--variant", variant, "--size", size, "--reference-image", "ref.png"
                    )
                self.assertEqual(stop.exception.code, 2)
                self.assertIn("requires Turbo at 512 px", err.getvalue())
                load.assert_not_called()
        self.core.device_info.assert_not_called()
        self.build_editor.assert_not_called()
        self.build.assert_not_called()

    def test_edit_rejects_unreadable_reference_before_setup(self):
        with (
            patch.object(self.worker, "load_reference", side_effect=OSError("corrupt PNG")),
            contextlib.redirect_stderr(io.StringIO()) as err,
            self.assertRaises(SystemExit) as stop,
        ):
            self.run_worker("--variant", "turbo", "--reference-image", "bad.png")
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("Invalid reference image: corrupt PNG", err.getvalue())
        self.core.device_info.assert_not_called()
        self.build_editor.assert_not_called()

    def test_reference_checks_png_size_aspect_and_single_frame_before_copy(self):
        source = self.root / "reference"
        source.write_bytes(b"fixture")
        image = Mock()
        open_png = Mock(side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(image))

        class DecompressionBombError(Exception):
            pass

        pillow = SimpleNamespace(open=open_png, DecompressionBombError=DecompressionBombError)
        with patch.dict(sys.modules, {"PIL": SimpleNamespace(Image=pillow)}):
            for size, frames in (((4097, 512), 1), ((512, 63), 1), ((0, 512), 1), ((512, 512), 2)):
                with self.subTest(size=size, frames=frames):
                    image.size, image.n_frames = size, frames
                    with self.assertRaisesRegex(ValueError, "still PNG"):
                        self.worker.load_reference(source)
            image.copy.assert_not_called()
            image.size, image.n_frames = (4096, 512), 1
            self.assertIs(self.worker.load_reference(source), image.copy.return_value)
            open_png.side_effect = DecompressionBombError()
            with self.assertRaisesRegex(ValueError, "safe decode dimensions"):
                self.worker.load_reference(source)
        open_png.assert_called_with(source, formats=["PNG"])
        image.copy.assert_called_once_with()

    def test_editor_keeps_local_quantized_language_and_adds_only_vision(self):
        local_weights = SimpleNamespace(
            components={"text_encoder": {"language": object()}},
            meta_data=SimpleNamespace(quantization_level=4),
        )
        language_value, vision_value = object(), object()
        self.worker.tree_flatten.return_value = [("layers.0.weight", language_value)]
        load = Mock(return_value=local_weights)
        apply = Mock(return_value=4)
        init = Mock()
        local_component = SimpleNamespace(name="text_encoder")
        edit_component = SimpleNamespace(name="text_encoder")
        local_definition = Mock()
        local_definition.get_components.return_value = [local_component]
        edit_definition = Mock()
        edit_definition.get_components.return_value = [edit_component]
        flatten = Mock(return_value={"whole": "tree"})
        modules = {
            "mlx.utils": SimpleNamespace(tree_unflatten=flatten),
            "mflux.models.common.weights.loading.weight_applier": SimpleNamespace(
                WeightApplier=SimpleNamespace(apply_and_quantize_single=apply)
            ),
            "mflux.models.common.weights.loading.weight_loader": SimpleNamespace(
                WeightLoader=SimpleNamespace(load_single_local=load)
            ),
            "mflux.models.qwen21.qwen21_initializer": SimpleNamespace(Qwen21Initializer=init),
            "mflux.models.qwen21.weights.qwen21_weight_definition": SimpleNamespace(
                Qwen21WeightDefinition=local_definition
            ),
            "mflux.models.qwen21.weights.qwen_image21_weight_definition": SimpleNamespace(
                QwenImage21WeightDefinition=edit_definition
            ),
        }
        vision_path = self.root / "vision.safetensors"
        with patch.dict(sys.modules, modules):
            for invalid in ({}, {"language_model.layers.0.weight": vision_value}):
                with self.subTest(vision=invalid):
                    self.core.load.return_value = invalid
                    with self.assertRaisesRegex(ValueError, "Unexpected tensors"):
                        self.worker.load_edit_encoder(self.model, self.root, vision_path)
            apply.assert_not_called()
            self.core.load.return_value = {"visual.patch_embed.weight": vision_value}
            self.assertEqual(self.worker.load_edit_encoder(self.model, self.root, vision_path), 4)
        self.core.load.assert_called_with(str(vision_path))
        load.assert_called_with(local_component, self.root)
        expected = {
            "language_model.layers.0.weight": language_value,
            "visual.patch_embed.weight": vision_value,
        }
        self.assertEqual(dict(flatten.call_args.args[0]), expected)
        apply.assert_called_once_with(
            local_weights, self.model, edit_component, None, edit_definition.quantization_predicate
        )
        init._validate_weights.assert_called_once_with(
            "text_encoder", self.model, expected, edit_definition
        )

    def test_checkpoint_and_scheduler_are_selected_together(self):
        # In the pinned MFLUX pipeline the model config selects either the fixed
        # 8-sigma Turbo scheduler or the resolution-dependent base scheduler.
        config = Mock()
        init = Mock()

        class ImageModel:
            transformer = object()

        modules = {
            "huggingface_hub": SimpleNamespace(snapshot_download=Mock(return_value=str(self.root))),
            "mflux.models.common.compute_precision": SimpleNamespace(ComputePrecision=Mock()),
            "mflux.models.common.config": SimpleNamespace(ModelConfig=config),
            "mflux.models.qwen21.qwen21_initializer": SimpleNamespace(Qwen21Initializer=init),
            "mflux.models.qwen21.variants.txt2img.qwen_image_21": SimpleNamespace(
                QwenImage21=ImageModel
            ),
            "mflux.models.qwen21.weights.qwen21_weight_definition": SimpleNamespace(
                Qwen21WeightDefinition=object
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(self.worker, "load_denoiser") as base,
            patch.object(self.worker, "load_turbo_denoiser") as turbo,
        ):
            for variant, expected, target, other in (
                ("turbo", config.qwen_image_21_turbo.return_value, turbo, base),
                ("base", config.qwen_image_21.return_value, base, turbo),
            ):
                with self.subTest(variant=variant):
                    model = self.actual_build(self.root, variant)
                    init.init_config.assert_called_with(model, expected)
                    target.assert_called_once_with(model.transformer, self.root)
                    other.assert_not_called()
                    target.reset_mock()

    def test_published_turbo_manifest_identity_before_loading_weights(self):
        (self.root / "quantization.json").write_text('{"source":"base"}')
        with self.assertRaisesRegex(ValueError, "Turbo EXL3"):
            self.worker.load_turbo_denoiser(None, self.root)
        self.core.load.assert_not_called()

    def test_loads_each_default_codebook_projection_from_its_own_file(self):
        # Synthetic manifest with the actual 32 x 7 module names. The digest is
        # replaced only for this fixture so CI needs neither Metal nor a model.
        names = [f"transformer_blocks.{i}.{p}" for i in range(32) for p in self.worker.PROJECTIONS]
        layers = {}
        for name in names:
            filename = hashlib.sha256(name.encode()).hexdigest() + ".safetensors"
            (self.root / filename).write_bytes(name.encode())
            layers[name] = {
                "file": filename,
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "bits": 4,
                "mcg": False,
                "mul1": False,
                "bias": False,
                "in_features": 128,
                "out_features": 256,
            }
        manifest = {
            "source": {"repo_id": self.worker.TURBO_SOURCE, "revision": self.worker.TURBO_REVISION},
            "layers": layers,
            "unquantized_sha256": hashlib.sha256(b"dense").hexdigest(),
        }
        (self.root / "unquantized.safetensors").write_bytes(b"dense")
        self.worker.tree_flatten.return_value = [
            (n + ".weight", SimpleNamespace(shape=(256, 128))) for n in names
        ]
        values = {"trellis": object(), "suh": object(), "svh": object()}
        dense = {"learned.weight": object()}
        self.core.load.side_effect = lambda path: dense if "unquantized." in path else values
        linear = self.enterContext(patch.object(self.worker, "EXL3Linear"))
        install = self.enterContext(patch.object(self.worker, "install_denoiser"))

        def load():
            raw = json.dumps(manifest).encode()
            (self.root / "quantization.json").write_bytes(raw)
            with patch.object(
                self.worker, "TURBO_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()
            ):
                self.worker.load_turbo_denoiser(self.model, self.root)

        load()
        self.assertEqual(linear.call_count, 224)
        self.assertEqual(install.call_args.args[2], dense)
        self.assertEqual([n for n, _ in install.call_args.args[3]], names)
        for name, invocation in zip(names, linear.call_args_list, strict=True):
            self.assertEqual(invocation.args[0], {f"{name}.{k}": v for k, v in values.items()})
            self.assertEqual(invocation.args[1:], (name, (256, 128), 4))
            self.assertEqual(invocation.kwargs, {"mcg": False})
        for field, wrong in (("mcg", True), ("mul1", True), ("bits", 3), ("file", "../outside")):
            with (
                self.subTest(field=field),
                patch.dict(layers[names[0]], {field: wrong}),
                self.assertRaisesRegex(ValueError, "Invalid Turbo EXL3 layer"),
            ):
                load()
        with (
            patch.dict(layers[names[0]], {"sha256": "0" * 64}),
            self.assertRaisesRegex(ValueError, "Checksum mismatch"),
        ):
            load()
        del layers[names[0]]
        with self.assertRaisesRegex(ValueError, "Incomplete Turbo"):
            load()

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
        self.assertEqual(
            order.method_calls, [call.limit(512 * 1024**2), call.build(self.root, "base")]
        )

    def test_allocator_cache_for_variant_size_and_hardware(self):
        gib = 1024**3
        cap = 512 * 1024**2
        previous = 4 * gib
        self.core.set_cache_limit.return_value = previous
        for gpu, memory, default_sizes, low_memory_sizes in (
            ("Apple M5", 24 * gib, (512, 768), (512, 768, 1024)),
            ("Apple M5", 16 * gib, (512,), (512,)),
            ("Apple M5", 32 * gib, (512,), (512,)),
            ("Apple M5", None, (512,), (512,)),
            ("Apple M5 Pro", 24 * gib, (512,), (512,)),
            ("Apple M5 Max", 24 * gib, (512,), (512,)),
        ):
            device = {"device_name": gpu}
            if memory is not None:
                device["memory_size"] = memory
            self.core.device_info.return_value = device
            for variant, (spec, dimensions), low_memory in product(
                ("base", "turbo"), IMAGE_SIZE_CASES, (False, True)
            ):
                with self.subTest(
                    variant=variant, size=spec, gpu=gpu, memory=memory, low_memory=low_memory
                ):
                    self.core.reset_mock()
                    self.build.reset_mock()
                    self.saver.reset_mock()
                    self.model.callbacks.reset_mock()
                    self.model.vae.decode = self.decode
                    width, height = dimensions
                    self.image.size = dimensions
                    self.image.save.reset_mock()
                    order = Mock()
                    order.attach_mock(self.core.set_cache_limit, "limit")
                    order.attach_mock(self.build, "build")
                    order.attach_mock(self.decode, "decode")
                    flags = ["--low-memory"] if low_memory else []
                    output = self.run_worker("--variant", variant, "--size", spec, *flags)
                    self.core.device_info.assert_called_once_with()
                    capped_sizes = low_memory_sizes if low_memory else default_sizes
                    if variant == "turbo":
                        capped_sizes = (512,)
                    expected = [call.limit(cap)] if width in capped_sizes else []
                    expected.append(call.build(self.root, variant))
                    if (
                        variant == "turbo"
                        and width in (1024, 2048)
                        and low_memory
                        and 1024 in low_memory_sizes
                    ):
                        expected.extend(
                            [call.limit(cap), call.decode(self.latent), call.limit(previous)]
                        )
                    else:
                        expected.append(call.decode(self.latent))
                    self.assertEqual(order.method_calls, expected)
                    self.model.generate_image.assert_called_with(
                        seed=371,
                        prompt="oak",
                        width=width,
                        height=height,
                        num_inference_steps=8 if variant == "turbo" else 40,
                    )
                    self.image.save.assert_called_once_with(output, format="PNG")

    def test_rejects_unknown_or_malformed_sizes_before_setup(self):
        for value in (
            "",
            "0",
            "-2048",
            "4096",
            "4096x4096",
            "3840x2160",
            "2048x2047",
            "2048x1536",
            "2752",
            "2752x",
            "x1536",
            "2752x1536x1",
            "2752X1536",
            "2752×1536",
            "x",
            "9" * 5000,
        ):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()) as err,
                self.assertRaises(SystemExit) as cm,
            ):
                self.run_worker("--size", value)
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("--size", err.getvalue())
        self.core.device_info.assert_not_called()
        self.build.assert_not_called()
        self.image.save.assert_not_called()

    def test_rejects_previous_rectangular_presets_before_setup(self):
        for variant, value in product(
            ("base", "turbo"),
            ("2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752"),
        ):
            with (
                self.subTest(variant=variant, value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as cm,
            ):
                self.run_worker("--variant", variant, "--size", value, "--low-memory")
            self.assertEqual(cm.exception.code, 2)
        self.core.device_info.assert_not_called()
        self.build.assert_not_called()
        self.image.save.assert_not_called()

    def test_checks_both_output_dimensions_before_writing(self):
        for size in ((2048, 1024), (1024, 2048), (1024, 1024)):
            with self.subTest(size=size):
                self.image.size = size
                self.saver.reset_mock()
                self.model.callbacks.reset_mock()
                self.model.vae.decode = self.decode
                with self.assertRaisesRegex(ValueError, "image dimensions"):
                    self.run_worker("--variant", "turbo", "--size", "2048")
        self.image.save.assert_not_called()

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
                for flags in ([], ["--low-memory"]):
                    with self.assertRaisesRegex(RuntimeError, "M5 GPU and macOS 26.3"):
                        self.run_worker("--size", "1024", *flags)
        self.build.assert_not_called()
        self.core.set_cache_limit.assert_not_called()

    def test_accepts_m5_family(self):
        self.core.device_info.return_value = {"device_name": "Apple M5 Max"}
        self.run_worker()
        self.image.save.assert_called_once()

    def test_interpreter_probe_only_prints_python_path(self):
        script = self.worker.__file__
        with (
            patch.object(sys, "argv", [script, "--interpreter"]),
            contextlib.redirect_stdout(io.StringIO()) as output,
            self.assertRaises(SystemExit) as cm,
        ):
            runpy.run_path(script, run_name="__main__")
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(output.getvalue(), sys.executable + "\n")
        self.core.device_info.assert_called_once_with()
        self.core.set_cache_limit.assert_not_called()

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

    def test_capped_decode_restores_previous_limit_on_errors(self):
        cap = 512 * 1024**2
        for previous, latent, result, error, calls in (
            (0, self.tensor([math.nan]), self.pixels, ValueError, 0),
            (12345, self.latent, self.tensor([math.inf]), ValueError, 1),
            (cap, self.latent, RuntimeError("decode failed"), RuntimeError, 1),
            (8 * 1024**3, self.latent, KeyboardInterrupt(), KeyboardInterrupt, 1),
        ):
            with self.subTest(previous=previous, error=error):
                self.core.set_cache_limit.reset_mock()
                self.core.set_cache_limit.return_value = previous
                decode = Mock(side_effect=[result])
                with self.assertRaises(error):
                    self.worker.capped_decode(decode, latent)
                self.assertEqual(decode.call_count, calls)
                self.assertEqual(
                    self.core.set_cache_limit.call_args_list, [call(cap), call(previous)]
                )

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
