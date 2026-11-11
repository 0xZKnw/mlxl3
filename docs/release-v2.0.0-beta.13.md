# MLXL3 Desktop 2.0.0-beta.13

Desktop prerelease **v2.0.0-beta.13**, internal version **2.0.0 / build 44**.
The bundled MLX engine remains **2.0.5**; no separate engine release is published.

## Changes

- The recommended LFM3 and LFM3-VL cards open the new standalone 8-bpw
  repositories. In each model's details, choose **4**, **4.49** or **8 bpw**;
  MLXL3 opens the corresponding repository and its pinned revision. Heretic NX
  remains available at **8 bpw**, the only accepted Heretic NX EXL3 release.
- Each LFM3 quantization is now in a separate Hugging Face repository with its
  configuration and weights at the root. Original checkpoint bytes and
  validation evidence have been retained. The former combined repositories are
  being retired, so update Desktop before using its LFM3 recommended cards.
- The legacy LFM3-VL 4-bpw checkpoint has uncalibrated vision and is not
  recommended; the 4.49-bpw VL checkpoint missed its initial fidelity gate and
  has not met the stricter native feature-parity threshold. Their model cards
  retain these limits. The 8-bpw VL checkpoint uses calibrated vision.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Enable the Desktop beta
channel in Settings or install the DMG manually. The application is ad-hoc
signed and not notarized. Models and conversations are not bundled.

## Validation and known limits

The model selection and download path was exercised with an isolated engine
fixture, including switching between the six text and vision variants, pinning
revisions, rejecting invalid choices and downloading a root checkpoint. The
published MLX engine also resolved all seven Hugging Face repositories and
recognized their root EXL3 checkpoints. The interface was rendered at 480 and
840 points. Source model files and stored validation reports were checked
against their original Hub file identities and weight SHA-256 values.

Hugging Face's public download number counts requests for selected files; it is
not a count of unique users or every direct weight download. Each new repository
has a root `config.json`, but a counter increase was not yet observed at release
time. New repositories do not inherit historical download counts. A real model
download and the quantization picker in an installed application were not
manually requalified. No Swift source-level formal proof or new numerical
quality/performance claim is made. Source and artifact identities accompany the
release assets.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the request and model
publications; [Liquid AI](https://huggingface.co/LiquidAI),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3),
[Apple MLX](https://github.com/ml-explore/mlx) and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for the underlying models,
format, runtime and components.
