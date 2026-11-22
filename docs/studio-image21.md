# Generate an image on Apple M5

In MLXL3 Studio, open **Generate an image** in the sidebar. Select the local EXL3
denoiser folder containing `manifest.json`, describe the image, then click
**Generate**. The seed can be reused. **Show PNG** reveals the saved image in Finder.
Start with 512 × 512 and 40 steps. The physical validation used an M5 with 24 GiB of memory.

The first launch requires internet access and uv installed in ~/.local/bin,
/opt/homebrew/bin or /usr/local/bin. Studio sets up an isolated Python 3.12 environment.
The tokenizer, Q4 text encoder and VAE come from a pinned snapshot of the
[MLX Community auxiliary weights](https://huggingface.co/mlx-community/Qwen-Image-2.1-mflux-q4/tree/746a58556820933a2df5c75887a2570f1ad200c0).
Studio reads the 33 local denoiser shards, checks their SHA-256 digests against the manifest
and runs all 224 EXL3 projections using the application's Metal tensor kernel.
The package contains 3.56 GB of denoiser weights. Additional disk space is required
for the Python environment and auxiliary components.

This version supports text-to-image with the base Qwen-Image-2.1 checkpoint.
Generation uses the [MFLUX image pipeline](https://github.com/mflux-community/mflux/tree/d39c9639e6ac22407a725432cc221a83fab19f1b/src/mflux/models/qwen21).
Built with Qwen. Model use is subject to the [upstream research license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE).
