# Inkling 2 vision — lignes de résultats (2026-11-23, Mac M5 24 Go, MLX 0.32.2, release)

Fixture de l'oracle : `benchmarks/inkling2_vision_oracle.py MODEL_DIR OUT_DIR` (4,4 Mo, non copiée ; empreinte dans
`oracle-fixture.sha256`, résumé dans `oracle-summary.json`). Modèle : `models/Inkling-2-Mini-EXL3-R3-vision` (= révision HF
`deac6c50`).

## Encodeur, `MLX_METAL_GPU_ARCH=applegpu_g16s` (matmuls FP32 exacts)

    GPU FP32 matmul relative L2 3.83e-7
    pixel values: 115200 values, max |diff| 0
    engine Float32 vs features_f32: relative L2 0.000005, worst token 0.000009
    engine Float32 vs features_bf16: relative L2 0.018859, worst token 0.052598
    engine Float32: 1920 x 1080 picture encoded in [2.615, 2.587, 2.599] s (warm)
    engine BFloat16 vs features_f32: relative L2 0.026467, worst token 0.074576
    engine BFloat16 vs features_bf16: relative L2 0.023405, worst token 0.064103
    engine BFloat16: 1920 x 1080 picture encoded in [2.428, 2.438, 2.429] s (warm)
    default vs FP32 reference: engine 0.00001; reference BF16 vs FP32 0.01886; engine BF16 activations vs FP32 0.02647 (not used)
    test result: ok

## Encodeur, noyaux M5 par défaut

    GPU FP32 matmul relative L2 7.26e-4
    pixel values: 115200 values, max |diff| 0
    engine Float32 vs features_f32: relative L2 0.003977, worst token 0.009469
    engine Float32 vs features_bf16: relative L2 0.018998, worst token 0.055021
    engine Float32: 1920 x 1080 picture encoded in [1.500, 1.510, 1.492] s (warm)
    engine BFloat16 vs features_f32: relative L2 0.024109, worst token 0.078736
    engine BFloat16 vs features_bf16: relative L2 0.022818, worst token 0.069633
    engine BFloat16: 1920 x 1080 picture encoded in [1.267, 1.241, 1.227] s (warm)
    NOT CHECKED: FP32 exactness (this GPU's FP32 matmuls are reduced precision; rerun with MLX_METAL_GPU_ARCH=applegpu_g16s on Apple M5)
    default vs FP32 reference: engine 0.00398; reference BF16 vs FP32 0.01886; engine BF16 activations vs FP32 0.02411 (not used)
    test result: ok

Premier essai (encodeur en BF16, critère fixé avant : < 0,02 contre la référence BF16) : `features_bf16: relative L2 0.02282,
worst token 0.06963` → échec, conservé.

## Mutations (fichier restauré ensuite, empreinte identique)

- RoPE `[row, column]` au lieu de `[column, row]` : `engine Float32 vs features_f32: relative L2 0.870372` ; test CPU
  `rotary_angles_put_columns_before_rows` en échec.
- Interpolation `align_corners=True` : `relative L2 0.246952` ; test CPU `position_taps_reproduce_bilinear_resampling` en échec.

## Modèle de langage (seul en mémoire)

    pixel values: 115200 values, max |diff| 0
    logits: image vs placeholders relative L2 0.0818, 7-token chunks vs 128 0.00140, argmax 200001 / 200001 / 200001
    test result: ok. 1 passed

## Bout en bout

- CLI `mlxl3-rs run MODEL --image text_test.png --prompt "Read the text in this image exactly, then answer the question it
  asks." --max-tokens 700` (greedy) : réflexion « The image shows: "MLXL3 sees" "42 + 17 = ?" … Answer: 59 » ; arrêt à 700
  tokens ; `[prefill 161.4 tok/s · decode 23.3 tok/s · TTFT 611 ms · 700 tokens]`.
- Bridge : `bridge_shapes_dspark.json` (DSpark, greedy, effort none), `bridge_text_sampled_2turns.json` (DSpark, T = 1,
  deux tours, réponse « 59 »). Pilote : `bridge_image.py`.
