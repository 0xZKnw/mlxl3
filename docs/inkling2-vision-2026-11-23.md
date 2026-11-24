# Inkling 2 Mini : entrée image dans MLXL3 (2026-11-23)

Entrée de journal : `FEAT-2026-11-23-INK2M-VISION` dans `opti.md`. Branche `claude/inkling2-vision` (base `origin/main`
`d87a9ff8d`), code local non commité au moment de ce rapport.

## Checkpoint

`0xzknw/Inkling-2-Mini-EXL3-3.33bpw` révision `deac6c50ece340a4096a0e26f86664ed50cf0274` : `model-vision.safetensors`
(422 tenseurs `model.visual.*` BF16, 636 563 512 o, SHA-256 `2d1341e9933ac4821c9b1665892f8f7bf55acde532e5704dc5dd0f5f804e9027`),
copiés à l'octet de la révision `0a4558409` ; les quatre shards du modèle de langage et `mtp.safetensors` sont ceux de
`dde31cd0`. L'index et `quantization_config.json` retrouvent les entrées vision. Contrôle statique seulement
(`vision_report.json` : empreinte de chaque tenseur).

## Ce que fait le moteur

- **Prétraitement** (`native/src/vision_input.rs`) : comme `Inkling2ImageProcessor` (Transformers 5.24.0.dev0) : résolution
  native, tuiles de 40 px (un token chacune), dernières tuiles complétées à −1 sur l'échelle 0–255, valeurs
  `(x − 127,5) / 127,5`, transparence composée sur le damier gris 8 px (204/153). Choix propre à MLXL3 (la référence ne
  redimensionne pas) : une image au-delà de `MLXL3_INKLING_IMAGE_MAX_PIXELS` (défaut 1920 × 1080, 1 296 tokens) ou d'un côté
  de 4 096 px est réduite (Catmull-Rom, proportions gardées). Budget de conversation inchangé (8 images, 16,7 Mpx, 16 384 tokens).
- **Prompt** : le gabarit d'Inkling rend chaque partie du message à part ; une image devient
  `<|content_image|><|unused_200054|><|end_message|>`. Le moteur garde donc les parties d'un message qui contient une image,
  puis répète le placeholder une fois par token d'image après le gabarit (comme `Inkling2Processor.replace_image_token`). Un
  placeholder écrit dans le texte ou absent fait échouer la requête. Les messages sans image sont aplatis comme avant.
- **Encodeur** (`native/src/inkling2_vision.rs`) : `Inkling2VisionModel` (23 couches, largeur 1024, 16 têtes, MLP × 4) :
  jeton de classe par image, positions apprises 32 × 32 rééchantillonnées en bilinéaire (`align_corners=False`, bords
  bloqués), RoPE 2D sur paires de canaux entrelacées (colonnes puis lignes, positions + 1, classe en (0, 0)), échelle de
  couche, puis 2 × 2 patchs fusionnés et projetés (4096 → GELU → 2304, sans biais). Chargé à la première image (0,64 Go de
  poids BF16 résidents). **Activations en FP32** (voir les mesures), LayerNorm, RoPE et rééchantillonnage en FP32.
- **Modèle de langage** (`native/src/inkling2.rs`) : `prepare_images` encode les images et associe chaque série de
  placeholders du prompt à son image (nombre et longueurs vérifiés) ; `run` remplace les lignes d'embedding de ces positions,
  y compris quand un morceau de prefill coupe une image ; `reset` les efface. Un prompt sans image suit exactement le même
  chemin qu'avant. DSpark : `prefill_with_images`. Pas de cache de prompt quand la conversation contient une image (ni
  reprise ni sauvegarde) ; les images de l'historique sont réencodées à chaque tour.
- **App Mac** : images acceptées pour `inkling2_loop` quand le moteur annonce `images_supported` (vrai seulement si le
  checkpoint contient l'encodeur) ; révision du catalogue `deac6c50`.

## Oracle et mesures

Oracle : `benchmarks/inkling2_vision_oracle.py` (Transformers 5.24.0.dev0, torch 2.10.0, torchvision 0.25.0, CPU), image
210 × 130 (dégradé, carré rouge, disque bleu ; grille 4 × 6 = 24 tokens), prompt de 52 tokens. La référence BF16 est à
0,01886 (L2 relatif) de la référence FP32.

Mac M5 24 Go, MLX 0.32.2, build release, encodeur seul :

| Contrôle | Résultat | Critère |
|---|---|---|
| Valeurs de pixels (115 200) | identiques au bit | égalité (fixé avant) |
| Tokens du prompt (52) | identiques | égalité (fixé avant) |
| Encodeur FP32, matmuls FP32 exacts, vs FP32 de référence | 5e-6 | < 1e-4 (fixé avant) : atteint |
| Encodeur BF16 vs BF16 de référence | 0,02282 | < 0,02 (fixé avant) : **échec** |
| Encodeur BF16 (noyaux M5) vs FP32 | 0,02411 | ≤ 1,25 × 0,01886 (révisé après coup) : échec |
| **Défaut : activations FP32 (noyaux M5) vs FP32** | **0,00398** | ≤ 1,25 × 0,01886 (révisé après coup) : atteint |

Constat : sur M5 (`applegpu_g17g`), MLX 0.32.2 calcule les matmuls FP32 avec une précision réduite (7,3–7,7e-4 d'erreur
relative sur un seul produit, contre 3,8–5,7e-7 avec `MLX_METAL_GPU_ARCH=applegpu_g16s` ou sur CPU). Le test d'exactitude
le mesure et ne conclut que si les matmuls FP32 sont exacts.

Temps de l'encodeur pour une image 1920 × 1080 (1 296 tokens, 5 184 patchs, à chaud, 3 répétitions, noyaux M5) : activations
FP32 1,49–1,51 s, BF16 1,23–1,27 s. Le FP32 coûte ≈ 0,26 s et reste 6 fois plus proche du résultat exact : il est le défaut.

Mutations temporaires (fichier restauré, empreinte identique) : RoPE lignes/colonnes inversées → 0,87 et test CPU en échec ;
interpolation `align_corners=True` → 0,25 et test CPU en échec.

## Modèle de langage et bout en bout

- Test GPU `image_features_reach_the_inkling_language_model` (seul en mémoire) : logits image vs placeholders 0,0818 (> 0,05) ;
  prefill en morceaux de 7 tokens coupant l'image vs 128 : 0,0014, même argmax ; texte seul identique au bit avant et après
  un prompt image ; prompts incohérents refusés.
- CLI `mlxl3-rs run --image` (greedy) : image « MLXL3 sees / 42 + 17 = ? » relue exactement, réponse 59.
- Bridge, DSpark, greedy : description juste de l'image des formes (dégradé et ses quatre coins, carré rouge à gauche, disque
  bleu à droite). Bridge, DSpark, T = 1, deux tours avec l'image dans l'historique : « 59 ».
- Preuves : `docs/measurements/inkling2-vision-20261123/`.

## Tests

- CPU (`cargo test --lib`) : patchs, ordre, remplissage, damier, plafonds, admission avant allocation, parties de message,
  expansion des placeholders, séries de placeholders, taps d'interpolation (oracle en forme fermée), angles RoPE, parties
  acceptées par le tokenizer.
- GPU, ignorés par défaut, à lancer seuls :
  `MLXL3_INKLING_MODEL=<dossier> MLXL3_INKLING_VISION_ORACLE=<json> cargo test --release --no-default-features --features
  mlx,chat --lib inkling2_vision::gpu_tests -- --ignored --test-threads=1 --nocapture` ; l'exactitude FP32 demande
  `MLX_METAL_GPU_ARCH=applegpu_g16s` sur M5. `image_features_reach_the_inkling_language_model` charge le modèle de langage
  (13 Go) : aucun autre modèle ne doit être chargé.

Résultats des suites (2026-11-23) : Rust `mlx,chat` lib 586 passés / 351 ignorés, bin 68 / 14 ignorés ; Swift `image-input`
et `workspace-features` (291) passés avec le SDK 26.5 ; mutations RoPE, interpolation et garde Swift détectées.

## Limites

- Fidélité mesurée sur une image synthétique ; aucun benchmark de tâches visuelles.
- Plafond de 1920 × 1080 par défaut : une image plus grande perd des détails par rapport à la référence (qui lit toute
  image en résolution native).
- Les images de l'historique sont réencodées à chaque tour et désactivent le cache de prompt.
