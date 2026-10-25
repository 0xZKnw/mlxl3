# Essai19 : préparation conditionnelle et sources Sushi

Le [rapport](../../lfm25-calibration-v2.md) explique la portée.
`verification.json` identifie les sources modifiées et les artefacts de cet essai.
Le HEAD de départ estc9aa43a ; il ne contient pas ces modifications locales.

- `focused-tests.log` :50passés, deux vrais bridges natifs1.4.4.
- `full-tests.log` :556passés/1ignoré (modèle Ling local absent).
- `crosshair.log` :3conditions confirmées, alignement entier uniquement.
- `mutation.log` : l'oracle détecte la capture tous-tokens incorrecte.
- `first-tests.log`, `hardlink-regression.log`, `oracle-empty-gemm-crash.log`,
  `crosshair-missing.log` : échecs initiaux conservés ; détails et corrections
  dansopti19. `routed-fixed-tests.log` conserve la relance intermédiaire.
- `sushi-expert-header.json` : métadonnées et en-tête du shard expert, requête
  HTTPRange bornée ; ni poids complets ni convertisseur privé téléchargés.
- `protection-size.json` : comptage exact des paramètres et estimations de
  payload K6/K8, pas une mesure qualité.

Aucune quantification complète24B v2 n'a été lancée pendant19. L'autorisation
ultérieure et les expériences CUDA/qualité sont suivies séparément sous20.

Les trois logs à espaces terminaux sont conservés sans perte en `.log.gz` ;
`compressed-logs.json` donne leurs noms originaux, tailles etSHA256 après
décompression. Les anciens hashes de `verification.json` restent ceux des
octets originaux. La révision finale du15octobre, incluant les gardes des
caches0octet/NPZ, est suivie dans les preuves de20 (52tests ciblés/570complets).
