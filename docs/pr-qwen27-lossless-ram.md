# Réduire la RAM Qwen27 avec un stockage exact et des snapshots compacts

Base : `HENK0O/mlxl3:codex/retour-version-normale` (`8c08c4c`).
Branche proposée : `HENK0O/mlxl3:codex/qwen27-lossless-ram`.

Les embeddings Qwen27 sont stockés de façon réversible puis reconstruits avant
le cast MLX canonique. Les snapshots de préfill copient les vues qui retiennent
une allocation inutilement grande. Cela réduit la RAM processus de 806 à 867 Mo
sur le profil court mesuré en AB/BA, sans réduire la précision, le KV F16, le
contexte ou la réutilisation de préfixe. Le mode Économie de mémoire reste exclu.
La clé Tune distingue les deux formats de stockage.

Validation : logits/hidden/128 états bitexacts sur 12 positions, préfills de
24/257/8193 tokens ; 180 cas MTP, sessions, bridge, annulation/reprise et Tune.
Rust CPU 62 passés/3 ignorés ; livré mlx,chat 78 passés/69 ignorés, GPU sélectionné
séparément ; Python 338 passés/4 skips ; suite Desktop passée. Kani 46 harnais,
CBMC des helpers et ASan/UBSan passent. Ces preuves CPU restent bornées et ne
prouvent pas Metal ou l’application entière. Deux timeouts MTP initiaux sont
conservés ; le harnais final passe ses 180 cas sans changer le binaire moteur.

La CI ajoute sanitizers et CBMC. Les résultats locaux ne préjugent pas des runs
GitHub de cette branche. Aucun boost de vitesse ou gain RAM universel revendiqué.
Aucune installation ou release incluse. L’amont a évolué de 76 commits : cette PR
cible explicitement la version de retour validée, sans les intégrer.

[Rapport détaillé, protocole et preuves](qwen27-ram-implementation-20261016.md).
