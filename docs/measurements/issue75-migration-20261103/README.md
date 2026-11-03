# Issue 75 — migration Flash Next EXL3 sur M5

## Verdict

MLX reste le backend du moteur. La tranche Metal facultative conserve les sorties du contrôle et réduit le nombre d’opérations de construction du graphe. Ses médianes AR128 sont de 10,27 tok/s MAIN et 9,96 HOLD, contre 10,34 et 10,04 sur le chemin existant. Conserver le prototype OFF, accessible avec MLXL3_FLASH_NATIVE_PREPARE=1 sur le matériel qualifié. Les huit bras, la dispersion et la révision du seuil à la demande de l’utilisateur sont documentés. [Comparaison AR128](ar128.md) · [Contrats, décision et suites QSA/KV/spéculatif](decisions.md).

## Provenance et reproduction

Worktree issu de main b9d5cbdce50aa98adfee560e06da75614e730bd3. Référence historique, PR 72 et 74 et versions d’AR-QUICK réconciliées séparément. Référence de migration recompilée avec le wheel MLX 0.32.2. Checkpoint Flash Next EXL3 2,05 bpw révision 65c895314393431c09050b2e04e250836b3a6eb4. M5, 24 Go, macOS 27.2 (26B123). Secteur et compteurs thermiques disponibles dans les plans par processus. Les capteurs de puissance/J par token et la température interne du GPU n’ont pas été échantillonnés. [Historique et patches](../issue75-lot0-20261103/README.md) · [Empreintes du checkpoint](checkpoint.json) · [Sources finales et binaires de chaque étape](source-manifest.json).

- Référence stock compilée depuis la base. Pour recréer les fixtures avec la même base, installer le source de test archivé à la place de native/src/qwen4/profile.rs puis construire uniquement le libtest. Pour R1, appliquer r1-tracked.patch sur la base et remettre les deux nouveaux fichiers natifs depuis r1-policy.rs.txt et r1-flash_prepare.h.txt. R2 a son patch et ses nouveaux fichiers archivés. R3 correspond aux sources finales recensées au manifeste. [Fixture de référence](ref-fixture-profile.rs.txt) · [Patch chronométré R1](r1-tracked.patch) · [Identités R1](candidate-r1.sha256) · [Identités R3](r3-binaries.sha256).
- Construire le CLI avec `cargo build --locked --release --features mlx,chat --bin mlxl3-rs` et le test avec `cargo test --locked --release --features mlx,chat --lib --no-run`. MLXL3_MLX_ROOT pointe vers la racine du wheel, DEVELOPER_DIR vers le Xcode du manifeste. Les fichiers `*-plan.json` contiennent argv, délai, options et conditions exactes. Les `*-identity.json` fixent les exécutables, superviseurs et dépendances lus. Les `*-plan-result.json` conservent la fin du processus, maxima et snapshots. Les scripts d’orchestration archivés s’exécutent à l’emplacement build/issue75 dans ce worktree avec les chemins historiques référencés. Le répertoire de fixtures du modèle doit être recréé dans un emplacement neuf. Les ancrages système sont propres à chaque campagne. [Ancre de campagne](anchor.json) · [Pilote supervisé](guard-v6.py.txt).
- Les fixtures enregistrées dans build/issue75/ar-fixtures comprennent 740 fichiers plus le marqueur de complétude. Pour MAIN/HOLD à la fin du prompt et après un token greedy, chacune conserve 136 entrées de session, un tenseur de 248 320 logits et 48 routes de 10 experts. Les checksums permettent de vérifier l’intégralité du contenu régénéré. Le test compare le contenu et l’ordre de chaque entrée. Un seul bit modifié dans la copie d’un tenseur de logits est détecté. [Inventaire SHA-256](fixtures.sha256) · [Log du témoin négatif](r2-negative.log) · [Oracle indépendant](r2-oracle.log).
- Le budget expert contrôlé de 3 Gio correspond à 53 × 48 slots, 3 174 942 528 octets logiques, 3 177 215 808 de backings mesurés à ready. Allocateur 64 Mio, quatre lecteurs, un modèle à la fois. Les fixtures, le profilage instrumenté, le débit, la mémoire et le contexte long sont mesurés séparément. L’ancre n’a pas été réinitialisée pendant les runs. Les limites de garde cumulées sont 32 Mio écrits et 256 Mio lus. [Témoin de migration AR128](ref-ar128/report.json) · [Réduction réelle des banques](r3-memory.log) · [Refus, annulation et réutilisation du bridge](r3-bridge-errors.log).

## Contexte long

Le contrôle long a franchi la première sélection de 513 blocs avec les mêmes 2050 entrées, 8 sorties greedy identiques et le hash 8333726d17995c0f. TTFT base/R3 196.57/195.28 s, footprint en fin de génération 7.213/7.213 Go. Chaque bras a exécuté une seule conversation ; aucune estimation de gain. [Base](long-base-events.json) · [R3](long-r3-events.json).


Les quatre traces détaillées du profilage sont compressées avec gzip. Le SHA-256 des octets décompressés est vérifié et le nom original est conservé. [Reçu de compression](ref-profile-data/compression.json).

## CI et portée des preuves

Le workflow Native Rust checks couvre les PR et les pushes sur main. Les nouveaux harnais de preuve figurent dans l’inventaire Kani. Le job macOS compile la variante MLX/Metal livrée. L’inventaire Python du test physique est collecté sur CI. Pour l’exécuter, il faut le checkpoint et un GPU physique. Sur le commit public de base, les jobs natifs macOS et Linux ont réussi. Kani flash, core et routes ont échoué, avec notamment sept timeouts déjà présents. Les shards allocator et speculative ont réussi. La CI de ce travail local requiert une publication séparée. [Run GitHub de la base](https://github.com/0xZKnw/mlxl3/actions/runs/41462426679).


Les répertoires et logs préfixés `final-` décrivent R4 (reprise du lecteur après déconnexion). R5 ajoute les compteurs du cache TLS et son nettoyage à une frontière idle. Les fichiers préfixés `r5-` décrivent cette dernière version. Les étapes R1/R2/R3 conservent leurs identités et leurs résultats distincts.
