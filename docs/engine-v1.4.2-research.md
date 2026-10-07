# Moteur v1.4.2 — revue Sushi et TensorFold du 7 octobre 2026

PR de travail : [#26](https://github.com/0xZKnw/mlxl3/pull/26), draft ouverte. Aucun merge, release ou remplacement de l'app installée. Base `5de518e0fbe7d3367416b9bc5999c7bd1f92c74d` ; première étape déjà poussée `3ed683617e5090a179fb4888c3db03f8cb79b5fd`.

| Source et mécanisme | Décision dans cette PR |
| --- | --- |
| [Sushi v1.2.0 / ae63f880](https://github.com/beamivalice/sushi/tree/ae63f880550fa52c0ea4549329ecc7a76e41e10d), continuations du contexte | Implémentation Rust indépendante conservée **opt-in/default OFF** (`MLXL3_MTP_LOOKUP=1`). Recherche bornée à 1 024 IDs, suffixe de 8 IDs, avec saut de ligne dans les 7 IDs engagés selon le vrai tokenizer, propositions vérifiées par la cible et caches réparés ; retour au draft neural après refus complet. |
| Sushi, nouveaux lanes EXL3 et grid de prefill | Pas de port direct : le chemin étudié exige BF16/MCG/clamp GLM ; grid T2048/H4096/top8/288 experts. Qwen35 utilise notre chemin EXL3 FP16 sans ce clamp. Supprimer ces gardes changerait le contrat numérique. |
| [TensorFold 0.6.6 / cb2ebf054](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248), soumission progressive des niveaux MTP | Prototype minimal CABI/Array/draft_chain testé puis **retiré** : pas de gain D2 fiable. Transfert final et vérification cible restaient inchangés. |
| TensorFold, soumission de couches cible par groupes de 4 | Mécanisme identifié, **non prototypé/non mesuré** dans cette campagne. Ne pas confondre avec les kernels row-exact du chemin Mac affine/BF16 étudié. |
| TensorFold, profondeur adaptative par coûts et acceptation | Différé : Tune choisit déjà D0..3 par modèle/runtime. Un ordonnanceur par bloc demanderait un essai distinct et des contrats d'état/coût ; moteur multi-stream amont non copié. |

Les [11 sources empreintées et 7 mécanismes](measurements/engine-v1.4.2/sushi-tensorfold-deepening.json) précisent les chemins et licences. Aucun code/binaire externe exécuté et aucun débit amont transposé.

Le premier crible lookup stock copie avec warmup32 donnait +6,41% decode et −3,87% complet. Les confirmations BAAB avec warmup32 puis ABBA avec warmup128 dépassent les limites de dérive : **gain non reproduit**, aucune activation par défaut. Code non répétitif : −0,0028% decode/−0,0616% complet ; ces petits mouvements ne sont pas un gain établi. Le [rapport lookup](measurements/engine-v1.4.2/lookup-decision.json) conserve chaque campagne et ses conditions. Aucun débit dense ou compact+lookup mesuré ; ces chiffres ne s'ajoutent pas aux gains du compact draft contre 1.4.1.

Le crible TensorFold async D2 FR+code, warmup128/128 mesurés/2 répétitions/ABBA avait FR+0,258% et code−3,945% decode/+4,314% complet. Dérive code B+14,614% et contrôle FR B−3,325% : cela ne prouve pas une régression stable, mais ne justifie pas de retenir le prototype comme boost. Pas de confirmation/D3/compact ; [décision et sources du prototype](measurements/engine-v1.4.2/async-decision.json) conservées.

Les boosts compact MTP de la première étape restent ceux de leur protocole propre : code+8,20% decode, tuner D2+11,62..15,30%, projection+map 87,77 MiB supplémentaires. Le Q4 full avait régressé sur code ; le NT4 MCG6 était du bruit et a été retiré. Le correctif Send/Tune pendant checks/downloads d'updates reste inclus et nécessite un nouveau build Desktop.

## Contrôles et limites

- `cargo fmt --all --check`, Clippy strict MLX/chat et CPU/chat, build release **mlx,chat** : réussis ; [commandes/logs](measurements/engine-v1.4.2/phase-2-checks.json).
- `cargo test --locked --release --features mlx,chat` : **66 passés/58 ignorés**. `cargo test --locked --no-default-features --features chat` : **54 passés/2 ignorés**. Les tests GPU ignorés ne sont pas couverts par ces seuls totaux.
- GPU sélectionné avec `--exact --ignored --nocapture --test-threads=1`, seul : stock et compact MoE chacun 24 blocs/80 états/hidden/KV exacts et 37 completions/306 lookup ; dense 24 blocs/128 états puis 37 completions/6 lookup au budget de copie 1 024, zéro écho inline et préfixe 256 exacts. **Échec de couverture dense 256 conservé**, pas transformé en parité.
- `MLXL3_MTP_LOOKUP=1 python scripts/check-mtp-depths.py <engine> <MoE> <stockhead> --tune` : 43 requêtes/35 completions/5 annulations/2 erreurs attendues/1 Tune. D1..3/IDs/budgets/préfixe/récupération/bestD/lignes Tune et clé lookup1 contrôlés. `check-mtp-model-switch.py` : 36 completions dense→MoE→dense, head reuse/OFF allocations/échecs et processus réapés.
- `cargo kani --lib --no-default-features -Z unstable-options --harness-timeout 600s --export-json <report>` : **37/37 harnais**, 5 449 assertions passées/90 covers satisfaits/72 obligations internes ou préexistantes inatteignables, 0 échec/timeout, 477,916 s. [Détails](measurements/engine-v1.4.2/kani-phase-2-summary.json). Nouveau matcher : 20 IDs u32 symboliques/len0..20/largeur usize complète/ancre u32/filtre bool, unwind33, aucune assume, 5 covers ; frontière 1024/1025 testée déterministiquement, pas prouvée par Kani.
- Le decoder HF, Session MLX/Metal, allocations/concurrence et génération complète ne sont **pas formellement vérifiés**. Kani du prototype async sur la vraie variante mlx a rencontré un ICE ; CBMC sur le vrai pont préprocessé C++20 refuse libc++/SDK. Prototype retiré ; diagnostics/harnais conservés sans stub GPU.
- Première étape inchangée : Python 229 passés/4 skips (PonyEXL3 absent : 3 ; checkpoint Ling absent : 1), Swift6 strict et E2E Desktop réussis ; strict build global bloqué par dépréciations SwiftMath préexistantes. [Validation générale](measurements/engine-v1.4.2/validation.json).

Mutations latest-match/filtre lookup/NULL async détectées puis restaurées. Les échecs, essais non concluants et commandes brutes sont dans [l'archive de phase2](measurements/engine-v1.4.2/phase-2-raw.tar.gz) et son manifest ; extraire dans le répertoire de mesures. L'archive de phase1 est conservée. Les sources complètes du prototype non filtré ancien n'ont pas été conservées : empreintes et binaire restent locaux, logs archivés ; il est remplacé/non livré.

Les workflows PR/push existants couvrent Rust CPU/Kani/Python/Desktop sans credentials de production. Les 8 jobs de 3ed6836 passent ; les checks du prochain SHA seront inspectés après push et consignés dans la description de PR. La CI hébergée ne prouve pas les chemins GPU physiques.
