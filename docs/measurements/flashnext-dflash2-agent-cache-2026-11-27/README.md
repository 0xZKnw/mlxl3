# Reprise du cache Agent avec DFlash2 sur Flash Next

Essai `FLASHNEXT-AGENT-DFLASH2-LIVE-01` du 27 novembre 2026. Base `1f967fe3c6daa98a5db071d869933de1efdbbfdf` de la PR #215, avec le diff de reprise local. Mac M5 24 Gio, macOS 27.3 (26C5047e). Checkpoint `Qwen3.8-Flash-Next-exl3-84147e95` à 2,05 bpw. Drafter `incoai/Qwen3.8-Flash-Next-DFlash2@a22a1744e85c17364dcc54a6a420af7135d4c7f6`. Moteur MLX+chat 2.0.39 en release, bibliothèques MLX 0.32.2 du moteur installé. SHA-256 du moteur `bc847b9d4d264b67711761b2582594d4c4d6813840fad7017c8287a2d17b02a0`.

Le pont réutilise les états déjà présents de la cible et du drafter pour le suffixe du prompt si leurs positions concordent et si les IDs retenus forment un préfixe exact. Il reste au préremplissage complet en cas de divergence, de préfixe réécrit ou de lignes spéculatives encore non délivrées. Le client de test demande explicitement la reprise via le pont réel. La capacité Flash Next reste à annoncer dans Studio.

## Correction

Sur le GPU du M5, après un bloc de vérification avec deux suggestions et une acceptation, le candidat repart de 15 tokens sur un prompt de 29. Les 136 éléments d’état de la cible et les 248 320 logits finis sont identiques, avec le même argmax et la même suite de quatre tokens. Les dix tenseurs du cache du drafter sont comparés sur leurs lignes occupées, avec une L2 relative maximale de 0,00364 (seuil 0,05). [Oracle GPU](gpu-oracle.log)

La garde CPU et la décision de préfixe ont passé leurs harnais Kani, respectivement avec 38 et 201 obligations sans échec. Le second harnais explore des invites de 0 à 7 IDs et marque une obligation inatteignable. Les deux obtiennent 7 couvertures sur 7. Les 5 tests directs de cache, 6 tests du nouveau harnais et 5 tests du scénario historique passent. Ces contrôles CPU complètent le test Metal. [Kani de la garde](dflash2-live-guard-kani.log.gz) · [Kani du préfixe](dflash2-live-prefix-kani.log.gz)

## Reproduire

Configurer un registre avec l’entrée du checkpoint local. Préparer les bibliothèques du moteur installé dans `ENGINE_BUNDLE`, un interpréteur Python conservant `DYLD_LIBRARY_PATH`, le binaire release dans `ENGINE`, le drafter épinglé dans `DRAFT` et l’exécutable Studio dans `HELPER`. Chaque passage utilise un nouveau processus. Le script fixe le cache SSD à 3 Gio, DFlash2 mode 2 à largeur 2, le Q4 et le cache circulaire du drafter, sans adaptation ni MTP. Il fixe le contexte à 4 096 et le budget à 192 tokens, à température nulle et effort `none`.

```sh
export DYLD_LIBRARY_PATH="$ENGINE_BUNDLE"
export MLXL3_METALLIB_PATH="$ENGINE_BUNDLE/mlx.metallib"
export MLXL3_PROMPT_CACHE_TRACE=1
trial=docs/measurements/flashnext-dflash2-agent-cache-2026-11-27/dflash_agent_trial.py
"$PYTHON" -B "$trial" --engine "$ENGINE" --helper "$HELPER" --registry "$REGISTRY" \
  --model Qwen3.8-Flash-Next-exl3-84147e95 --dflash-draft "$DRAFT" \
  --ready-timeout 180 --task-timeout 900 --label A1 --output /tmp/A1.json
```

Pour le candidat, ajouter `--live-cache` et choisir une autre étiquette et un autre fichier. Le harnais utilise les trois mêmes lectures de fichiers que la #215 et compare les résultats des outils, le contexte et le hash final. Le temps de tâche commence après l’initialisation du serveur. Il inclut le chargement du drafter à la première requête. Le pic principal est l’empreinte physique maximale du processus depuis son démarrage, avec le pic MLX relevé séparément.

Pour rejouer l’oracle isolé, construire les tests release avec `--no-default-features --features mlx,chat`, définir `MLXL3_FLASH_MODEL` et `MLXL3_FLASH_DRAFT` vers les checkpoints, puis exécuter `qwen4::draft::runtime::tests::dflash2_agent_live_tool_prefix_matches_cold_prefill` avec `--exact --ignored --nocapture --test-threads=1`. Fixer `MLXL3_SSD_CACHE_GIB=3`, `MLXL3_FLASH_DRAFT_ADAPTIVE=0`, `MLXL3_FLASH_DRAFT_WIDTH=2`, `MLXL3_FLASH_DRAFT_Q4=1` et `MLXL3_FLASH_DRAFT_RING=1`.

## Série ABBA sur la tâche réelle

| Passage | Temps de tâche (s) | Préfill (s) | Decode (s) | Tokens de préfill évalués | Pic processus (Gio) | Pic MLX (Gio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [A1](A1.json) · DFlash2 seul | 477,29 | 453,46 | 22,77 | 1 859 | 7,8496 | 6,9254 |
| [B1](B1.json) · avec reprise | 121,88 | 102,06 | 19,21 | 459 | 7,8303 | 6,9270 |
| [B2](B2.json) · avec reprise | 115,17 | 100,49 | 14,14 | 459 | 7,8220 | 6,9272 |
| [A2](A2.json) · DFlash2 seul | 422,03 | 405,75 | 15,70 | 1 859 | 7,8060 | 6,9269 |

Les quatre passages ont les mêmes trois appels et contenus d’outils, le même contexte final et la réponse `TOTAL=18`, avec 85 tokens émis et le hash final `aaee49135bd200dc`. DFlash2 accepte 57 propositions sur 58 à chaque passage. Avec la reprise, 416, 467 puis 517 tokens sont réutilisés, soit 1 400 tokens de préfill évités sur 1 859 (−75,3 %).

La tâche médiane passe de 449,66 à 118,53 s (3,79×). Le préfill après les trois appels passe de 326,91 à 14,35 s au total (médianes). A1 démarre sur batterie puis bascule sur secteur. Les trois autres démarrent et finissent sur secteur. B2 et A2, la paire exécutée dans la même fenêtre avec des premiers préremplissages comparables (86,26 et 82,30 s), donnent 115,17 et 422,03 s. Ces durées portent sur cette tâche et ces réglages.

Le plus haut pic processus des deux témoins est de 7,8496 Gio. Le plus haut pic des deux reprises est de 7,8303 Gio, ce qui satisfait le seuil prévu. L’écart des médianes est de −1,68 Mio et la paire B2/A2 varie de +16,42 Mio. Le pic MLX maximal monte de 0,48 Mio. Ces écarts indiquent un usage RAM comparable sur cette série. [Comparaison](summary.json)

Le build release, le format Rust, Clippy strict sur tous les targets MLX+chat et le test de portée du cache dans le pont passent. Le comparateur ABBA reproduit le même résumé après retrait des identifiants de processus et des chemins locaux. Les contrôles GitHub ne sont pas lancés pour ce diff local. [Test du pont](bridge-scope.log) · [Build et lint](clippy.log)

## Activation dans le vrai Agent Studio

Studio active automatiquement la reprise sur une tâche Agent lorsque le moteur chargé annonce la capacité Flash Next et que le même envoi sélectionne DFlash2. Le pont contrôle encore les deux états et le préfixe avant chaque reprise.

Le test d’intégration utilise le vrai `StudioModel.send()`, le pont et le sandbox natif en lecture seule, avec les trois fichiers du comparatif. Le client test est compilé sous Swift 6 strict contre les sources Studio. Une copie transparente de la sortie JSON du processus natif relève les statistiques de chaque tour. Le moteur, les modèles et le profil DFlash2 sont identifiés dans le rapport.

Le M5 a terminé en **175,75 s**, avec des lectures identiques dans le même ordre, `TOTAL=18` et le hash final `aaee49135bd200dc`. Les trois reprises ont réutilisé **733, 784 et 834 tokens**, soit 2 351 tokens réutilisés et 776 évalués sur 3 127. Pic processus 7,873 Gio, pic MLX 6,949 Gio. Le premier prompt contient 706 tokens avec les consignes et le catalogue de Studio. Les temps et la RAM comparatifs viennent de la série ABBA avec les mêmes prompts dans les deux bras. [Résultat Studio](studio-agent.json)

La vérification `studio-modes-check` passe 2 543 assertions, dont les sept requêtes de compatibilité du nouveau chemin. Le moteur release passe Clippy strict et le harnais Kani de la garde passe à nouveau ses 38 obligations et sept couvertures. [Test Studio](studio-modes-check.log) · [Kani](studio-native-kani.log.gz)
