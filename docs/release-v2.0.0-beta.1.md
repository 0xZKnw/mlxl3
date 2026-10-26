# MLXL3 Desktop 2.0.0-beta.1 — build 31

Cette préversion v2 rend **Flash Next EXL3 2.05bpw avec streaming SSD** testable dans le Desktop. Tag `v2.0.0-beta.1`, `prerelease=true`, `latest=false`. L'app et son moteur utilisent la version interne numérique **2.0.0**, build Desktop 31. Source du paquet : [`e04c6558959609443dda362f2e8509b998adaa6d`](https://github.com/0xZKnw/mlxl3/commit/e04c6558959609443dda362f2e8509b998adaa6d), issue de la [PR34](https://github.com/0xZKnw/mlxl3/pull/34).

## Nouveautés

- **Flash Next sur SSD :** les experts du checkpoint texte qualifié sont chargés vers des banques réutilisables par jusqu'à quatre lecteurs ; un cache borné conserve les experts résidents.
- **MTP1 intégré :** sélectionner MTP dans Génération utilise la tête du checkpoint Flash, sans tête externe. Les propositions sont vérifiées par la cible.
- **DFlash2 expérimental :** un draft local compatible peut être configuré avec conversion Q4 en mémoire, cache circulaire et vérification adaptative. MTP/DFlash2 sont des choix distincts et imposent le sampling greedy.
- **Statistiques :** propositions acceptées, blocs et profils du moteur, avec distinction des allocations MLX, de l'empreinte du processus et de sa compression.

Les fonctions 1.5 — conversations à branches, projets, documents, MCP, JSON, voix et export — sont conservées. La recherche reste lexicale, sans OCR.

## Correctifs et réglages

Le choix d'un **draft Flash local** est désormais envoyé au bridge résident ; le validateur legacy des autres familles ne le refuse plus avant chargement. Aucun téléchargement automatique de draft Flash ni Tune legacy Flash n'est ajouté. **Auto correspond au mode 1 adaptatif**, pas au B2 fixe mesuré en benchmark.

Le catalogue inclut désormais `ngram_embedding.safetensors` et `mtp_hyper_connection_mixer_patch.safetensors`. Leur omission rendait le modèle incomplet et sous-estimait son téléchargement de **26 253 280 720 octets**, soit 26,253 Go. Les variantes restent isolées par dossier et quantification ; aucun poids local existant n'est modifié.

Défauts : cache experts **automatique ≤ 6 Gio**, buffers MLX **512 Mio**, quatre lecteurs et overlap actifs. GPU Hot/Wide, cohortes uniques, vérificateur multi-token 63, prélecture prédictive et préplafonnement restent opt-in. Q4/ring/adaptatif sont actifs lorsqu'on choisit DFlash2 Flash. La RAM totale peut dépasser les caches annoncés.

Les corrections de logits, états, annulation/reprise et mémoire au chargement sont conservées. **K1 est exclu du moteur embarqué**, faute de gain réel ; [résultats négatifs, PR47](https://github.com/0xZKnw/mlxl3/pull/47). Flash reste textuel ; vision, autres variantes et DSpark Desktop ne sont pas qualifiés.

## Performances

Aucune accélération du DMG bêta n'est revendiquée. Repère historique conditionnel : environ 11 tok/s en DFlash2 B2/MTP1 sur MacBook Pro M5 standard, 10 cœurs GPU, 24 Go, MLX 0.32.2, EXL3 2.05bpw, secteur, MAIN/HOLD fixes, warmup 128 puis 128 tokens émis/127 décodés. Elles utilisent **3 Gio/53 slots et 64 Mio d'allocateur**, avec options expérimentales explicites ; ce ne sont pas les défauts de l'app.

Le débit varie avec le prompt, le contexte et les conditions du Mac. Aucun avantage universel DFlash2/MTP ni **15 tok/s** établi ; les gains de composants ne s'additionnent pas. [Protocoles et limites historiques](https://github.com/0xZKnw/mlxl3/blob/e04c6558959609443dda362f2e8509b998adaa6d/docs/flash-next-ssd-verifier-63-2026-10-21.md).

## Installation et compatibilité

Apple Silicon arm64, macOS 26.2+, MLX 0.32.2 ; signature ad-hoc, sans notarisation Apple. Télécharger explicitement le DMG bêta et l'ouvrir. Les modèles et le draft ne sont pas inclus.

Conserver une sauvegarde de l'historique et l'app stable avant essai. Les deux applications peuvent partager données et préférences : changer leur nom ne les isole pas. Le format d'historique 2 de 1.5 est conservé ; une migration depuis le format 1 empêche sa relecture par les anciens Desktop. Les sauvegardes tournantes peuvent aussi avoir migré. Les modèles ne sont pas déplacés.

Les canaux moteur/Desktop restent indépendants et les deux bêta sont non-latest ; aucun canal bêta ajouté à l'updater stable. **Si le moteur bêta est installé indépendamment, revenir au moteur embarqué avant la future stable 2.0.0**, dont le numéro interne sera identique.

## Vérification et limites

**228 tests CPU passés / 5 ignorés**, 7 tests Hub sous MLX réussis, format et Clippy strict CPU/MLX réussis. Régression catalogue et mutation MTP détectées, source restaurée. Contrôles Desktop : 8 cas Flash, 240 contexte, 114 assertions idle, 965 sampling ; 45 tests packageur réussis.

Kani sur la vraie sélection du catalogue atteint 180s sans verdict : **propriété non vérifiée**. Les preuves CPU bornées ne prouvent ni SwiftUI/Metal, ni concurrence ou génération entière. Swift reste non prouvé ; CrossHair indisponible. Autres Mac, contextes longs, qualité sur corpus, voix et MCP distants ne sont pas requalifiés.

**Les31contrôles du paquet passent : signatures, arm64, manifestes et moteur relocalisé. La suite Desktop complète passe avec les vrais paquets (installation, exécution, fallback et démontage), en366,27s,87sources inchangées et processus joints. À la demande de publication immédiate, aucune nouvelle génération de modèle ni campagne de débit n’a été lancée sur le paquet final ; les contrôles numériques historiques restent distincts.**

**CI e04c6558 : Linux, macOS et conversion/packing réussissent ; Flash-Kani et régression sont encore en cours au relevé. Kani global reçoit un signal d’arrêt après44harnais safety réussis, le suivant sans verdict. Aucun contre-exemple observé ; CI globale incomplète.**

## Artefacts

| Fichier | Octets | SHA-256 |
| --- | ---: | --- |
| `MLXL3-Desktop-v2.0.0-beta.1-b31-Apple-Silicon.dmg` |73990428|`2042d20a2927723f4d386c8edb6b70f4e1f90eb257c24991d0021065ae3567b3`|

Rapport commun : `MLXL3-v2.0.0-beta.1-validation.json`, SHA-256 **4fe729786783d99e6bd4956ca95090858d642ec29bcfbef986f0cbdbc446bc0d** ; preuves : `MLXL3-v2.0.0-beta.1-verification.tar.gz`, taille et empreinte dans ce rapport. Le même moteur est embarqué dans l'app et l'archive indépendante ; [notes moteur](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.0-beta.1).

## Remerciements

[@0xZKnw](https://github.com/0xZKnw), intégration/validation sur Mac ; [Qwen](https://github.com/QwenLM), modèles ; [MLX](https://github.com/ml-explore/mlx)/[MLX LM](https://github.com/ml-explore/mlx-lm)/[MLX-VLM](https://github.com/Blaizzy/mlx-vlm), runtime et références ; [ExLlamaV3](https://github.com/turboderp-org/exllamav3)/[PonyExl3](https://github.com/beamivalice/PonyExl3), EXL3 et kernels adaptés ; [Sushi](https://github.com/beamivalice/sushi), références d'optimisation ; [Strata](https://github.com/Niko1221/Strata)/[strata-mlx](https://github.com/yibie/strata-mlx)/[OpenBMB LES](https://github.com/OpenBMB/llama.cpp-omni), streaming/planification ; [Splash](https://github.com/incoai/splash), draft/attention/budget adaptés ; [SGLang](https://github.com/sgl-project/sglang), équations étudiées ; [MTPLX](https://github.com/youssofal/MTPLX), MTP ; [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique ; [MCP](https://github.com/modelcontextprotocol), protocole des outils ; [Kani](https://github.com/model-checking/kani), contrats CPU. [Attributions et licences](https://github.com/0xZKnw/mlxl3/blob/e04c6558959609443dda362f2e8509b998adaa6d/THIRD_PARTY_NOTICES.md).
