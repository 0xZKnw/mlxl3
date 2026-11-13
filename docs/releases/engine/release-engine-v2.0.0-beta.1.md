# MLXL3 moteur 2.0.0-beta.1 — Flash Next sur SSD

Première préversion moteur v2, issue de la [PR34](https://github.com/0xZKnw/mlxl3/pull/34). Tag `engine-v2.0.0-beta.1`, `prerelease=true`, `latest=false` ; version interne numérique **2.0.0**. Source propre du paquet : [`e04c6558959609443dda362f2e8509b998adaa6d`](https://github.com/0xZKnw/mlxl3/commit/e04c6558959609443dda362f2e8509b998adaa6d).

## Nouveautés

- **Flash Next EXL3 2.05bpw, texte :** les experts du checkpoint qualifié restent sur SSD et alimentent des banques réutilisables. Un cache borné conserve les experts résidents ; jusqu'à quatre lecteurs chargent les autres directement dans les banques.
- **MTP1 intégré :** Flash Next utilise la tête incluse dans son checkpoint, sans téléchargement MTP séparé. Chaque proposition est contrôlée par la cible avant émission.
- **DFlash2 expérimental :** draft local compatible, conversion Q4 en mémoire, cache KV circulaire de 2048 positions et vérification adaptative. Le bridge gère préfill, acceptation, commit, annulation et reprise.
- **Diagnostic mémoire et génération :** durées draft/vérification/commit, propositions et largeurs vérifiées ; allocations MLX, empreinte et compression du processus distinguées du swap système.

Les fonctions et familles précédentes sont conservées. Cette bêta ne qualifie pas les images, vidéos ou variantes arbitraires de Flash Next.

## Correctifs et réglages

Le catalogue télécharge désormais `ngram_embedding.safetensors` et `mtp_hyper_connection_mixer_patch.safetensors`. Leur omission rendait Flash incomplet et sous-estimait son téléchargement de **26 253 280 720 octets**, soit 26,253 Go. La sélection conserve l'isolation des dossiers et quantifications ; aucun poids existant n'est modifié.

Le Desktop associé peut soumettre un draft Flash local au vrai bridge résident, au lieu de le refuser dans le validateur réservé aux drafts legacy. Les corrections de logits/états, de rétention mémoire au chargement et de restauration spéculative sont conservées.

Par défaut, cache experts **automatique ≤ 6 Gio**, cache de buffers MLX **512 Mio**, quatre lecteurs et overlap actifs. Ces plafonds ne sont pas la RAM totale. GPU Hot/Wide, cohortes uniques, vérificateur multi-token 63 et prélecture prédictive restent désactivés ; le préplafonnement du draft reste opt-in.

MTP/DFlash2 sont des choix explicites et utilisent le sampling greedy. Lorsque DFlash2 Flash est choisi, Q4, ring et adaptatif sont actifs. **Auto est le mode 1 adaptatif**, distinct du B2 fixe des benchmarks. Fournir un draft local : ni téléchargement automatique Flash, ni Tune legacy Flash ajoutés. DSpark reste expérimental hors interface.

**K1 est exclu** : les comparaisons réelles n'ont montré aucun gain ; [preuves négatives et reproduction, PR47](https://github.com/0xZKnw/mlxl3/pull/47).

## Performances

Aucun nouveau gain du paquet bêta n'est revendiqué. Les campagnes historiques ont observé environ 11 tok/s sur certains passages DFlash2 B2/MTP1 : M5 standard, 10 cœurs GPU, 24 Go, MLX 0.32.2, Flash EXL3 2.05bpw, secteur, prompts MAIN/HOLD, warmup 128 puis 128 tokens émis/127 décodés. Elles utilisaient **3 Gio/53 slots et 64 Mio d'allocateur**, avec options expérimentales explicites, contrairement aux défauts ci-dessus.

Ces résultats ne garantissent ni le débit du Desktop, ni la supériorité DFlash2/MTP. **15 tok/s ne sont pas établis** ; gains de composants et mesures amont ne s'additionnent pas. [Protocoles et limites historiques](https://github.com/0xZKnw/mlxl3/blob/e04c6558959609443dda362f2e8509b998adaa6d/docs/flash-next-ssd-verifier-63-2026-10-21.md).

## Installation et compatibilité

Apple Silicon arm64, macOS 26.2+, MLX 0.32.2 ; signature ad-hoc, sans notarisation Apple. Préférer Desktop 2.0.0-beta.1/build 31 pour tester l'ensemble. Les modèles restent séparés : cible qualifiée `65c895314393431c09050b2e04e250836b3a6eb4`, draft compatible `a22a1744e85c17364dcc54a6a420af7135d4c7f6`.

Les canaux moteur/Desktop sont indépendants ; les deux bêta sont non-latest. L'updater stable n'active pas de canal bêta. **Si ce moteur 2.0.0 bêta est installé indépendamment, revenir au moteur embarqué avant la future stable 2.0.0**, dont le même numéro interne pourrait empêcher une mise à jour automatique.

## Vérification et limites

Format, Clippy strict CPU/MLX réussis ; **228 tests CPU passés / 5 ignorés**, 7 tests Hub en configuration MLX réussis. Le défaut catalogue et une mutation MTP ont été détectés, puis la source restaurée. Contrôles Desktop concernés : 8 cas Flash, 240 contexte, 114 assertions idle, 965 sampling ; 45 tests packageur réussis.

Kani a analysé la vraie sélection `variants` sans stub, puis atteint 180s : **propriété non vérifiée**. Les preuves CPU antérieures ne prouvent ni Metal, ni concurrence ou génération entière. Swift n'est pas formellement prouvé ; CrossHair indisponible pour les contrôleurs concernés. Les tests ignorés, autres Mac et contextes longs restent des limites.

**Les31contrôles du paquet passent : signatures, arm64, manifestes et moteur relocalisé. La suite Desktop complète passe avec les vrais paquets (installation, exécution, fallback et démontage), en366,27s,87sources inchangées et processus joints. À la demande de publication immédiate, aucune nouvelle génération de modèle ni campagne de débit n’a été lancée sur le paquet final ; les contrôles numériques historiques restent distincts.**

**CI e04c6558 : Linux, macOS et conversion/packing réussissent ; Flash-Kani et régression sont encore en cours au relevé. Kani global reçoit un signal d’arrêt après44harnais safety réussis, le suivant sans verdict. Aucun contre-exemple observé ; CI globale incomplète.**

## Artefacts

| Fichier | Octets | SHA-256 |
| --- | ---: | --- |
| `MLXL3-Engine-v2.0.0-beta.1-arm64.tar.gz` |67229225|`ca2e650f8cb944486592b60fc02f773e28388dd46a4ab38707971c0a11057412`|
| `MLXL3-Engine-v2.0.0-beta.1-arm64.json` |13588|`9262b7cdd07fd000561d86424c174783341c5903a0d144a7d35417cb73c913bc`|

Rapport commun : `MLXL3-v2.0.0-beta.1-validation.json`, SHA-256 **4fe729786783d99e6bd4956ca95090858d642ec29bcfbef986f0cbdbc446bc0d** ; preuves : `MLXL3-v2.0.0-beta.1-verification.tar.gz`, taille et empreinte dans ce rapport.

## Remerciements

[@0xZKnw](https://github.com/0xZKnw), intégration et essais sur Mac ; [Qwen](https://github.com/QwenLM), modèles ; [MLX](https://github.com/ml-explore/mlx), [MLX LM](https://github.com/ml-explore/mlx-lm), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), runtime/références ; [ExLlamaV3](https://github.com/turboderp-org/exllamav3)/[PonyExl3](https://github.com/beamivalice/PonyExl3), EXL3 et structures de kernels ; [Sushi](https://github.com/beamivalice/sushi), références d'optimisation ; [Strata](https://github.com/Niko1221/Strata)/[strata-mlx](https://github.com/yibie/strata-mlx)/[OpenBMB LES](https://github.com/OpenBMB/llama.cpp-omni), planification/streaming ; [Splash](https://github.com/incoai/splash), ABI draft et attention/budget adaptés ; [SGLang](https://github.com/sgl-project/sglang), équations de draft ; [MTPLX](https://github.com/youssofal/MTPLX), mécanismes MTP ; [Kani](https://github.com/model-checking/kani), contrats CPU. [Licences et attributions précises](https://github.com/0xZKnw/mlxl3/blob/e04c6558959609443dda362f2e8509b998adaa6d/THIRD_PARTY_NOTICES.md).
