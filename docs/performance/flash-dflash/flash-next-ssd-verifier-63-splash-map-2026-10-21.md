# 63 — comparaison précise avec Splash

Source Splash :1482f6bbc2abf12375332725302019594150b6cb, HEAD distant inchangé le21octobre2026. Les quatre blobs relus sont archivés avec SHA256 et URL dans `measurements/flash-next-ssd-impl/opt-63-verifier/splash-review-source.json`. Les noms Git remontés initialement depuis le dossier de blobs appartenaient au dépôt parentMLXL3 ; cette erreur de provenance est corrigée dans `opti.md` avant toute utilisation.

| Partie | Splash | Premier prototype63 local | Écart restant |
| --- | --- | --- | --- |
| Vérification | `Qwen4ExpTarget::addVerify`, bloc de8positions par lane, graphe construit autour de scratchs réutilisables | B2..B8, projections EXL3 multi-lignes exactes et MoE déjà groupé, tableaux MLX immuables | Pas de graphe Metal persistant ni scratchs réutilisables équivalents |
| Projections | Projections préparées et format natif GGUF/Table16 ; entrées GDN packées qkv/z/beta/alpha | Même kernel EXL3 existant, décodage d’une tuile partagé entre2lignes, NT1/SG4/splits et réductions de référence | Pas de repacking GGUF/Table16 ; EXL3 Hadamard et ses arrondis doivent rester exacts |
| Préparation GDN | `verify_gdn_fused*`, convolution+SiLU, RMS q/k, portes, scan et sortie normalisée/gatée fusionnés ;256threads | Convolution et normes précédentes conservées parposition, scan multi-token/tape déjà disponible dansMLXL3 | Fusions conv/normes/portes pas encore portées ; ne pas prétendre avoir tout repris |
| Scan | Ordre causal decay→memory→delta→update→output conservé ; plusieurs lignes d’état en vol, non positions causales indépendantes | Même ordre dans le kernel GDN existant, étatFP32, entréesFP16 ;8positions séquentielles dans le même kernel | Les valeursBF16 de Splash ne sont pas substituables à notreFP16 |
| Tape/commit | Entréesconv+k/v+decay/beta, slots pending/step disjoints, pliage des pending dans le scan suivant ; commitPLE/QSA distinct | Étatinitial partagé viaRc, clés+delta+decay compacts ; replay seulement du préfixe retenu, historiqueconv/PLE propre à chaque préfixe | Replay séparé par couche ; fusion commit+passage suivant non implémentée |
| HC/PLE | HC, projectionsPLE et captures groupés ; convolutionPLE causale à partir de l’historique | HC etPLE encore parposition, étatPLE conservé à chaque préfixe | Fusions/batchHC etPLE à qualifier séparément, non revendiqués ici |
| QSA | Indexation, pooling et listes de blocs groupés ; clés pending commitées selon retainedCounts | Parcours causal QSA actuel conservé | BatchingQSA reste absent dans ce premier palier |
| MoE | Opérateur groupé et poids natifs résidents de leur modèle | SSD,53slots/couche,10experts/token ; cohortes et hybride existants | Partage réel EXL3 des experts répétés58 distinct, à qualifier après63 |

Le prototype63 doit d’abord vérifier les projections et le scan sur GPU puis tous les préfixes du modèle. Une simple concaténation ne reproduit pas le graphe/fusions de Splash ; chaque écart ci-dessus reste visible. Leurs chiffres M5Pro20cœurs/64Go ne sont pas une baseline du M5/10cœurs/24Go avec expertsSSD. Aucun gain de débit63 annoncé avant mesures.
