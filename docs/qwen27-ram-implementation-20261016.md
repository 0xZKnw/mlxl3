# Qwen3.8-27B : stockage exact et cache de contexte conservé

Campagne du 16 octobre 2026, base `8c08c4c`, changements locaux.
**Terminé : code intégré, copie locale construite, économie de RAM mesurée.**
L’application installée et les poids restent inchangés.

## Résultat mesuré

Après un échange de 687 tokens et 32 tokens générés, avec 512 tokens de préfixe
réutilisés, l’économie de RAM processus est de **805,96 à 867,38 Mo** dans les deux
ordres de comparaison. Les allocations MLX actives baissent exactement de
**839 817 216 octets (839,82 Mo)** dans les deux ordres. Mo désigne ici 1 000 000 octets.

| Cache chaud, après réponse | Contrôle, octets | Optimisé, octets | Économie, octets |
| --- | ---: | ---: | ---: |
| Ordre AB | 10 004 583 872 | 9 198 623 656 | 805 960 216 |
| Ordre BA | 10 068 497 904 | 9 201 113 856 | 867 384 048 |

MLX actif passe de 9 855 680 380 à 9 015 863 164 octets. Le cache libre de
l’allocateur vaut zéro au point observé dans les deux bras, **sans lui imposer une
limite de zéro**. Sa politique habituelle est conservée. Dans chaque cas chaud,
512 tokens sont réutilisés et 175 évalués, pour un contexte final de 719 tokens.

Le même moteur signé et les mêmes poids sont comparés avec les deux optimisations
OFF puis ON, dans les ordres AB et BA. Quatre processus fermés et rejoints,
32 cas, 24 réponses non vides de 32 tokens, quatre annulations et quatre erreurs
attendues. L’oracle indépendant retrouve les mêmes IDs, texte, contexte et
compteurs MTP entre les quatre processus.
[Mesures, timings et conditions](measurements/qwen27-ram-impl-20261016/memory-summary.json).

Le **mode « Économie de mémoire » reste désactivé** dans toutes ces requêtes.
KV F16, taille de contexte, réutilisation de préfixe et politique de l’allocateur
MLX sont conservés. Aucune réduction de précision n’est appliquée.

## Changements intégrés

- Embeddings BF16 rangés par blocs de 128 valeurs : signe et mantisse sur huit
  bits, exposant relatif de largeur 0 à 8 bits, huit octets de métadonnées par
  bloc. Le gather GPU restitue les BF16 originaux, puis applique le **même cast
  MLX vers F16** que le chemin canonique. Le format dense est conservé si la
  forme ou le dtype ne convient pas, ou si l’encodage est plus gros. Activation
  par défaut limitée à la géométrie dense validée : hidden 5120, 64 couches,
  vocabulaire 248320, aucun expert. Repli au prochain chargement :
  `MLXL3_LOSSLESS_EMBEDDING=0`.
- Copie exacte des vues de snapshot qui retiennent plus de deux fois leur
  payload et au moins 64 Kio inutiles. Appliquée aux checkpoints de préfill
  cible, logits et MTP, hors snapshots de rollback par token. Le critère existant
  de 256 Mio de payload cible et l’éligibilité du préfixe sont conservés.
  Repli : `MLXL3_CACHE_COMPACTION=0`.
- Clé Tune séparant les stockages dense et packed pour recalibrer le bon moteur.
  Aucun MTP adaptatif, KV quantifié, conseiller de contexte ou plafond
  d’allocateur n’est importé de l’autre branche.

L’embedding réel retient **2 542 796 800 → 1 957 249 024 octets**, soit
585 547 776 octets de moins. C’est 4912 octets d’économie de moins que la prévision
initiale, en raison de l’alignement. Le snapshot de préfill réel à 256 tokens
retient **425 197 568 → 170 928 128 octets**, soit 254 269 440 octets de moins,
avec le même payload logique de 170 731 520 octets. Ces deux différences
retrouvent la baisse d’allocation MLX mesurée ensemble ; elles ne sont pas
additionnées à la RAM processus ou aux anciens résultats de l’autre branche.

## Vérification

- **Modèle physique :** préfills de 24, 257 et 8193 tokens, chunks de 256, puis
  trois décodages chacun. Douze positions avec 248320 logits finis, hidden et
  128 états aux dtypes originaux **identiques bit à bit** au stockage dense
  canonique. Gather aux frontières et échantillons du vocabulaire exact.
- **Codec et copies GPU :** tous les 65536 motifs BF16, conversion F16 canonique,
  permutations, IDs I32/U32, largeurs 0 à 8, croisements interoctets, formes et
  erreurs. Copies exactes sur huit fixtures de 65536 valeurs, dont vues stridées,
  en F16/BF16/F32/U32. Le contrôle compaction OFF échoue comme attendu, ON repasse.
- **Snapshots et MTP :** restauration puis token suivant exacts ; cache de tête
  et prochaine prédiction exacts. Sessions aux profondeurs 1 à 3 : 24 blocs,
  IDs/hidden/128 états/cache de tête exacts face au target mono-token et à une
  réparation indépendante. Matrice complète **180/180 cas** : cinq préfills
  (1, 23, 24, 129, 256), largeurs 1 à 8 et chaque préfixe retenu, plus erreurs
  de vocabulaire, entrées vides, largeur 9 et budgets de contexte.
- **Bridge signé :** cold/warm, MTP 2/3, erreur depth 4, annulation, reprise et
  cache après erreur passent. Tune annulé puis reprise exacte ; Tune complet
  aux quatre profondeurs, 190 tokens décodés par profondeur et deux hashes non
  vides identiques entre profondeurs. Clé de stockage reconnue, ping et génération
  après Tune identiques au golden. `memory_saver=false` partout.
- **Suites :** Rust livré `mlx,chat` : 78 passés / 69 ignorés ; neuf sélections GPU
  positives exécutées séparément, dont la répétition ON après contrôle négatif.
  Rust CPU : 62 passés / 3 ignorés. Python : 338 passés / 4 skips historiques
  (convertisseurs PonyExl3 et modèle Ling absents). Suite Desktop complète passée :
  import, cycle de vie, bridge, annulation/reprise, rendu et transport CLI.
  Build release, format et lints stricts Rust/C++ réussis.
- **Preuves bornées :** Kani sur les contrats réels, 46/46 harnais, 5705 obligations
  réussies, 126 couvertures satisfaites ; 72 obligations inatteignables inspectées,
  toutes dans les modèles/stdlib. Domaines scalaires complets pour les deux gardes
  ajoutées, sans `assume`. CBMC sur les vrais helpers C++ : 43/43 obligations du
  codec et 52/52 de largeur d’exposant ; pointeurs, bornes, shifts et unwinding.
  Couverture 18/19, seul bras d’assertion fausse inatteignable. ASan/UBSan :
  590976 reconstructions exactes et 32896 couples min/max. Mutation supprimant
  l’écriture interoctets détectée dans une copie du codec ; original intact.

Le skill `formal-proof-skill` et ses références Rust/C++/Python ont été chargés.
Ces preuves CPU ne couvrent pas MLX, Metal, l’allocateur, la concurrence ou
l’application complète. Le loader C++ complet reste non prouvé : tentative
CBMC bloquée par son parseur sur libc++/C++20. CrossHair a été tenté et est absent ;
le contrôleur Python/IO n’est pas vérifié formellement. Son validateur accepte une
fixture valide et rejette dix corruptions ; format, lint et pycompile passent.
Un contrôle négatif confirme que le pilote RAM refuse une qualité physique
incomplète avant de démarrer un moteur.

Les deux premiers passages de la matrice MTP ont expiré à 180 puis 600 secondes,
sans divergence signalée ; processus fermés et rejoints, rapports d’échec conservés.
Le harnais V2 synchronise et libère uniquement son scratch inutilisé entre
comparaisons, puis affiche sa progression. Les 180 cas et toutes leurs assertions
sont conservés. Il passe en **834,99 s**, limite 1200 s. Le binaire de production
reconstruit est **identique octet par octet au candidat V1** ; aucun changement de
politique mémoire de l’application. Ce temps de test ne prouve aucune accélération.
Les erreurs initiales de compilation, instrumentation CBMC et accès sandbox
restent également archivées.

## Frontières vérifiées

| Entrée → fonctions | Contrat et effet |
| --- | --- |
| Chargement Qwen → Embedding.load/from_weight → checkpoint_array / mlxl3_pack_bf16_embedding | BF16 sources conservés, blocs de 128, offsets/dimensions bornés ; repli dense non rentable |
| Préfill/decode/MTP → Embedding.take → shader → view(BF16).astype(F16) | IDs validés par les appelants, mêmes forme/dtype ; pas de synchronisation CPU par token |
| PromptCache.capture → QwenSnapshot.compact_for_cache / Array.compact_for_cache | Offset/historique et payload cible de 256 Mio inchangés ; seules vues coûteuses copiées |
| prefill_mtp → Head.snapshot → Cache.compact_for_cache | Checkpoint de réutilisation compacté ; snapshots ordinaires de draft/verify/rollback conservés |
| ready/Tune → runtime_key | Calibrations séparées, champs smallM et autres options conservés |

## Copie locale et limites

Fermer l’application actuelle puis utiliser `dist/Lancer MLXL3 optimisé.command`.
Le lanceur sélectionne le moteur embarqué de cette copie et désactive le mode
« Économie de mémoire » pour ce lancement via `NSArgumentDomain`, sans écrire les
préférences persistantes. Il refuse une UI ou un moteur déjà actif. Syntaxe Zsh
et comportement Foundation YES/NO contrôlés. La copie est signée et vérifiée
(arm64, plist, signature stricte) ; les hooks release de timeline et préférences
MCP passent avec préférences isolées. Leur contrôle Metal était explicitement
sauté, les tests physiques ci-dessus couvrant séparément cette frontière.

Les anciens `dist` et runtime ont été sauvegardés avant construction. Les trois
empreintes de l’app installée sont inchangées. Aucun push, installation ou
publication. Les deux runs GitHub et leurs quatre jobs de la **base exacte
8c08c4c** sont réussis et archivés ; aucune CI distante du diff local n’est
revendiquée. La CI locale modifiée ajoute sanitizers du codec et CBMC.

Matériel : M5 / 24 Gio, macOS 27.2, MLX 0.32.2, SDK 26.5. Les sondes d’alimentation divergent : les contrôleurs imbriqués
consignent AC, tandis que le contrôle séparé hors sandbox voit la batterie à
80 % en décharge à la clôture. Température et fréquences non mesurées.
Les débits diagnostiques autour de 3 tokens/s dans les deux bras ne permettent
aucune conclusion de boost par rapport aux anciennes mesures. Le gain de RAM
processus est validé sur le profil court décrit, sans extrapolation aux longs
contextes. Le contexte 8193 est un contrôle de correction uniquement. Temps de
chargement et pic de chargement isolé : **non mesurés**. Aucun test ne garantit
l’absence universelle de bug.

## Preuves conservées

- [Identité finale et versions](measurements/qwen27-ram-impl-20261016/final-source-identity.json), [checkpoint SHA-256](measurements/qwen27-ram-impl-20261016/checkpoint-payload-hashes.json).
- [Modèle réel](measurements/qwen27-ram-impl-20261016/embedding-model-quality.json), [snapshot](measurements/qwen27-ram-impl-20261016/snapshot-quality.json), [clôture GPU](measurements/qwen27-ram-impl-20261016/gpu-quality-final-status.json).
- [Préservation du runtime après réparation du harnais](measurements/qwen27-ram-impl-20261016/runtime-source-preservation.json), [Tune réel](measurements/qwen27-ram-impl-20261016/tune-bridge.json).
- [RAM AB/BA](measurements/qwen27-ram-impl-20261016/memory-summary.json), [package](measurements/qwen27-ram-impl-20261016/package-checks.json), [lanceur](measurements/qwen27-ram-impl-20261016/launcher-checks.json).
- [Index des logs compressés sans perte](measurements/qwen27-ram-impl-20261016/log-archives.json), [bilan de clôture](measurements/qwen27-ram-impl-20261016/verification-summary.json).

Les commandes, options, environnements, rapports partiels et archives des sources
baseline/V1/final sont dans le même dossier de mesures. Les logs ne dépendent
plus des fichiers temporaires de build.
