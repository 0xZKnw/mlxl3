# Qwen3.8-27B : réduire la RAM sans altérer les valeurs

Recherche du 16 octobre 2026, checkout `codex/retour-version-normale`, HEAD
`8c08c4c45c1f7795f476ca514e7c7d3b8cebce7b`, propre avant ce travail.
Le livrable est une recherche avec un filtre CPU sur les poids réels ; aucun
changement exécutable du moteur, chargement GPU ou remplacement d’application.

La meilleure nouvelle piste est une **compression réversible des embeddings** :
le stockage calculé descend de 2 542 796 800 à 1 957 244 112 octets, métadonnées
incluses, soit **585,55 Mo / 558,43 Mio**. Ce n’est pas encore une économie de
RAM processus mesurée. Une autre branche possède déjà une compaction exacte
des snapshots, absente du checkout actuel. Le KV8 a été essayé et rejeté ; il
ne convient pas à la contrainte de qualité demandée.

## Ce qui occupe la mémoire

Métadonnées du modèle installé : dense, 64 couches, hidden 5120, vocabulaire
248320, EXL3 nominal 2 bits/tête 3 bits. Le nom du dossier est Qwen3.8-27B ;
la configuration utilise l’architecture `qwen3_5`. Il s’agit de conserver
exactement le comportement de ce checkpoint, sans nouvelle quantification.

- **Embeddings : 2,543 Go**, `[248320,5120]`, BF16 dans le fichier, convertis
  intégralement en F16 par `half_weight` au chargement. Cette table est utilisée
  par les entrées du modèle, le préfill et le draft MTP.
- **KV cible : 65 536 octets/token**, seulement dans les 16 couches full attention :
  `16 × 2(K,V) × 4 têtes × 256 dimensions × 2 octets`. Les 48 autres couches sont
  récurrentes et ne conservent pas un KV croissant pour chaque token.
- **GDN : 146,8125 Mio fixes** de payload logique : 144 Mio d’états récurrents
  FP32 et 2,8125 Mio de convolution F16. Les vues et tableaux intermédiaires
  peuvent retenir davantage de stockage que ce payload.
- **MTP : 4 096 octets/token de KV supplémentaires**, plus ses poids et buffers.
  La tête n’a pas de deuxième embedding dédié. Le code libère déjà la tête
  lorsqu’elle est désactivée (`set_mtp` et génération ordinaire).
- Poids EXL3, temporaires du préfill, snapshots et allocateur complètent ces
  postes. Un snapshot clone des handles MLX, pas systématiquement tout le KV ;
  ses anciennes allocations peuvent cependant rester vivantes après append.

| Tokens réellement en cache, prompt + génération | KV cible F16 | KV cible + tête MTP F16 |
| --- | ---: | ---: |
| 4 096 | 256 Mio | 272 Mio |
| 8 192 | 512 Mio | 544 Mio |
| 16 384 | 1 Gio | 1,0625 Gio |
| 32 768 | 2 Gio | 2,125 Gio |
| 65 536 | 4 Gio | 4,25 Gio |

Ces nombres sont des tailles logiques, pas des footprints ou pics. Le contexte
configuré est une borne ; le code courant concatène les tokens effectivement
traités et ne réserve pas d’emblée tout ce maximum.
[Calculs indépendants](measurements/qwen27-ram-audit-20261016/memory-dimensions.json),
[identités des sources et configurations](measurements/qwen27-ram-audit-20261016/source-identity.json).

L’inventaire des headers trouve aussi 398 057 872 octets vision et 106 468 384
octets de tenseurs `mtp.*` natifs dans les fichiers. Le loader texte courant ne
les charge pas ; enlever ces tenseurs économiserait du disque, sans justifier
une économie de RAM d’inférence. Le MTP géré charge son sidecar séparé.
[Inventaire](measurements/qwen27-ram-audit-20261016/checkpoint-headers.json).

## Économies déjà connues et état réel

| Piste | Résultat historique | Présence dans ce checkout |
| --- | --- | --- |
| Libération après génération, RAM-02 | Environ 8,44 Go de footprint rendus après 10 138 tokens sur ce dense | Présente ; pic de préfill inchangé |
| Mode Économie de mémoire, RAM-04 | 474,677 / 778,650 Mo rendus sur deux ordres courts | Présent, OFF par défaut ; recalcule le contexte au prochain envoi |
| Compaction exacte des snapshots, CACHE-COMPACT | Backing du snapshot 425 197 568 → 170 928 128 octets, soit −254 269 440 octets | Absente ; conservée sur `codex/adaptive-memory-qwen` |
| Ancien embedding exact haut13 bits + exceptions | Table −301 465 600 octets | Absent ; opt-in sur cette autre branche |
| Compaction + ancien embedding, même campagne | Footprint −654 113 232 / −566 180 256 octets, MLX actif −555 735 040 octets | Absents de ce checkout ; mesures du couple uniquement |

RAM-02 et RAM-04 sont documentés dans [le diagnostic RAM](ram-idle-context.md)
et [le mode économie](memory-saver.md). Le toggle actuel est dans les réglages ;
il conserve les poids, les messages et les paramètres de génération. Son coût
est la répétition du préfill, pas une suppression de contenu.

Les autres résultats viennent de la branche `codex/adaptive-memory-qwen`,
tête locale `0f53d2cadc9f5a556340abdc4d86763400a3f4a4`, notamment son rapport
`docs/adaptive-memory-qwen-20261009.md`. Cette branche est issue de la même
base historique mais contient aussi conseil de contexte, capture et MTP adaptatif.
Une reprise mémoire doit isoler ses composants utiles ; leur présence ailleurs
ne signifie pas qu’ils sont intégrés ici ou dans l’app installée.

La compaction copie exactement les vues qui retiennent >2× leur payload et au
moins 64 Kio inutiles, uniquement au checkpoint de préfill, hors rollback MTP.
Son intérêt est de conserver la réutilisation du contexte en réduisant le backing.
Son économie seule n’a pas été isolée en footprint dans cette campagne.

La mesure du couple utilisait budget snapshot 1024 Mio et allocateur libre 0
communs aux deux bras, contexte 4096, prompt 687 tokens, réponses 32 tokens,
réutilisation 512 tokens, ordres AB puis BA. Il serait incorrect de promettre les
mêmes deltas avec les défauts actuels, d’additionner ces chiffres à RAM-04 ou
de déduire un gain de vitesse. Les contrôles précédents étaient exacts sur leurs
trajectoires : huit positions de logits et 128 états pour l’embedding ; 32 cas bridge
pour la combinaison. Les débits du diagnostic Tune dérivaient de 11 à 15 %.
[Extraits bruts identifiés par commit et empreintes](measurements/qwen27-ram-audit-20261016/historical-evidence.json).

## Nouvelle piste : exposants des embeddings par blocs, sans perte

L’ancien format rangeait les F16 convertis sur 13 bits hauts et des exceptions
de 3 bits bas. La nouvelle hypothèse part des **BF16 originaux** : conserver
signe et mantisse dans un octet, puis coder l’exposant relativement au minimum
du bloc de 128 valeurs, avec une largeur variable. Après reconstruction exacte,
appliquer la même conversion BF16→F16 que le moteur actuel. Aucun bit du poids
source n’est jeté. Ce mécanisme n’est pas une quantification des poids.

Filtre CPU sur **toute** la table locale : 1 271 398 400 valeurs, 9 932 800 blocs,
chunks d’au plus 1 048 576 valeurs. Python3.9.6, NumPy2.0.2. Headers, compte,
dimensions et bornes du fichier contrôlés ; SHA256 du payload embedding :
`a158f24c6ad72736067768b4255e70a384e2e931f601b12f853e98ce83ec6bbe`.

| Largeur d’exposant relatif | Nombre de blocs |
| --- | ---: |
| 0 | 79 |
| 1 | 8 |
| 2 | 25 |
| 3 | 1 909 833 |
| 4 | 7 952 875 |
| 5 | 65 980 |
| 6 | 0 |
| 7 | 4 000 |
| 8 | 0 |

Le calcul réserve **8 octets de métadonnées par bloc**, pour un offset 32 bits,
la base, la largeur et deux octets réservés ; corps signe/mantisse128 octets et
exposants `16 × largeur` octets. Fallback brut256 octets si non rentable. La taille
totale proposée reste sous 2^32 octets ; un format général doit contrôler cette
borne, et conserver le fallback sur d’autres checkpoints.

- Dense : **2 542 796 800 octets**.
- Ancien packed : **2 241 331 200 octets**, historique.
- Nouveau format calculé : **1 957 244 112 octets**.
- Différence face au dense : **585 552 688 octets**, soit585,55Mo/558,43Mio.
- Différence face à l’ancien packed : **284 087 088 octets**.

La reconstruction logique des BF16 est exacte sur l’ensemble des valeurs.
Les bornes de largeur sont vérifiées par inégalités entières indépendantes du
logarithme utilisé pour les choisir. Un bloc représentatif pour chacune des sept
largeurs présentes passe également le vrai packbits/unpackbits. Cela établit la
faisabilité du rangement sur ces données, **pas une preuve de futur kernel**.
L’analyse conserve les bits BF16, y compris les cas que la conversion F16 actuelle
pourrait arrondir ; cette conversion n’a pas été rejouée ni modifiée ici.

[Résultats complets](measurements/qwen27-ram-audit-20261016/embedding-exponent-feasibility.json),
[commande exacte](measurements/qwen27-ram-audit-20261016/embedding-analysis-command.json).
RAM processus, active MLX, pic de chargement et débit du nouveau format :
**non mesurés**. L’alignement des allocations et les transitoires ne figurent pas
dans la taille logique calculée. Aucun format runtime ni shader ajouté.

## KV cache et autres pistes : choix

**Conserver le KV F16.** Les variantes8bits G64, G32/F16 puis G32/F32 de l’autre
branche ont déjà été filtrées. La dernière passait le court mais échouait à
8701tokens : RMSElogits0,0365358610 > seuil0,03 ; maxabs0,169921875. Aucun argmax
greedy ne changeait sur les quatre positions, ce qui ne suffit pas à garantir
les suites ou la qualité. Prototype retiré ; rollback quantifié/bridge/vitesse
non exécutés après cet échec. Le [cache quantifié MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
utilise bien une quantification des K/V, distincte d’un rangement réversible.

**Ne pas tronquer le contexte, supprimer des tokens ou réduire les états GDN.**
Ces modifications changent l’information ou les calculs accessibles au modèle.
Réduire seulement la borne configurée n’économise pas un KV préalloué absent,
et peut tronquer une conversation plus tard.

**KV paginé / append mutable versionné : seconde priorité de recherche.**
Le code concatène K/V et la vérification MTP répète cet append par proposition.
Une gestion de pages F16 pourrait réduire copies et pics en conservant chaque
token. L’ancienne réserve fonctionnelle past+1024 (POST-SMALLM-B) était exacte
mais sans signal utile ; footprint non mesuré. Refaire ce même essai serait
inutile. Une architecture mutable versionnée serait différente, mais doit
préserver snapshots, branches et rollback sans fuite future. L’attention paginée
pourrait changer les réductions ; un simple rapprochement avec le
[KVCache amont MLX-LM](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
ne prouve pas la parité. Gain ici **non mesuré**.

**Limiter les buffers libres pendant le préfill : candidat distinct du KV.**
RAM-02 agit en fin de tour. POST-SMALLM-H, sur une autre branche, purgeait aussi
le scratch entre chunks longs ; environ3,99Go rendus après réponse, pic proche
du stock, vitesse non établie et OFF. Ces chiffres précèdent RAM-02 et ne sont
pas un gain supplémentaire sur ce checkout. Une limite allocateur intermédiaire
non nulle pourrait réduire la rétention pendant les requêtes ; coût de
réallocation à qualifier. Ne pas imposer cache0 à partir de mesures mémoire
seules. L’API [clear_cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html)
vide le cache allocateur ; elle ne supprime pas les tableaux vivants.

La résidence/wired-limit a déjà été essayée et retirée sans gain robuste. Un
déport CPU seul ne constitue pas une économie de RAM totale sur mémoire unifiée.
Un stockage disque/recalcul exact serait possible au prix d’I/O et de latence,
mais aucun besoin de ce changement plus lourd n’est établi avant les pistes
réversibles ci-dessus. Le déchargement automatique du modèle au repos existe
déjà ; son économie ne concerne pas la génération modèle chargé.

## Ordre d’implémentation proposé et vérification requise

1. Isoler la compaction de snapshot déjà qualifiée, avec ses tests et sa vraie
   comptabilité de backing, sans importer MTP adaptatif ni le conseiller de contexte.
2. Prototyper le **nouveau** format d’embedding exact derrière une option. Conserver
   IDs et gather sur GPU, y compris MTP ; ne pas ajouter un retour CPU par token.
3. Vérifier les erreurs, bornes et fallback, puis tous les motifs BF16 utiles,
   gathers avec répétitions/IDs frontières et comparaisons F16 en bits contre le
   chemin `half_weight` actuel. Charger le skill formal-proof-skill et références
   Rust/C++ avant code ; Kani sur les vrais contrats et tentative vérificateur C++.
4. Comparer les 248320 logits finis, hidden et 128 états nonvides en bits sur préfills
   courts/longs, puis sessionsMTP, rollback, bridge/CLI, cache froid/chaud, Tune,
   annulation/erreur/reprise. Un mutant de décodage doit être détecté.
5. Seulement après correction : mesure même binaire OFF/ON, processus propres,
   warmups et ordres alternés, un seul modèle GPU, aucun build/prover concurrent.
   Distinguer loading/ready/préfill/decode/repos, footprint/MLXactif/cache/pic,
   ainsi que temps de chargement/TTFT/débit. Ne pas promouvoir si ralentissement
   stable inacceptable ; conserver les résultats négatifs.

Ce plan n’est pas un essai lancé. Chaque prototype aura son préenregistrement
dans `opti.md`, ses identités et ses seuils avant exécution. L’égalité des bits
des données reconstruites est le contrat recherché pour éviter toute altération
supplémentaire du modèle ; les quelques textes identiques ne la remplacent pas.

## Contrôles de cette recherche

Journal et rapports historiques RAM/round2/smallm/post-smallm/adaptive/Tune lus ;
sources et appelants Desktop→bridge→Qwen/MTP examinés. Les checkpoints ne sont
pas modifiés. Les empreintes des14sources/configurations du dépôt sont conservées
et contrôlées en clôture. Une seule analyse CPU nouvelle, terminée sans divergence ;
aucun modèle GPU ou processus enfant persistant créé.

Les workflows PR/push sont lus : Rust/Kani/Linux/macOS, compilation `mlx,chat`
avec MLX0.32.2 et Desktop. Les tests physiques restent ignorés sur CI hébergée.
La lecture distante `gh run list --repo 0xZKnw/mlxl3 --commit 8c08c4c…` échoue par
connexion à `api.github.com` ; aucun statutCI exact n’est revendiqué.
Pas de tests moteur, Kani, build ou benchmarkGPU relancés pour ce diff documentaire.
Le filtreCPU est une exploration finie des données réelles, pas une preuve
formelle ni une validation du parcours utilisateur. Aucun essai encore en cours,
aucune publication et aucune app installée modifiée.
