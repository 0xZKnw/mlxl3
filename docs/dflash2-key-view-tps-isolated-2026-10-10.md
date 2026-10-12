# DFlash2 35B : contrôle isolé du débit, 10 octobre 2026

Le moteur local avec KEY-VIEW donne **67,18 tok/s sur le code** et **58,74 tok/s
sur le français court**. Les écarts avec l'ancien moteur sont **+1,97 et
+3,38 tok/s**. Ils restent **indicatifs** : les variations de vitesse entre
passages dépassent le seuil de 1 % fixé avant mesure. Aucun gain de génération
n'est qualifié ; aucun BAAB ni nouvelle relance pour bruit n'a été lancé.

## Résultats

| Prompt | A ancien, tok/s | B KEY-VIEW, tok/s | Écart, tok/s | Écart observé | Dérive de vitesse A / B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code Python | 65,20957 | 67,17915 | +1,96958 | +3,020 % | −4,206 % / −3,567 % |
| Français court | 55,36002 | 58,74135 | +3,38133 | +6,108 % | −3,141 % / +0,843 % |

Ce sont les médianes des deux médianes de passage de chaque variante (leur
moyenne à deux valeurs), suivant le driver existant. On ne regroupe pas les six
échantillons. Dérive = `100 × (deuxième médiane / première médiane − 1)`.
Seuils préenregistrés : gain code ≥3 %, régression du court ≤1 %, toutes les
dérives ≤1 % en valeur absolue. Les deux premiers critères passent, le dernier
échoue. Ces écarts ne certifient pas un gain attribuable à KEY-VIEW.

| Passe | Variante | Français, tok/s | Code, tok/s | Swap utilisé au relevé, Mio |
| --- | --- | ---: | ---: | ---: |
| 0 | A | 56,24339 | 66,61036 | 3 492,94 |
| 1 | B | 58,49493 | 68,39908 | 5 588,00 |
| 2 | B | 58,98777 | 65,95922 | 6 900,56 |
| 3 | A | 54,47665 | 63,80877 | 5 546,88 |

ABBA complet : **24 mesures de 128 tokens et huit warmups de 256 tokens**, tous
terminés, métriques finies et positives, cache réutilisé zéro, empreintes de
tokens, texte et historique identiques aux références par prompt et budget.
Parité globale `true` sur ces 32 réponses. La variation de vitesse est distincte
de cette égalité des sorties. Aucun timeout, annulation ou divergence.

Le swap global reste variable sans autre MLXL3 détecté. Il ne mesure ni la RAM
résidente du moteur ni les transferts pendant une réponse. La cause exacte des
variations de débit reste indéterminée ; aucune attribution causale à un nouveau
défaut du moteur. `pmset` ne rapporte pas d'alerte thermique. Températures et
fréquences exactes non mesurées.

## Reprise et protocole

Le [test du 9 octobre](dflash2-key-view-tps-2026-10-09.md) est conservé après
timeout et exécutions concurrentes : l'utilisateur a confirmé avoir lancé MLXL3
par accident. Cette reprise est motivée par la **correction de l'isolation**.
Les conditions sont maintenant sur secteur. Les deux campagnes ne sont pas
mélangées ou utilisées comme baseline l'une de l'autre.

Garde toutes les deux secondes : noms complets normalisés des apps/moteurs
MLXL3, compilateurs et proveurs, dont `cargo-kani` manqué auparavant. Seuls le
driver possédé et ses descendants sont exclus. Sur conflit, changement de
secteur ou timeout, terminer/rejoindre uniquement le groupe possédé. Huit
assertions vérifient les noms complets, l'app et les arbres PID. Tests de contrôle,
sans preuve formelle des effets du système. Aucun conflit détecté pendant la
campagne ; aucun processus moteur/proveur pertinent actif au contrôle final.
L'échantillonnage ne garantit pas l'absence d'un processus très bref entre relevés
ni l'absence de toute activité système.

Qwen3.6-35B-A3B EXL3 **2,49 bpw**, draft DFlash2 packed Q4 ; M5 Air 24 Gio,
macOS 27.2, MLX 0.32.2. Secteur constant, batterie en charge : 25 % au
préenregistrement, 28/29/29/30 % aux passes, 31 % au contrôle final. Le premier
relevé exploratoire était à 22 % ; le plan conserve les conditions de référence.

`benchmarks/compare_native.py` inchangé : DFlash Auto, MTP OFF, contexte 4096,
greedy, sans réutilisation de cache, MCP ni outils. Pour A et B :
`MLXL3_DFLASH_CONTEXT_COPY=0`, `MLXL3_QWEN_PIPELINE=0`, `MLXL3_MTP_LOOKUP=0`.
App homes jetables séparés ; mêmes deux prompts que précédemment, warmup 256
puis trois mesures de 128 tokens par prompt/passe, repos fixe de 30 secondes.
ABBA unique et BAAB seulement si tous les seuils passent. Chargement 120 s,
requête 90 s, campagne 600 s ; durée totale du contrôle **243,38 s**, distincte
du débit de décodage. Un seul modèle/processus GPU de campagne à la fois,
aucun build, preuve ou autre mesure GPU pendant le timing.

## Identité, vérification et état

Base `db52b7304a057ad8f0d8fcf464cd66b22b44fe08`, diff local existant conservé.
A signé : `ad0bef499cd1850795d5ad9c8a54055d91299e9564fa55ebeb898e7ffde03c64`.
B signé : `23e0f3c16ec08a54be6b54ee83009f16e4a1e3be43253a609088181661c55e69`.
Bibliothèques communes. Douze sources, deux moteurs, trois bibliothèques,
driver et prompt vérifiés avant/après : aucune empreinte changée. Aucun code,
shader ou réglage de quantification modifié durant cette reprise.

Commande exacte, chemins et empreintes dans le plan. Driver `exit 0`, inspection
indépendante : 24 mesures, huit warmups, zéro erreur. `git diff --check` passe.
Aucun nouveau test de code/build/lint/proveur lancé : sources exécutables
inchangées depuis les suites et vérifications de
[WEB-03](dflash2-exact-optimisations-2026-10-08.md), dont les limites restent
applicables. Les contrôles de métadonnées ne constituent pas une preuve numérique
ou formelle supplémentaire.

Le microgain CPU de **24,9 %** reste celui d'une recherche de clé,
**3,50 → 2,63 µs**. Sa part du temps de génération n'est pas mesurée, donc aucun
facteur de conversion vers un gain global de 25 % n'est déduit. KEY-VIEW reste
dans le code/bundle local avec sa qualité ciblée et son microgain CPU validés ;
son gain de génération demeure non qualifié.

[Plan](measurements/dflash2-key-view-tps-isolated-2026-10-10/plan.json),
[contrôles de la garde](measurements/dflash2-key-view-tps-isolated-2026-10-10/guard-self-check.json),
[rapport brut](measurements/dflash2-key-view-tps-isolated-2026-10-10/abba/results.json),
[log](measurements/dflash2-key-view-tps-isolated-2026-10-10/raw-evidence.tar.gz),
[décision](measurements/dflash2-key-view-tps-isolated-2026-10-10/abba-decision.json),
[conditions et identités finales](measurements/dflash2-key-view-tps-isolated-2026-10-10/post-inspection.json),
[clôture](measurements/dflash2-key-view-tps-isolated-2026-10-10/final.json).

Essai clos **non concluant pour la vitesse**, aucun autre essai actif. Preuve
C++ non vérifiée et alerte d'alignement Q4 MPP héritée ouvertes. Pas de GPU 27B,
comparaison MTP, remplacement de l'app personnelle, push ou publication. Les
32 sorties identiques ne prouvent pas la génération sur tous les prompts,
températures, modèles et matériels.

Preuves brutes archivées sans perte pour publication : chaque chemin original
dans les anciens rapports JSON est répertorié dans `raw-evidence-index.json`
et récupérable dans `raw-evidence.tar.gz` de son dossier de mesures. Les sources
historiques et les rapports originaux restent également dans ces archives.
