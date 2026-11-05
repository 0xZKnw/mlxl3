# DFlash2 : premier contrôle du débit réel, 9 octobre 2026

Clôture le **10 octobre 2026 : non concluant**. Le gain de 24,9 % porte sur
une recherche de clé CPU, de 3,50 à 2,63 µs. Aucun gain de génération en tokens/s
n'est établi par cette campagne. Les valeurs partielles ci-dessous ne permettent
pas davantage d'attribuer une régression au moteur.

## Mesures partielles

| Prompt | Ancien moteur A, tok/s | Nouveau moteur B, tok/s | Écart observé, tok/s |
| --- | ---: | ---: | ---: |
| Court français | 55,940 | 41,220 | −14,720 |
| Code Python | 62,495 | 48,910 | −13,586 |

Ce sont les médianes de trois réponses de 128 tokens lors du **premier passage**
de chaque variante. L'ordre ABBA n'a pas été terminé : le warmup de B2 dépasse
la limite de 90 secondes, puis le contrôleur termine et rejoint l'enfant.
A3 n'est pas exécuté. On ne peut donc calculer les dérives entre deux passages
d'une même variante ni qualifier le gain. BAAB n'a pas été lancé.

Les **12 réponses mesurées et quatre warmups terminés** ont des métriques finies,
des comptes valides et les mêmes empreintes de tokens, texte et historique pour
leurs prompts/budgets respectifs. Cela ne certifie pas la campagne incomplète :
son statut reste `failed`, sa parité globale `null`. Le rapport brut conserve
le `TimeoutError`. Le contrôleur externe a en outre enregistré un
`AssertionError` en tentant de traiter ce rapport incomplet ; ce second diagnostic
ne remplace pas l'erreur d'origine.

## Conditions et invalidation du protocole

Le swap utilisé passe de 2 938,06 à 6 141,62 puis 7 352,81 Mio aux relevés des
passes. Les réponses du court français B1 varient de 56,80 à 15,95 tok/s.
Le niveau du swap ne mesure ni la RAM résidente du moteur ni la fréquence des
échanges pendant chaque réponse.

Le 10 octobre, l'utilisateur confirme avoir lancé MLXL3 en parallèle par
accident. Un autre `cargo-kani` a également démarré le 9 octobre à 23:07:03,
avant la campagne de 23:07:58 à 23:12:27 ; un `cbmc` démarré à 23:11:58 chevauche
sa fin. Les horaires observés sont en Europe/Paris. La garde du contrôleur a
enregistré à tort une isolation suffisante (`conflicts_before: []`). Cette
contradiction reste explicitement conservée dans les preuves.

L'exécution concurrente et les variations de mémoire rendent la comparaison
impropre à l'attribution d'un gain ou d'une perte. La contribution respective de
MLXL3 et du proveur au swap, au timeout et aux ralentissements n'est pas mesurée.
Le proveur est un processus séparé ; il ne fait pas partie du runtime livré.

## Identité et protocole

Base `db52b7304a057ad8f0d8fcf464cd66b22b44fe08`, diff local préexistant conservé.
A signé : `ad0bef499cd1850795d5ad9c8a54055d91299e9564fa55ebeb898e7ffde03c64`.
B signé : `23e0f3c16ec08a54be6b54ee83009f16e4a1e3be43253a609088181661c55e69`.
Les deux variantes chargent les mêmes bibliothèques ; identités des binaires,
des bibliothèques, du driver et de 12 sources contrôlées après l'essai,
sans différence.

Qwen3.6-35B-A3B EXL3 2,49 bpw, draft DFlash2 packed Q4 ; M5 Air 24 Gio,
MLX 0.32.2/macOS 27.2. Batterie autorisée, 42 % avant campagne,
41/39/38 % aux relevés des passes. `pmset` ne relève pas d'alerte thermique ;
températures et fréquences exactes non mesurées.

Driver existant `benchmarks/compare_native.py` inchangé : mode DFlash Auto,
MTP OFF, `MLXL3_DFLASH_CONTEXT_COPY=0`, `MLXL3_QWEN_PIPELINE=0`,
`MLXL3_MTP_LOOKUP=0` pour A et B, contexte 4096, greedy, sans réutilisation du
cache, MCP ni outils. Un warmup de 256 tokens par prompt et par passe,
trois mesures de 128 tokens, repos fixe de 30 secondes avant chaque passe.
ABBA unique, BAAB seulement si tous les seuils passent : gain code ≥3 %,
régression du court ≤1 %, toutes les dérives entre passes ≤1 %.
La commande complète et les empreintes sont dans le plan.

## Preuves et état

[Plan préenregistré](measurements/dflash2-key-view-tps-2026-10-09/plan.json),
[rapport brut](measurements/dflash2-key-view-tps-2026-10-09/abba/results.json),
[snapshot de l'échec](measurements/dflash2-key-view-tps-2026-10-09/abba/results-at-failure.json),
[log](measurements/dflash2-key-view-tps-2026-10-09/raw-evidence.tar.gz),
[inspection indépendante des résultats](measurements/dflash2-key-view-tps-2026-10-09/inspection.json),
[clôture](measurements/dflash2-key-view-tps-2026-10-09/final.json).

Aucun changement exécutable, build ou test de code supplémentaire pendant cette
campagne ; les tests et tentatives de preuve du code conservé sont ceux de
[WEB-03](dflash2-exact-optimisations-2026-10-08.md). L'optimisation de recherche
CPU reste dans le code et le bundle locaux avec son microgain qualifié.
Preuve C++ non vérifiée, alerte d'alignement MPP héritée ouverte. Pas de GPU 27B,
comparaison MTP, push, publication ou remplacement de l'application personnelle.
Cet essai est clos. Une nouvelle mesure doit être préenregistrée avec la raison
concrète de corriger l'isolation, sans masquer cet échec ni relancer pour bruit.

Révision du 10 octobre : la
[reprise avec isolation corrigée](dflash2-key-view-tps-isolated-2026-10-10.md)
termine ses 32 réponses exactes. Les écarts de débit restent indicatifs car
les variations entre passes dépassent le seuil prévu. Les deux campagnes
utilisent des conditions différentes et ne sont pas mélangées.

Preuves brutes archivées sans perte pour publication : chaque chemin original
dans les anciens rapports JSON est répertorié dans `raw-evidence-index.json`
et récupérable dans `raw-evidence.tar.gz` de son dossier de mesures. Les sources
historiques et les rapports originaux restent également dans ces archives.
