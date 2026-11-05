# Qwen3.6-35B-A3B sans MLX sur M5

La branche part de `6030540b5573185a4bf8b13207011872d94d0045`, qui inclut le portage Qwen de la PR #128. La qualification utilise le checkpoint EXL3 2,49 bpw sur Apple M5 24 Gio. Le modèle est chargé par le runtime Rust/Metal. Les références historiques viennent de la branche expérimentale locale, avec leur propre binaire. [Issue #144](https://github.com/0xZKnw/mlxl3/issues/144) · [Portage initial](https://github.com/0xZKnw/mlxl3/pull/128)

| Piste | Couverture sur la branche |
| --- | --- |
| 1 | P01, P02 et P05 groupent les commandes MoE et chaînent les couches. P07 retire la frontière hôte du KV. |
| 2 | P07 conserve le KV en buffers Metal, avec préfixes immuables, croissance bornée et copie sur branche. |
| 3 | P09 groupe les QMV, puis P10 groupe la préparation, le SwiGLU et la finition des huit experts. |
| 4 | Les indices du routeur passent encore par le CPU. Une banque de poids résidente adressable par les shaders reste nécessaire pour retirer cette frontière à coût borné. |
| 5 | P08 calcule les logits de la dernière ligne du suffixe prérempli. |
| 6 | P03 vérifie plusieurs lignes dans un bloc causal. Les lignes ne partagent pas encore la lecture des poids du tronc. |
| 7, 9, 12 | Réemploi des frames GDN, partitionnement de l’attention et fusion Q/K/RoPE restent des pistes séparées. |
| 8 | P11 sélectionne sur GPU en greedy, avec la normalisation Half et le rejet Qwen des logits bruts non finis. |
| 10, 11, 13–15 | Des encodeurs de l’expert partagé, des projections et du GDN sont déjà enchaînés. Les fusions de kernels et les regroupements propres à ces formes restent des essais distincts. |
| 16–18 | LUT EXL3 et géométries spécialisées restent à qualifier sur Qwen. P10 place les temporaires d’une étape dans des buffers contigus par commande. |
| 19–21 | Points de reprise supplémentaires, embeddings et fusion softmax/top-8 restent à mesurer. |
| 22–23 | MTP natif est disponible sur option explicite. Le réglage de débit retenu reste la génération ordinaire. DFlash Qwen reste une piste séparée. |
| 24 | La mutualisation des pipelines GDN concerne le temps de chargement. |

P10 réutilise les deux QMV groupés de P09. Trois nouveaux lancements traitent la préparation des 16 projections gate/up, le SwiGLU des huit routes et la finition des huit down. Chaque job écrit dans sa tranche. Les shaders partagent les corps de calcul SwiGLU et de finition avec le chemin unitaire et conservent les mêmes frontières d’arrondi. [Arguments Metal Tier 2](https://developer.apple.com/documentation/metal/improving-cpu-performance-by-using-argument-buffers)

P11 joint la norme finale, la tête, le logsumexp et la sélection au dernier buffer de commandes. Le GPU publie un index de quatre octets et le moteur publie l’état après validation de cet index. Le préremplissage, l’échantillonnage, les sorties structurées, les outils et la spéculation conservent leurs parcours. [Journal des essais](../opti.md)

## Qualification

Le commit exécutable `3b3426074b8b938ab727c2f13edb3e1d7ce7c68e` est la source du binaire release Metal sans MLX. SHA-256 du CLI `5e1384c721ca04adefda44c36957a5050ead9da306c77e506b8cd6c1fec9627e`. Le commit qui ajoute ce rapport change uniquement la documentation.

| Contrôle local | Passés | Ignorés |
| --- | ---: | ---: |
| Rust CPU | 396 | 9 |
| Rust natif et chat | 415 | 107 |
| Rust MLX et chat | 451 | 326 |
| Python | 106, plus 2 sous-tests | 0 |
| Sélections physiques ciblées | 10 | 0 parmi les sélections exécutées |

Formatage et Clippy strict passent sur les trois variantes Rust. Les neuf harnais Kani passent dans leurs domaines bornés, avec 676 obligations, une inatteignable et 32 couvertures. La mutation autorisant un job de plus est détectée par l’assertion de borne. Le harnais et les tests passent après restauration. Le premier lancement Kani échouait à typer un harnais et le diagnostic a été conservé.

L’oracle MLX indépendant vérifie bit à bit les logits et les états du tronc, y compris les branches du préfixe. La vérification groupée contrôle 10 181 120 mots de logits finis et 362 762 240 mots d’état finis. P10 donne 8 192 mots FP16 exacts face au chemin antérieur sur deux couches réelles. Le bridge valide les sorties, l’annulation, la reprise, le cache, le MTP et le repli sur pénalité non unitaire. Le bras de référence avec toutes les options Qwen à zéro passe également l’oracle sur quatre pas, 993 280 logits et 65 966 080 mots d’état.

Les shaders compilent et s’exécutent sur Metal. Le compilateur Metal autonome est absent sur ce poste et CBMC ne lit pas le MSL. Les shaders n’ont pas de preuve formelle. CrossHair était indisponible pour le contrôle Python. Les tests ci-dessus mesurent le binaire local et couvrent le cas M5 testé. Les jobs GitHub du code de la branche restent à exécuter au moment de la qualification locale.

## Protocole de mesure

Les trois campagnes alternent A/B/B/A sur le même binaire. Chaque processus charge seul le modèle, puis chauffe chaque prompt sur 8 tokens et en génère 64. Les deux entrées comptent 27 et 35 tokens, contexte maximal 4096, température 0 et pénalité 1. Le cache de prompt, le MTP, DFlash et les outils sont éteints. Le débit de décodage compte les 63 tokens qui suivent le premier. Les temps de chargement du modèle sont hors mesure.

Le pilote est `benchmarks/compare_native.py MODEL --baseline CLI --candidate CLI --output RAPPORT --prompt-file docs/measurements/dflash2-key-view-tps-isolated-2026-10-10/code-prompt.txt --tokens 64 --warmup-tokens 8 --repeats 1 --order ABBA --settle-seconds 3 --load-timeout 90 --request-timeout 120 --shutdown-timeout 10 --campaign-timeout 420`. Les options de chaque bras sont transmises par `--baseline-env` et `--candidate-env`. Les échantillons et les options exactes sont dans le [résumé des mesures](measurements/qwen36-144-p10-20261105/summary.json).

Pour la composition retenue, mettre les variables suivantes à 1 sur le bras B et à 0 sur le bras A.

- `MLXL3_NATIVE_QWEN_MOE_JOIN`
- `MLXL3_NATIVE_QWEN_LAYER_JOIN`
- `MLXL3_NATIVE_QWEN_LAYER_CHAIN`
- `MLXL3_NATIVE_QWEN_DEVICE_KV`
- `MLXL3_NATIVE_QWEN_PREFILL_LAST_HEAD`
- `MLXL3_NATIVE_QWEN_GROUP_EXPERT_QMV`
- `MLXL3_NATIVE_QWEN_GROUP_EXPERT_STAGES`

Les deux bras fixent `MLXL3_NATIVE_QWEN_DEVICE_GREEDY=0`, `MLXL3_NATIVE_DEVICE_GREEDY=1` et `MLXL3_NATIVE_QWEN_MTP=0`. Les chemins expérimentaux de la branche sont sur option explicite.

## Résultats sur le M5

| Comparaison | Français, tok/s | Code, tok/s | Décision |
| --- | ---: | ---: | --- |
| P10 regroupement complet des experts | 13,76 → 15,56 | 14,30 → 15,88 | Retenu sur option |
| P11 sélection greedy GPU sur P10 | 14,85 → 15,49 | 16,37 → 15,73 | Option inactive dans le réglage retenu |
| Composition face au chemin natif de référence | 5,12 → 14,65 | 5,17 → 14,86 | Retenue sur option |

Le débit médian de la composition est environ 2,9 fois celui de la référence sur ces deux entrées. Les deux mesures B se situent entre 13,78 et 15,52 tok/s en français et entre 14,14 et 15,58 tok/s sur le code. Le premier token médian arrive à 1,40 s contre 5,19 s en français et à 1,88 s contre 6,71 s sur le code. Les huit générations et les huit chauffes reproduisent les tokens, le texte et l’historique. Le Mac est sur secteur, batterie de 43 à 46 %, sans avertissement thermique relevé.

L’allocation Metal passe de 11,264 à 11,267 Go et les pics d’empreinte du processus de 11,803–11,807 à 11,983–11,989 Go. Les répétitions du bras optimisé varient de 10 à 13 % et le critère de dérive du pilote ne qualifie pas le pourcentage exact. Les échantillons restent nettement séparés. Cette campagne couvre des prompts courts et ne mesure pas le débit soutenu en contexte long.

Pour P10, les quatre mesures optimisées dépassent leurs témoins et les médianes atteignent 15,56 et 15,88 tok/s. La dispersion du témoin atteint 10,80 % en français et 5,88 % sur le code. Pour P11, le bridge compte 63 têtes GPU sur 64 tokens, ce qui confirme le parcours mesuré. Le logsumexp et le départage parcourent toujours le vocabulaire complet. Les mesures en français se chevauchent et les deux mesures code se dégradent. Le bénéfice de P11 reste à établir.

L’objectif de 100 tok/s reste ouvert. Les pistes 4 et 6 concernent encore deux frontières du chemin retenu, le retour CPU des huit IDs de routage et la relecture des poids lors de la vérification de plusieurs lignes. [Mesures, options et contrôles](measurements/qwen36-144-p10-20261105/summary.json) · [Issue #144](https://github.com/0xZKnw/mlxl3/issues/144)
