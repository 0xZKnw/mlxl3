# Reprise du cache de Flash Next dans l’Agent

Essai `FLASHNEXT-AGENT-LIVE-01` du 26 novembre 2026 sur un M5 24 Gio. Le moteur MLX 2.0.39 et le prototype utilisent le checkpoint installé `Qwen3.8-Flash-Next-exl3-84147e95` en décodage standard, MTP et DFlash2 désactivés. Le budget de cache SSD est de 3 Gio, le contexte de 4 096 et le budget total de 192 tokens. Température 0 et raisonnement désactivé.

Le script utilise le vrai pont et l’outil local de Studio. Il crée trois fichiers jetables, n’autorise que leur lecture et vérifie les trois appels, leur ordre, leurs résultats et la réponse `TOTAL=18`. La durée commence après le chargement du modèle et la préparation des outils. Le compteur de tokens évalués distingue la relecture de la reprise effective.

## Reproduire

Utiliser un registre contenant l’entrée du modèle local. Le script en extrait une copie isolée pour chaque processus. Définir `ENGINE` vers le moteur à mesurer, `ENGINE_BUNDLE` vers les bibliothèques du moteur livré, `HELPER` vers l’exécutable Studio et `REGISTRY` vers le registre. Lancer un seul moteur à la fois, après les compilations. Pour le témoin sur le même binaire, retirer `--live-cache --require-live`.

```sh
DYLD_LIBRARY_PATH="$ENGINE_BUNDLE" \
MLXL3_METALLIB_PATH="$ENGINE_BUNDLE/mlx.metallib" \
MLXL3_SSD_CACHE_GIB=3 \
"$PYTHON" docs/measurements/flashnext-agent-cache-2026-11-26/flash_agent_trial.py \
  --engine "$ENGINE" --helper "$HELPER" --registry "$REGISTRY" \
  --model Qwen3.8-Flash-Next-exl3-84147e95 --ready-timeout 180 --task-timeout 900 \
  --label candidate --live-cache --require-live --output /tmp/flash-agent-candidate.json
```

`PYTHON` doit désigner un interpréteur dont l’environnement conserve `DYLD_LIBRARY_PATH`. L’oracle physique est `qwen4::tests::flash_agent_live_tool_prefix_matches_cold_prefill`, à exécuter avec les features Rust `mlx,chat`, `MLXL3_FLASH_MODEL` vers le checkpoint et les mêmes bibliothèques et réglages. `live-start-kani.log` couvre la décision de reprise sur un domaine borné CPU. `gpu-oracle.log` couvre les états et logits de l’échange synthétique sur le M5.

Pour vérifier le même échange à partir de la capacité que le moteur annonce à Studio, remplacer `--live-cache` par `--advertised-live-cache` et garder `--require-live`. Le harnais attend l’événement `ready`, exige que la capacité soit annoncée pour le modèle chargé et ne demande la reprise qu’une fois ce contrôle passé.

## Mesures sur M5

| Passage | Tâche (s) | Préremplissage (s) | Décodage (s) | Tokens évalués | Pic processus (Gio) | Reprises |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [reference-3-gib](reference-3-gib.json) | 359.66 | 343.89 | 15.57 | 1859 | 6.845 | 0 |
| [B1](B1.json) | 192.38 | 166.34 | 25.58 | 459 | 6.864 | 3 |
| [B2](B2.json) | 95.62 | 84.05 | 11.41 | 459 | 6.856 | 3 |
| [A2](A2.json) | 385.15 | 370.02 | 14.93 | 1859 | 6.867 | 0 |

La comparaison A2/B2 utilise le même binaire et les mêmes bibliothèques. Les deux passages démarrent sur batterie. B2 prend 75.2 % de temps en moins. Les quatre passages ont le même transcript final et chaque reprise réussie repart après 416, 467 et 517 tokens. B1 subit une première lecture beaucoup plus lente. Cette dérive limite la généralisation de la durée mesurée.

L’essai initial au budget automatique est conservé dans [reference.json](reference.json) et [son stderr](reference.stderr.log). L’admission exigeait 8,56 Gio alors que 7,06 Gio étaient disponibles. Le budget de 3 Gio du cache d’experts permet de charger la même quantification.

Les rapports publiés remplacent les chemins locaux par des variables symboliques. Les identifiants de processus et de batterie ainsi que les empreintes dérivées des chemins locaux ont été retirés. Les traces synthétiques, mesures, empreintes des binaires et résultats des vérifications sont conservés.
