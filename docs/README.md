# Documentation MLXL3

Point d’entrée pour parcourir le dépôt et ses archives. Les rapports sont classés
par sujet ; leur date, leur révision et leur statut priment sur leur emplacement.

## Commencer ici

1. Lire les [consignes du dépôt](../AGENTS.md) et le [README principal](../README.md).
2. Pour toute optimisation, lire d’abord le [journal `opti.md`](../opti.md),
   puis les rapports et les preuves associés avant de proposer un nouvel essai.
3. Choisir une rubrique ci-dessous ou chercher un modèle, une issue, une version
   ou une date avec `rg -n 'terme' docs/ opti.md`.
4. Vérifier les conditions, les échecs, les limites et le statut réel de chaque
   essai : un prototype ou un résultat ancien ne décrit pas forcément le produit.

## Arborescence

- [Moteur et configuration](engine/README.md)
- [Desktop et interface](desktop/README.md)
- Modèles : [Qwen](modeles/qwen/README.md), [LFM](modeles/lfm/README.md),
  [Ling](modeles/ling/README.md), [Gemma](modeles/gemma/README.md)
- Performance : [audits transversaux et kernels](performance/general/README.md),
  [Flash, DFlash et SSD](performance/flash-dflash/README.md)
- Issues et revues : [issues 163–164](issues/163-164/README.md),
  [autres revues](issues/reviews/README.md)
- Opérations : [mémoire et matériel](operations/memory-hardware/README.md),
  [publication](operations/release/README.md)
- Notes de version : [Desktop](releases/desktop/README.md),
  [moteur](releases/engine/README.md)
- [Mesures et campagnes](measurements/README.md) : preuves brutes classées
  par campagne, avec les fichiers isolés regroupés dans des dossiers dédiés.
- [Couverture native](native-coverage/README.md) : registre historique et
  preuves classées par composant ; vérifier sa date avant de l’utiliser.

Chaque dossier de rapports comporte son propre index. Suivre les références
vers les mesures plutôt que de déduire une validation d’un fichier brut isolé.
Quelques fichiers à la racine redirigent vers les rapports cités par d’anciennes
releases. Le lien `audit-runtime-2026-09-04.md` conserve le chemin utilisé par
des tests.

## Ajouter un document

Placer le rapport dans le dossier du sujet principal et l’ajouter à son index.
Conserver les noms des campagnes, les résultats négatifs et les limites ; mettre
à jour le journal d’optimisation et les liens lorsqu’ils sont concernés.
