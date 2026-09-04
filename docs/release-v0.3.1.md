# MLXL3 Desktop v0.3.1

## Nouveautés

- Support Gemma 4 EXL3 et optimisations decode du moteur intégrés dans Desktop.
- Attention groupée pour les couches Gemma d=512, activée dans le GUI à partir
  de 2048 tokens KV, avec retour au chemin normal pour les autres cas.
- À 16k de contexte sur Gemma 4 26B A4B EXL3 et Apple M5 : environ +9,6 % de
  decode (33,57 → 36,81 tok/s). Gain faible à contexte court ; les résultats
  dépendent du Mac, du modèle et du contexte.
- Aucun speculative decoding ni quantification supplémentaire. Le calcul
  couvre tout le contexte, mais l'ordre des réductions flottantes peut différer.
- Retour au dialogue natif macOS pour les permissions de fichiers.
- Interface et moteur autonome livrés ensemble.

## Installation et mise à jour

Mac Apple Silicon, macOS 26.2 minimum. Télécharge le DMG, ouvre-le et glisse
MLXL3 Desktop dans Applications. Les modèles sont à télécharger séparément.
Aucun Python ou Homebrew n'est nécessaire. Le bundle utilise une signature
ad hoc locale, pas une notarisation Apple Developer.

Depuis v0.3.0, utilise Réglages → rechercher les mises à jour : le même DMG
est téléchargé, vérifié avec son SHA-256 GitHub puis installé au redémarrage.
Les versions antérieures nécessitent cette mise à jour manuelle.

## Validation

86 tests passent, un test ignoré faute de fixture locale. Génération contrôlée
avec le moteur autonome : 2771 tokens de prompt et huit tokens générés,
terminaison normale. Benchmarks comparatifs détaillés dans
`docs/gemma-sdpa512-investigation.md`. La comparaison de tokens sur les prompts
testés ne garantit pas une identité bit à bit sur toutes les générations.
