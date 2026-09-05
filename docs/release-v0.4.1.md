# MLXL3 Desktop v0.4.1

## Un parcours MCP plus clair

- Les réflexions, appels outils et réponses s'affichent dans leur ordre réel.
  La réflexion qui suit Exa apparaît désormais sous l'appel terminé.
- Le traitement des résultats affiche un statut, le nombre de tokens nouveaux
  et réutilisés, ainsi que le temps d'attente.
- Cette chronologie est conservée à la réouverture des nouveaux échanges.
  Les anciennes conversations restent lisibles, mais leur ordre détaillé
  n'avait pas été enregistré et ne peut pas être reconstruit.

## Prefill EXL3

Un nouveau kernel SIMD fusionne le gather, les scales et la transformation
Hadamard du prefill MoE SwiGLU. Sur Qwen3.6-35B-A3B EXL3 2.49 bpw, la dernière
paire contrôlée donne **+5,7 % de débit prefill**, avec une attente après outil
de **6,48 à 6,13 secondes** pour 3 742 tokens évalués.

Les logits finaux et états du cache sont identiques dans la continuation de
contrôle. Aucun speculative decoding, changement de quantification ou résultat
MCP tronqué n'est introduit. Le chemin Gemma/GeGLU reste inchangé.

Limites : 20 essais sur un Mac M5 24 Gio sur batterie, avec variations de régime.
Ce n'est pas un gain universel ni un gain de decode. Le pic MLX augmente
d'environ 81 Mo sur cette charge. Les essais détaillés et variantes rejetées
sont documentés dans `docs/tool-prefill-rd-2026-09-05.md`.

## Installation et mise à jour

Télécharger le DMG ci-dessous, puis glisser **MLXL3 Desktop** dans Applications.
Le moteur Python/MLX est inclus ; les poids des modèles sont téléchargés séparément.
La version est également détectable depuis les réglages de mise à jour de l'app.

- Mac Apple Silicon, macOS **26.2 ou plus récent**.
- Application signée ad hoc, **non notarisée** : macOS peut demander une autorisation
  à la première ouverture.
- Exa reste préconfiguré ; le choix d'activation MCP reste mémorisé.

Validation : 347 tests Python réussis, 2 fixtures locales absentes ; contrôles
Swift de chronologie/persistance et préférences MCP ; vérification visuelle
du parcours après outil. Un fichier SHA256 accompagne le DMG.
