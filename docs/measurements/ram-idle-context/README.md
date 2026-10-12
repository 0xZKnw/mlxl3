# Preuves de la campagne RAM / contexte

Base officielle : v1.4.2 (`9f2a572e22a1560399ac8d77a3088573ed8e74cb`).
Pas de modification des poids ni de l'app installée. Registres/prompts de test
jetables ; les PDF et conversations privés ne figurent pas dans les mesures.

- `installed-{ui,engine}-loaded.txt` : vmmap des processus installés avant test.
- `installed-ui-unloaded.txt` et `installed-processes-unloaded.txt` : après
  éjection par l'interface, UI seule et disparition du moteur.
- `identity.json` : exécutables et empreintes, métadonnées des checkpoints,
  sources de production et versions de driver distinguées par lancement.
- `moe-{ab,ba}.json` et `moe-summary.json` : deux ordres complets du vrai bridge
  Qwen3.6-35B-A3B, avec 36 cas et comparaison des oracles entre les quatre moteurs.
- `dense-ab.json` : **échec conservé**, stock seulement, six cas courts terminés,
  timeout du préfill long à 180 s ; aucune parité ni réduction RAM dense validée
  par ce fichier. Son stderr temporaire n'a pas été exporté avant nettoyage ; le
  driver v2 corrige ce manque avec un test négatif exerçant un vrai processus.
- `dense-{ab,ba}-v2.json` : reprises identiques avec délai d'échange 300 s,
  campagne 1200 s et contrôleur 1230 s ; résultats distingués de l'essai interrompu.
- `*-command.json` : commandes/options réelles, délais, pid possédé, sortie,
  durée et conditions power/swap/therm avant/après. Températures/fréquences GPU
  non mesurées. La fin réussie du contrôleur suit la fermeture et le join du child.
- `driver-v{1,2}.py.txt` : scripts exacts des deux protocoles, sans reconstruire
  les commandes après les résultats. `owned-controller.py.txt` garde le watchdog.
- `gpu-{cache,snapshot}-check.*` : tests physiques sélectionnés explicitement,
  1/1 test chacun, tableaux vivants et restauration des logits/états exacts.
- `kani-{targeted,full}.*` : seuil symbolique u64 (4 couvertures) et 40 harnais
  CPU passés. Pas de preuve des bibliothèques externes MLX/Metal ou de concurrence.
- `python-full.*`, `rust-full.*` : premières suites sous sandbox interrompues
  faute de device Metal ; `*-physical.*` et `python-full-final.*` conservent les
  reprises sur Mac réel. Final : 319 Python passés/4 skips, 68 Rust passés/60
  ignorés, dont deux ignorés sélectionnés et passés séparément sur GPU.
- `driver-tests*`, `driver-lint*`, `clippy.*`, `fmt.*` : tests/format/lint ciblés,
  diagnostics initiaux et corrections conservés. 50 tests driver/transport passent.
- `crosshair.*` : outil Python absent, propriété non vérifiée formellement.

Les statistiques `generation_memory` sont prises pendant la génération, avant
la libération au repos. `idle_bytes` utilise le footprint macOS après pong ;
`idle_memory` du candidat décrit alors les allocations MLX actives et inutilisées.
Les timings sont diagnostiques et variables sur batterie. Ni un gain de vitesse
ni une baisse du pic de préfill ne sont revendiqués.

Les logs bruts peuvent être compressés sans perte en `.gz`. Le manifeste final
inclut leur taille et SHA-256 ; les identités des originaux seront également
conservées pour vérifier leur décompression à l'identique.
