# T17 — revue finale du diff et des identités

30–31 octobre 2026. Propriété examinée : conformité du diff final au périmètre
T17 préenregistré et conservation des sources locales antérieures, après
FIX-01, CAPTURE-02 et les corrections de lint. Cette revue ne relance pas
l'audit mathématique ni les mesures. Aucun build, test, GPU ou proveur n'a
été lancé ; seul ce rapport est ajouté.

**Aucun défaut bloquant identifié dans le diff relu.** Le candidat reste une
option locale désactivée par défaut. Cette conclusion de source ne vaut ni
validation universelle du moteur ni décision de performance/promotion.

## Références et contrôle des empreintes

Référence : [before.json](before.json), HEAD
`7978fceac30d991a8c06d72114fc932b2443f555`, arbre déjà modifié localement,
T05 conservé OFF et T17 absent. Le SHA Git seul n'est pas la référence de
contenu : les empreintes de ce manifeste sont utilisées.

- **95 sources antérieures comparées : 92 identiques, 3 modifiées, 0 absente.**
- Les trois sauvegardes `native__src__qwen4.rs.before`,
  `native__src__lib.rs.before` et `.github__workflows__rust.yml.before`
  correspondent exactement aux hashes de `before.json`.
- Les **183/183** entrées de [final-identity.json](final-identity.json) et
  les **185/185** entrées de
  [post-xcode-identity.json](post-xcode-identity.json) correspondent à leurs
  fichiers actuels. Les 183 entrées communes sont identiques entre ces deux
  manifestes ; les deux ajouts sont le plan de qualité après Xcode et
  l'inspection des dispatchs, sans changement de source exécutable.
- Les binaires finaux ont été relus et leur SHA256 recalculé. Le CLI
  antérieur T05 `2fb12d79…` est également toujours identique à sa référence.

| Artefact final | Taille, octets | SHA256 |
|---|---:|---|
| CLI `build/issue59-t17-20261029/engine/prototype-engine` | 10 706 272 | `202f96dca5ceed6cd78f70c524ad48b324e2c35a347cd7dc0bcabb11baf141aa` |
| Libtest `build/issue59-t17-20261029/libtest-final/prototype-engine` | 10 961 984 | `a751cf592d9e28622d1c062a9f93213f4c55d8a6dcce0fa8b1c035be16c6b862` |
| Interpréteur du bridge | 34 640 | `30b2a6bf9706029b609327ca5db722cd458239df96efc87e1f8e37beed49db2e` |

Les vérifications ont utilisé des lectures de fichiers, `hashlib.sha256`,
les diffs unifiés des trois sauvegardes et `git status`. Elles ne constituent
pas une nouvelle exécution des artefacts.

## Diff borné par rapport aux sources antérieures

| Source existante modifiée | Avant T17 | Final |
|---|---|---|
| `.github/workflows/rust.yml` | `af22bd6200f014274de371df67c57908d8dbc47d6558a9dce5ef77da36cb3874` | `d6621fa5e5d629a688719ea7b6a0994deb6d006fe9a85fcd76d2585482c3f846` |
| `native/src/lib.rs` | `b6647cad0fdbfc97043fc8b295fb69a7a94fed04a12d84bfa388d4883c736139` | `294ff5e9c1ef74b712f06d745963cbb45fdced7db3956027d877d519a5f2eefe` |
| `native/src/qwen4.rs` | `32f44122598c8e3ed7c54f85e7ce6a99f98fc0413eac5d22350c8d702c3221f1` | `8f74fbb4ffd612993f556bb8a263b196370c365442a977cd858bdda99a0aeef6` |

Quatre fichiers natifs sont nouveaux dans le périmètre final :

| Nouveau fichier | SHA256 |
|---|---|
| `native/src/gdn_epilogue.rs` | `ab0e52de616869f2aa14aa1dd6cdbc7e8dcb12550ae03a5793b7060e5fe2b2e4` |
| `native/shaders/flash_gdn_epilogue.metal` | `1fa5b6ae214104186021ff2aa43a5e7b083e838fb0a410cf886326aaf8f9ca44` |
| `native/src/gdn_epilogue/tests.rs` | `5a4d9cbf5fd0383d1aa925b33f306d526d02ab0a0e646078631529898d853ba0` |
| `native/src/gdn_epilogue/capture_tests.rs` | `fda5cd9cd555ff4d651bdce582f8801417b5558d21f764eed474282bff441d6e` |

Le FFI, `Array`, les kernels de récurrence, le moteur SSD, ses lecteurs,
les projections EXL3 et T05 font notamment partie des sources dont le hash
reste identique à `before.json`. Aucun remplacement global depuis les PR61
ou63 n'apparaît dans ces différences. Les modifications locales de profilage
et les optimisations antérieures contenues dans ces fichiers n'ont pas été
réinitialisées.

## Raccord GDN relu

- `qwen4.rs:188,194–215` ajoute un booléen chargé à la construction de
  chaque `Gdn`. Variable absente ou chaîne exacte `0` : false ; chaîne
  exacte `1` : true ; toute autre valeur, y compris erreur Unicode : erreur.
  Aucun choix automatique par modèle ou par résultat du benchmark.
- En B1, `qwen4.rs:269–291` garde la projection `z` et son cast FP16,
  puis `self.norm.forward(&value)` inchangés. Le bras activé passe
  directement `[normed]` et `z` à `gdn_epilogue::apply`. Le reshape4D
  supplémentaire du stage observé est sous `cfg(test)` ; la projection out
  reçoit la matrice2D attendue. Le bras OFF reprend les mêmes opérations
  stock et le même reshape `[1,6144]` que la sauvegarde.
- En multi-lignes, `qwen4.rs:412–438` garde une norme native par ligne, avec
  les mêmes slices qu'auparavant. Les résultats sont des `Array` FP32
  distincts dans un vecteur hôte, pas une concaténation FP32. Ils sont passés
  directement à `apply(&normed, &z)`. Le bras OFF conserve sa boucle et sa
  concaténation finale FP16.
- `gdn_epilogue.rs:58–100` impose normed FP32, z FP16, rows1..8 et les formes
  `[1,1,48,128]` / `[1,rows,48,128]` avant la requête ; le kernel produit
  directement FP16 `[rows,6144]`. Aucun readback, eval, fence ou concaténation
  n'est ajouté au parcours de production de cet appel.
- Dans le diff GDN, les projections QKV/a/b, `convolve`, l'historique, les
  appels de récurrence, le stockage des états, la construction des tapes et
  snapshots, ainsi que les paramètres de la norme ne sont pas modifiés.
  L'appel `verification_projection(&self.out, &gated)` et le cast de sortie
  restent identiques. H128, les poids, les splits et les réductions de la
  projection out sont dans des sources inchangées.

Le shader final est bien FIX-01, avec `metal::precise::exp`, les deux
intermédiaires FP32 explicites et le produit avec la norme. La mutation
temporaire supprimant la porte n'est pas la source conservée ; son hash
actuel correspond à l'identité finale. La justification numérique et ses
limites restent dans [diag-source-audit.md](diag-source-audit.md).

Le nouveau test GDN dans `qwen4.rs:3657–3803` et les nouveaux modules
regroupent les contrôles T17 ; les anciens tests ne sont pas remplacés par
des assertions de texte seulement. Cette revue constate le raccord, sans
requalifier leur résultat d'exécution.

## CI et portée de la conservation

`lib.rs:26–27` expose le nouveau module sous `mlx`, `test` ou `kani`.
Le diff du workflow contient seulement :

1. Le harnais
   `gdn_epilogue_geometry_checks_symbolic_shapes_and_addresses` ajouté près
   des deux harnais weighted dans `kani-flash-contracts` (ligne42).
2. Le test filesystem CAPTURE-02 exact sur le job macOS compilant `mlx,chat`,
   avec vérification de **1 passé /0 échec /0 ignoré** (lignes347–348).

Les déclenchements PR/push main, l'exposition du solver Kissat, le pin
Kani0.68.0, les harnais préexistants et les commandes de compilation/lint
MLX restent ceux de la sauvegarde. Pour la nouvelle sélection, le nom est
pleinement qualifié, `--exact` est présent et `set -euo pipefail` précède
`tee` : une sélection vide/ignorée ou un échec Cargo ne peut pas être
transformé en succès par cette étape.

Les 95 sources de départ couvrent le moteur et les frontières ciblées,
**pas une sauvegarde complète de tous les fichiers du checkout**. Les autres
documents et résultats préexistants ne sont pas déclarés byte-identiques
sans référence correspondante. L'identité après Xcode confirme néanmoins
l'absence de dérive des 185 fichiers qu'elle répertorie. Aucun run GitHub
d'un commit publiant cet arbre local n'a été vérifié dans cette revue ; le
câblage CI seul n'est pas une CI distante verte.

Cette revue ne modifie ni `opti.md`, ni le code, ni les options du candidat.
Elle n'autorise pas à activer T17 par défaut, installer l'app ou publier.
