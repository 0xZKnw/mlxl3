# MLXL3 Desktop 1.4.8 — build 29

## Nouveautés

Le texte reçu en streaming apparaît avec un fondu continu de 120 ms. L’arrivée
d’un nouveau fragment conserve la progression des précédents. Seul un suffixe
borné à 64 caractères estimés est animé ; aucun délai de type machine à écrire
ni timer permanent n’est ajouté.

Le défilement suit la hauteur réellement préparée avec une courte animation.
Le suivi se suspend quand on commence à remonter, pendant la décélération et
pendant la lecture des anciens messages. « Dernier message » reprend le suivi.
La préférence macOS de réduction des animations est respectée.

## Dessin et correctifs

Les portions de texte de même opacité sont dessinées ensemble. Une fois le
texte opaque, le renderer dessine directement les lignes. Les correctifs récents
LaTeX, import, mises à jour et réglages du Desktop 1.4.6 sont conservés.

Le transport et son regroupement de deltas de 50 ms restent inchangés. Le
nombre de caractères source reste une estimation des glyphes formés, notamment
pour les écritures complexes ; le contrat conserve le rendu et les textes.

## Performances

La campagne du code final compare les 32 sources Swift intégrées sur Apple M5,
24 Go, macOS 27.2, sur secteur, avec le **même moteur public 1.4.8** et
Qwen3.6-35B-A3B EXL3 2,49 bpw + tête MTP 4 bits. Huit passages ABBA + BAAB,
48 tokens de warmup, puis deux prompts de 128 tokens, contexte 4096, température
0, top-k 1 et MTP1 donnent les médianes suivantes :

| Prompt | Avant | Après | Écart observé |
| --- | ---: | ---: | ---: |
| Français | 65,76 tok/s | 65,16 tok/s | −0,91 % |
| Code | 67,61 tok/s | 68,88 tok/s | +1,89 % |

Les 16 générations ont le même texte, le même nombre de tokens et les mêmes
compteurs d’acceptation MTP. La dispersion et la dérive entre blocs dépassent
1 % : **la marge stricte de 1 % reste non démontrée**, et aucun gain de débit
n’est revendiqué. Ces chiffres qualifient ces deux prompts, pas tous les modèles.

Les 88 relectures UI couvrent prose, code et raisonnement, de 64 Kio à 1 Mio,
avec cadences de 50 ms, 15 arrivées/s et rafales. La médiane du CPU total
par passage, pour les 11 cas, passe de **7,537 à 8,034 s (+6,6 %)**. Pour la prose 64 Kio à 50 ms,
le CPU passe de **0,501 à 0,625 s** par relecture de 24 appends, et l’empreinte
du processus UI de **168,0 à 182,0 Mio**. Le fondu continu a donc un coût mesuré ;
le regroupement du dessin ne rend pas l’ensemble gratuit.
Sur la prose de 1 Mio à 50 ms, le surcoût CPU médian observé est de 45,2 %
(0,519 → 0,754 s), avec une forte dispersion de la baseline.

Les mesures de délai d’arrivée concernent l’ordonnancement des callbacks ; elles
ne mesurent pas la présentation de chaque pixel. FPS et énergie sont **non
mesurés**. Une première tentative sans fenêtre visible a été rejetée avant tout
échantillon et conservée ; la série complète a ensuite été effectuée sur le bureau
standard, avec les gardes de visibilité et de concurrence inchangés. Les anciens
essais APP-02 restent archivés séparément et ne sont pas additionnés à cette série.

## Installation et compatibilité

**MLXL3-Desktop-v1.4.8-b29-Apple-Silicon.dmg**, Apple Silicon arm64,
macOS **26.2+**, MLX **0.32.2**. Signature ad-hoc, sans notarisation Apple.
Les modèles et conversations existants sont conservés.

Le bundle réutilise le moteur **1.4.8 déjà publié**, sans recompilation ni
modification de ses fichiers. Les versions Desktop et moteur sont indépendantes.
Le numéro Desktop est aligné sur le moteur pour que les anciens clients qui
exigent encore leur égalité acceptent aussi le paquet. Avec certains anciens
clients et LLVM lipo, une installation manuelle du DMG peut rester nécessaire.
Le canal moteur [engine-v1.4.8](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.8)
reste publié avec `latest=false` ; cette publication Desktop fournit le DMG stable.
Aucun remplacement de l’app utilisateur n’est exécuté par nos tests.

## Vérification et limites

Les tests de régression ajoutent 786 cas finis de fondu, 63 comparaisons natives
de dessin, l’ancrage et l’interpolation du scroll, ainsi que la politique de
tracking, lecture et reprise. Une mutation qui redémarre les âges précédents
est rejetée. Les tests existants de LaTeX, copie complète, bridge, annulation et
historique long sont conservés.

La suite locale `PYTHONPATH=src scripts/check-e2e.sh` réussit avec **840 tests
Python passés et 115 sautés**, ainsi que tous les contrôles Desktop. Elle utilise
les mêmes 32 sources Swift de production que le paquet final ; elle a été
exécutée sur le candidat 1.4.7 avant la correction du numéro de version. Les
contrôles affectés ont ensuite été rejoués sur le DMG 1.4.8 : signature stricte,
manifestes, quatre empreintes runtime, compatibilité, installation jetable et
15 cas updater signés. Le vrai préflight de l’ancien client 1.3.0 accepte aussi
ce paquet ; il rejetait le candidat 1.4.7 avec moteur 1.4.8.

`swiftc -swift-version 6 -warnings-as-errors -Xfrontend -sil-verify-all` et
l’exécution du contrôle natif de réponse réussissent. Ce sont des contrôles de
compilation et des tests, sans preuve formelle du comportement SwiftUI.
CrossHair n’est pas disponible pour le contrôleur Python. Le test physique de
buffer Metal est explicitement sauté ; aucun code moteur n’a changé. Les gestes
physiques, les autres Mac, modèles et contextes longs, les FPS et l’énergie ne
sont pas requalifiés.

La [CI Desktop du commit exact](https://github.com/0xZKnw/mlxl3/actions/runs/38953567273)
est **réussie** : 840 tests Python passés, 115 sautés, puis toute la suite native
Desktop, dont les 786 cas et 63 images. La [CI de conversion](https://github.com/0xZKnw/mlxl3/actions/runs/38953567239)
est également réussie. Les jobs Rust/Kani sont relevés avec leur statut réel
dans la validation jointe ; aucun succès global de ces jobs encore en cours
n’est annoncé pour cette modification exclusivement Desktop.


Actualisation après publication : le [job Kani du moteur hérité](https://github.com/0xZKnw/mlxl3/actions/runs/38953567272)
a terminé en **échec par interruption (code 143)** pendant
`dflash_copy_len_10`, après 65 harnais réussis. Les obligations déjà
rapportées n’affichent aucun échec ; le harnais interrompu et la suite complète
restent **non vérifiés**, sans cause d’interruption établie. Les jobs Rust Linux
et macOS sont réussis. Les sources du moteur et de Kani sont identiques à la
baseline ; aucun changement ni nouvelle exécution du moteur n’est ajouté ici.

## Sources et artefacts

- Source compilée et tag Desktop :
  [`e032a8e814375dea3bedc529930e1918ff401306`](https://github.com/0xZKnw/mlxl3/commit/e032a8e814375dea3bedc529930e1918ff401306),
  aucun fichier suivi modifié au moment de l’assemblage.
- DMG : **73 423 924 octets**, SHA-256
  `ca5dbea362c7709cc36b834684fcd091230f5be457822de330b17a7ffccea151`.
- Runtime public conservé : source
  [`26987a2c1aa9cc10a1908cdb6ffbec0eca55ccef`](https://github.com/0xZKnw/mlxl3/commit/26987a2c1aa9cc10a1908cdb6ffbec0eca55ccef),
  archive **66 959 926 octets**, SHA-256
  `2c79f9b13b7e28f863c6156c45eb88b66cd10565f8f111af0a7555a3635aba44`.

Le fichier joint `MLXL3-Desktop-v1.4.8-validation.json` contient les identités
publiques, le checksum du DMG, les comptes de tests, les liens CI et les mesures
agrégées. Les journaux bruts, sources de preuve et métadonnées locales sont
conservés sur la machine ; ils ne sont pas joints aux téléchargements publics.
Les résultats de publication font l’objet d’un compte rendu séparé du commit compilé.

## Remerciements

- [@0xZKnw](https://github.com/0xZKnw), demandes, intégration et validation sur Mac.
- [SwiftMath](https://github.com/mgriebling/SwiftMath), rendu mathématique conservé ;
  [MLX](https://github.com/ml-explore/mlx), runtime Apple Silicon du moteur embarqué.
- [Qwen](https://github.com/QwenLM), modèles utilisés pour les contrôles. Les
  attributions et licences amont du moteur restent dans sa release et le dépôt.

## Fichiers téléchargeables

|Fichier|Octets|SHA-256|
|---|---:|---|
|[MLXL3-Desktop-v1.4.8-b29-Apple-Silicon.dmg](https://github.com/0xZKnw/mlxl3/releases/download/v1.4.8/MLXL3-Desktop-v1.4.8-b29-Apple-Silicon.dmg)|73423924|`ca5dbea362c7709cc36b834684fcd091230f5be457822de330b17a7ffccea151`|
|[MLXL3-Desktop-v1.4.8-validation.json](https://github.com/0xZKnw/mlxl3/releases/download/v1.4.8/MLXL3-Desktop-v1.4.8-validation.json)|3394|`e1daed2dff4a457749a902f390091440cb0aaac740c3f569b9cb913f7c1fae19`|
