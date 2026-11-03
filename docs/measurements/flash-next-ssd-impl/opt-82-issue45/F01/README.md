# 82F01 — sortie QMV FP16 après réduction FP32

**État final : correction numérique validée sur le domaine testé, débit non concluant, prototype effectivement retiré le 25 octobre. Aucun gain F01 livré.** Le retrait a appliqué le patch inverse exact des seuls hunks F01 archivés après contrôle des 83 sources. Deux fichiers de tests non référencés ont été préservés ; ils ne sont plus compilés. Les sources privées et l’application installée sont inchangées. Les archives ne décrivent pas une option active du moteur actuel.

## Débit réel mesuré

M5 10 cœurs / 24 Gio, Flash Next 2,05 bpw `65c89531`, draft Q4 `a22a1744`, MLX 0.32.2. Secteur, cache 3 Gio / 53 slots, allocateur 64 Mio, DFlash2 fixe B2. Chaque processus produit 128 tokens de warmup puis 128 tokens mesurés, dont 127 de decode. Seul `MLXL3_SSD_QMV_HALF_OUTPUT=0|1` différait dans le prototype, avec le même CLI `19c9e1f2` et le contrôleur `8935` corrigé par 82B.

| Prompt | H puis C (tok/s) | C puis H (tok/s) | Géomoyenne des rapports C/H |
|---|---|---|---|
| MAIN | 11,138480 → 10,633186 (−4,5365 %) | 11,355970 → 11,084252 (+2,4514 %) | −1,1042 % |
| HOLD | 10,169871 → 10,321749 (+1,4934 %) | 10,041335 → 10,417939 (−3,6150 %) | −1,0937 % |

Les signes changent avec l’ordre. Aucun gain reproductible ni régression causale universelle de 1,1 % ne sont établis. Les seize réponses, textes et hashes correspondent exactement à l’oracle gold60 ; nombres 128/127, acceptations, largeurs et octets SSD logiques sont identiques entre bras par phase. Les dérives warmup→mesure restent visibles, dont +5,72 % pour le premier HOLD-H. Aucun débit OFF/MTP1 supplémentaire n’a été lancé faute de gain à promouvoir. Les 15 tok/s ne sont pas atteints.

## Qualité et vérification

- CPU : 117 tests lib réussis / six ignorés, plus seize tests d’intégration ; format, Clippy CPU/MLX et builds CLI/natif inspectés. La suite native physique standard indique 130 réussis / 149 ignorés ; les tests physiques concernés ont ensuite été sélectionnés explicitement.
- Kani sur le vrai garde : trois obligations réussies, quatre couvertures satisfaites, booléens et domaine i32 complets sans hypothèse. Ce résultat ne prouve ni Metal, ni MLX, ni la concurrence ou les arrondis GPU. Le mutant raw a été détecté, puis la source restaurée et revérifiée. L’échec du premier sélecteur de mutation reste conservé.
- 253 962 valeurs de frontière F16, incluant ±0, sous-normaux, max finite, overflow, Inf et NaN canoniques : store Metal et cast MLX byte exacts sur cet ensemble, pas sur tous les motifs F32.
- 240 cas QMV synthétiques, dont 200 fast paths F16 et 40 fallbacks multisplit F32 réellement exercés ; partiels avant H128 et sorties exacts. Puis 64 cas MoE réels sur quatre couches × deux dtypes × quatre largeurs × deux tours Cold/Hot, avec les vrais partiels GU/down.
- L’oracle historique 176 cas / 528 masques / 112 groupes est exécuté avec `half=false` : régression de référence, pas des cas F01 supplémentaires.
- 36 préfixes DFlash et session MTP à quatre budgets : captures, logits et états exacts, avec flag 1 commun aux voies comparées. Le complément v3 compare directement flags 0/1 sur les 48 couches et quatre tokens fixes, tous les 248 320 logits ainsi que les états GDN/QSA/PLE/hidden/offset à chaque pas, byte exacts, avec un seul modèle.
- Bridge : 43 requêtes exactes, erreurs attendues, annulation et reprise. Son `JsonProcess` ne certifie pas les événements tardifs de fermeture ; aucune erreur tardive observée. Le contrôleur TPS 82B vérifie explicitement sa fermeture et sa file finale.

**Échec conservé :** quality-v1 est FAILED parce que le test B4/B8 appelait une entrée limitée à deux lignes. Le test a été corrigé vers `forward_draft_rows` limité à huit lignes ; seuls les contrôles affectés et non encore lancés ont été exécutés en v2. Les autres cas précédemment réussis ne transforment pas le statut global v1 en succès.

## Mémoire et limites

HC et CH ont zéro nouveau swap-in/out système dans chacune de leurs six phases, avec des pics de durée de vie processus de 7,825 et 7,845 Go. Le ledger de swap-in propre vaut zéro aux snapshots ; cela ne prouve pas l’absence de swap-out privé.

Le bridge présente 70 observations mémoire, incluant des doublons stats/round_stats/ready : pic 7,913 Go et ledger swap-in propre zéro, **mais 268 173 312 octets de swap-out système (255,75 Mio)** pendant sa phase, sous la garde de 256 Mio. L’attribution de ces pages reste inconnue. Les zéros des pré/post ne remplacent pas ce delta. L’audit initial annonçant zéro à tort est archivé puis explicitement corrigé. Les caps de compression 77B restent non qualifiés ; aucune fuite n’est déduite de ces seuls chiffres.

## Archive, retrait et publication

Les identités `identity-frozen.json`, `identity-frozen-v2.json` et `identity-frozen-v3.json` décrivent les sources et binaires effectivement testés. Les 83 membres de `evidence/source-snapshot.tar.gz` concordent tous avec les SHA v3 ; archive SHA-256 `963d610bbb99a1a97a5b4e3760c0c702fa1ee58f7ccab4dd24c1408c5c6aaf28`. CLI identique `19c9e1f2`, tests v1 `9412fc5b`, v2 `8165235a`, v3 `f0165b9d` ; changements de tests seuls entre ces versions.

Une première commande combinant archive, réécriture HEAD et suppression a été rejetée par l’approbation automatique avant exécution. Le retrait a ensuite réussi par la procédure limitée : archive seule, revue et contrôle des 83 SHA, puis inversion du diff F01 exact. Les deux sources de tests `linear_half_output_tests.rs` et `ssd_moe/half_output_tests.rs` sont préservées mais leurs références de modules ont été retirées. Aucune fonction F01 ni son flag ne reste dans le code compilé. `withdrawal-doc-audit.json` distingue cette constatation statique du patch prototype archivé.

Le runner stock consigne les SHA sans les imposer : les contrôles externes des vrais binaires/sources, gold60 et traces ont été inspectés séparément. La CI du commit publié `62cae` concerne 82B et la base, pas la publication d’un moteur F01 : régressions, macOS/Linux et Flash Kani réussissent ; le Kani global échoue, diagnostic non obtenu. Aucune CI globalement verte n’est revendiquée.

Ce dossier est destiné à une publication **documentaire uniquement**, avec résultats négatifs, commandes/options, bruts, échecs, contre-revues et patchs historiques. Les binaires `runtime/`, le script local `apply-source.py`, les poids et les deux fichiers de tests détachés du moteur ne sont pas à stage. Aucun remplacement de l’application. Les essais F06/F05/F03 sont indépendants ; aucun de leurs changements n’est inclus dans l’archive F01 et aucun gain n’est additionné.

Les contre-revues gardent leur état au moment de lecture : `cross-review-quality` précède le complément v3, et `cross-review-confirm` précède le retrait. Leurs mentions historiques « en attente » ou « à retirer » ne remplacent pas l’état final du présent README et de `summary.json` : **runtime F01 retiré**.
