# Pool partagé d’experts — état du 29 octobre 2026

Le nombre de lectures supplémentaires ne permet pas de conclure sur les tok/s. Le pool a donc été branché dans le vrai moteur Rust, qualifié numériquement, mesuré sur deux prompts, puis audité avant décision : **cette adaptation ARC est rejetée**, sans généraliser à toutes les variantes de pool.

## Comparaison figée

| Bras | Banques | Politique | Binaire |
|---|---|---|---|
| A | 48 jeux séparés | LFU53 par couche | a60eb4ad… |
| U | 7 banques communes, 2544 slots physiques | 48 LFU53 inchangés | 934164bd… |
| C | mêmes banques communes | ARC global | 934164bd… |

Les poids occupent 3 174 942 528 octets dans chaque bras, sous le budget de 3 Gio. Les ghosts ARC ne contiennent aucun poids. Le prototype a été retiré du runtime après audit ; le binaire restauré est byte-identique à la référence A. Aucun changement de l’application installée.

Les sources exactes, dépendances et empreintes sont dans [pool-identity-v1.json](pool-identity-v1.json) ; l’extension figée pour les mesures bridge est [pool-identity-v1-speed.json](pool-identity-v1-speed.json). Le routeur SIMD préparé auparavant est isolé et ne participe pas à ces binaires.

## Résultats inspectés

- Replay : 60 912 appels et 609 120 paires physiques `(slot, hit)` de la baseline reconstruits exactement. ARC ajoute 5,78–6,95 % de chargements sur MAIN et 2,30–2,65 % sur HOLD. Ces chiffres ne sont pas des mesures de vitesse : [arc-current-report.json](arc-current-report.json).
- Suites CPU : 254 passés, 8 ignorés ; variante MLX : 290 passés, 172 ignorés. Les tests physiques ignorés nécessitent une sélection explicite : [runtime-cpu-plan-result.json](runtime-cpu-plan-result.json).
- Microqualité GPU réelle : 2 tests sélectionnés, chacun 1 passé / 0 échec / 0 ignoré. Le premier couvre 16 projections EXL3 et 48 masques FP16/FP32 aux slots 512 et 2543 ; le second couvre les deux politiques, les consommateurs en attente, le partage entre couches et la récupération après échec I/O. Tous processus joints, zéro nouvelle écriture swap : [pool-micro-quality-plan-result.json](pool-micro-quality-plan-result.json).
- Qualité modèle : **10/10 tests physiques passent**, sans sélection ignorée, pour U et C. Logits complets et états GDN bit à bit face à l’oracle indépendant sur cinq positions ; paires/MTP et 36 préfixes DFlash par politique comparés à la série avec logits/états exacts. Les composants et captures ont leurs tolérances explicites. Tous processus joints, zéro nouveau Swapout : [pool-model-quality-inspection.json](pool-model-quality-inspection.json).
- Bridge réel : **6/6 processus passent**, A/U/C sur MAIN/HOLD, une chauffe puis une génération de 17 tokens, hash et préfixe texte contre l’oracle externe : [pool-bridge17-inspection.json](pool-bridge17-inspection.json). Ces petits chronos ne décident pas du gain.
- Driver de mesure : 78 tests driver/runner passés, puis mutant du contrôle de hash détecté et 12 tests driver repassés après restauration. Vérification symbolique Python indisponible : CrossHair absent.
- Kani : géométrie des destinations et remapping uniforme réussis, 7 couvertures atteintes. Le vrai premier fill global a expiré à 90 s ; la vraie entrée MLX rencontre l’ICE `intrinsics.rs:243`. Ces chemins restent non vérifiés formellement ; aucune preuve Metal ou de concurrence n’est revendiquée.

## Débit réellement mesuré

La seconde série est complète : **36 générations mesurées de 128 tokens, plus 12 chauffes**, ordre A-U-C-C-U-A par prompt, 127 tokens de decode par génération. Tous les hashes/textes correspondent à l’oracle externe ; tous les processus sont joints, zéro nouvelle écriture swap sous l’ancre w2.

| Prompt | A actuel | U partagé LFU | C partagé ARC | ARC / actuel |
|---|---:|---:|---:|---:|
| MAIN | 10,703 tok/s | 10,766 tok/s | 10,330 tok/s | −3,48 % |
| HOLD | 10,585 tok/s | 10,664 tok/s | 10,358 tok/s | −2,15 % |

Débits agrégés = somme des tokens de decode / somme des secondes. Le préfill ARC varie de −1,58 % et −2,27 % face à A. U reste non concluant : +0,58/+0,75 % en decode, +0,34/−0,94 % en préfill, avec des plages de référence qui ne permettent pas d’établir un petit gain. Aucun échantillon exclu. Données, plages et empreintes : [pool-speed128-w2-inspection.json](pool-speed128-w2-inspection.json).

Le plan [pool-speed128-plan.json](pool-speed128-plan.json), A-U-C-C-U-A par prompt, prévoyait 36 générations mesurées de 128 tokens plus 12 chauffes. **Il s’est arrêté sur A avant la première mesure**, après 128 Kio de nouveaux Swapouts système. Une seule chauffe de référence terminée : elle ne constitue pas une mesure de comparaison. U/C n’ont pas été chronométrés dans cette série. Rapport incomplet et arrêt conservés dans [pool-speed128-plan-result.json](pool-speed128-plan-result.json), tous processus arrêtés et joints.

Le premier chargement modèle avait été refusé **avant création du processus** : marge estimée de 7,815 Go, seuil préenregistré de 8,5 Go. Cette interruption reste dans [admission-pool-model-quality-plan-result.json](admission-pool-model-quality-plan-result.json). La reprise avec marge de 9 Go a ensuite permis les 10 tests modèle et 6 bridges ci-dessus, sous la même ancre 04-w1 sans swap. Cette ancre est désormais close après l’arrêt du débit128.

La série distincte [pool-speed128-w2-plan.json](pool-speed128-w2-plan.json) a démarré après libération de RAM observée et création d’une nouvelle ancre explicitement documentée. Ses 12 processus passent : [pool-speed128-w2-plan-result.json](pool-speed128-w2-plan-result.json). Gardes inchangées : secteur, zéro nouveau Swapout, Swapin cumulé ≤256 Mio, RSS ≤10 Gio, empreinte native <12 Go, aucun autre modèle/compilateur/proveur concurrent. Aucun résultat ancien n’est effacé ni réutilisé comme mesure propre de cette série.

## Audit avant retrait

Cinq diagnostics physiques supplémentaires passent, avec oracles externes et contrôles complets des traces. La barrière globale coûte environ 0,02 ms/token ; elle n’explique pas le recul. La réserve/import CPU passe de 0,110 ms/token en LFU à 0,663 en ARC dans le diagnostic light. Celui-ci observe 9 207 lectures logiques contre 10 115 en ARC, et des remplissages plus longs. Ces durées sont des scopes host inclusifs ; elles ne prouvent ni le temps GPU idle exclusif, ni les lectures NVMe physiques, ni une décomposition causale additive du recul release. Preuves : [pool-audit-light-inspection.json](pool-audit-light-inspection.json), [pool-audit-detail-inspection.json](pool-audit-detail-inspection.json).

Les formes, dtypes, fallbacks, copies, allocations, consommateurs et chemins réels ont été contrôlés ; aucun défaut concret de branchement ou de calcul n’a été trouvé. Les recherches linéaires ARC ajoutent du CPU ; une version indexée O(1) reste **non mesurée**. Le rejet concerne le prototype testé. Les banques partagées seules n’ont pas établi de gain justifiant leur conservation dans le runtime.

Le préfill AR appelle l’API par chunks de huit pour omettre les heads inutiles, mais `forward_tokens` boucle sur le MoE scalaire : les traces montrent 87×48 appels d’une ligne. Le pool ne transforme pas ce parcours en préfill vectorisé. Le harnais light calcule aussi les heads de chaque token ; ses tok/s ne servent donc pas à remplacer les mesures release. Le vrai batching préfill P01B a déjà été testé puis retiré pour régression decode ; il n’est pas relancé ici.

Les sources exactes mesurées (91 fichiers) sont conservées dans [retired-runtime-v1-sources.tar.gz](retired-runtime-v1-sources.tar.gz), empreinte et décision dans [runtime-retirement.json](runtime-retirement.json). Les replays CPU/ARC et gardes de validation restent disponibles. Les performances MTP/DFlash n’ont pas été mesurées, faute de gagnant AR ; leurs contrôles numériques sont passés. Aucun push ou remplacement d’application.

## État final vérifié

Le CLI final retrouve exactement le SHA-256 `a60eb4ada3a948110fc057d9551032b6d03598ab404201145f5a5410f26f8e75`. Les cinq sources runtime restaurées correspondent aux sauvegardes précédant le pool ; les modifications locales antérieures sont préservées. Le routeur SIMD reste isolé, non qualifié.

Contrôles finaux : CPU **252 passés / 8 ignorés**, variante MLX **288 passés / 170 ignorés**, Python **344 passés / 0 ignoré**, format, Clippy strict CPU/MLX et Ruff réussis. Kani sur l’API réelle de destinations des slots finaux 0..511 : 228 obligations, zéro échec, une inatteignable, deux couvertures atteintes. Cela ne prouve pas Metal ou la concurrence. Les anciens timeouts/ICE restent conservés ; la CI distante de cet arbre local n’a pas été exécutée. Le CLI étant byte-identique à la référence déjà mesurée physiquement, aucun test modèle GPU supplémentaire n’a été relancé après restauration.

Preuves : [retirement-final-state.json](retirement-final-state.json), [retirement-cpu-plan-result.json](retirement-cpu-plan-result.json), [retirement-python-inspection.json](retirement-python-inspection.json). Aucun essai POOL04 actif.

Journal de référence : [opti.md](../../../opti.md). Aucun push, publication ou remplacement d’application.
