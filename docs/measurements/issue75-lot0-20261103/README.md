# Lot 0 de l’issue 75 — récupération d’AR-QUICK, 3 novembre 2026

La campagne locale `AR-QUICK-2026-11-02-01` du 2 novembre est close. Les cinq bras prévus pour les comparaisons individuelles sont terminés. Le routeur SIMD reste désactivé. Le regroupement des callbacks MLX reste expérimental. Le plan conditionnel S+K n’a produit aucune mesure. Ce dossier restitue le pilote historique et ses sources. Le lot 0 reste ouvert pour la référence de la migration. [Issue 75](https://github.com/0xZKnw/mlxl3/issues/75) · [Photographie précédente dans l’issue 73](https://github.com/0xZKnw/mlxl3/issues/73)

## Débits retrouvés

M5 standard, 24 Go, sur secteur. Flash-Next EXL3 2,05 bpw, checkpoint `65c89531`, MLX 0.32.2, Rust 1.98.1. Une session AR B1, contexte configuré à 4 096, prompts de 87 et 82 tokens. Cache experts de 3 Gio, allocateur de 64 Mio, quatre lecteurs. DynamicReaders, ColdReady, Hot, Wide et Unique activés. Les deux limites MLX par buffer sont à 40. Chaque processus a fait une chauffe et une mesure par prompt, dans l’ordre des bras du tableau. Les débits portent sur les 127 transitions après le premier des 128 tokens générés.

| Bras, dans l’ordre d’exécution | MAIN (tok/s) | HOLD (tok/s) |
|---|---:|---:|
| A1 · wheel, témoin initial | 10,8945 | 11,0236 |
| S · wheel, routeur SIMD activé | 11,0944 | 10,3531 |
| K0 · MLX recompilé, regroupement désactivé | 10,0640 | 10,6187 |
| K · même MLX, regroupement activé | 10,9530 | 11,1033 |
| A2 · wheel, témoin final | 11,1989 | 11,0211 |

- **SIMD01 (S).** Face à la moyenne A1/A2, +0,43 % sur MAIN et −6,07 % sur HOLD. Les témoins eux-mêmes varient de +2,79 % et −0,02 %. Le pilote écarte l’activation par défaut.
- **Keepalive (K).** Face à K0, +8,83 % et +4,56 %. Face à la moyenne du wheel, −0,85 % et +0,73 %. K0 n’a été mesuré qu’une fois par prompt, avant K. Le gain de débit sur le moteur distribué reste non concluant.
- **Contrôle hors ligne.** Recalcul des dix débits depuis les durées de décodage. Les 20 réponses, chauffes comprises, ont le texte non vide et l’empreinte des IDs du contrôle historique indépendant du pilote, `post-p09-dynamic-cold-20261101/short-AR-2-C`. Empreintes MAIN `f6b5f974c6db2e7a` et HOLD `9d02c1a97e015196`. Les cinq superviseurs se sont terminés normalement et les cinq lecteurs ont été rejoints. Swapout système cumulé sur ces chronos, 720 896 octets. [Rapports par requête](reports/) · [Empreintes et calculs](evidence.json)

## Diagnostic SSD et suite distincte

Le recomptage de la trace MAIN32 donne 31 pas de décodage, 858 lots Cold, 852 répartis en W1/W2 et 590 encore incomplets juste après la soumission de W1. Par pas, l’attente hôte initiale vaut 14,975 ms, la jointure finale 5,582 ms et la somme des écarts entre les deux dernières fins de lecture 2,784 ms. La trace ne contient pas la corrélation temporelle GPU qui permettrait de chiffrer la part récupérable.

La découpe SSD a été mesurée **ensuite**, dans `ar-ssd-split-20261102`. Les rapports par requête confirment les débits publiés pour les deux versions. R1 (tous les records admissibles) donne −5,57 % / −1,57 % face à ses témoins. R2 (dernier record ciblé) donne −5,26 % / +0,05 % face à son témoin de même compilation. L’admission mémoire de ces essais est passée à 8 Go. Conserver la lecture entière par défaut. [Rapport SSD d’origine](historical/ar-ssd-split-20261102.md.txt)

## Provenance récupérée

| Élément | Identité |
|---|---|
| Base Git du pilote | `b9d5cbdce50aa98adfee560e06da75614e730bd3` + modifications locales |
| CLI mesuré dans A1/S/K0/K/A2 | `181346d9fe43fe85e4a08643dd19523258dca2d36cfcb82f3629ae51dd634eec` |
| CLI antérieur mentionné par #73 | `04a8a5cd0b51522fe47b5fc713826d9de10ba39ad1de921f4626b523eacc1863` |
| MLX recompilé K0/K | révision `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, dylib `1784179fa98de2ab84ca73f88c2e0efac792dc3a362516996138c6b2353f27cc` |
| Moteur installé de la comparaison séparée | 2.0.2-beta.1, source `9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8`, CLI `43d02fbc9867b8152e74f549304f164a30a08643faa4eb30ff378372d4824839` |

Les six fichiers binaires du manifeste initial, le libtest après correction de la fixture stridée, le CLI antérieur et les quatre fichiers du moteur installé correspondent aux empreintes archivées. Les mesures de ce dernier étaient de 11,1841 / 10,9334 tok/s dans sa comparaison séparée. [Manifeste original du pilote](ar-quick-identity.json) · [Comparaison du moteur installé](historical/ar-engine-regression-20261102.md.txt)

Le worktree d’essai a évolué. Treize des 217 entrées de la liste historique des sources différaient à leur emplacement d’origine. Le patch MLXL3 joint reconstitue, relativement à la base Git, les **16 différences** parmi les 183 entrées du manifeste sous `native/`, `scripts/` et les trois fichiers Cargo/build à la racine. Pour les fichiers divergents, les octets proviennent de la base ou de copies historiques identifiées par SHA-256. Le routeur correspond à la version mesurée avant la correction de la fixture de test. Les quatre entrées du patch MLX ont également été retrouvées. Contrôle effectué avec `git read-tree`, `git apply --cached --check`, application dans des index temporaires, puis vérification des empreintes des 183 + 4 entrées. La reproduction binaire par recompilation reste à faire si elle devient nécessaire. [Origine de chaque différence](source-recovery.json) · [Patch MLXL3 du pilote](mlxl3-measured.patch) · [Patch MLX original](mlx-0.32.2.patch)

## Portée numérique et limites

Les résultats archivés de S, K0 et K couvrent chacun cinq pas de l’oracle indépendant mlx-vlm 0.7.3 + PonyExl3 `8e7fa6b`. Le test historique exige l’égalité des octets des 248 320 logits et de deux états GDN couplés par pas, ainsi que les mêmes dix experts après tri pour chacune des 48 couches. Le test vérifie aussi le reset avec un seuil RMSE relatif de 0,01 et le rejet d’entrées invalides. Les 15 fichiers de logits/états sont présents aux tailles attendues et leurs SHA-256 actuels sont consignés. Les trois bridges MAIN17 ont produit six réponses exactes. La comparaison de tous les états internes entre backends pour K reste ouverte. [Rapport de clôture d’origine](historical/ar-quick-20261102.md.txt)

Les archives comptent six sélections de composants utiles et quatre sélections modèle réussies. Le test S stridé corrigé couvre 48 combinaisons et 96 comparaisons. La sélection K de vidage du cache s’est terminée sans marqueur d’activation et a échoué au garde du superviseur. Kani CPU a validé 19 obligations et 3 couvertures du domaine et des indices du shader. Les deux tentatives Kani avec MLX ont rencontré une erreur interne du compilateur avant les obligations. CBMC s’est arrêté à l’analyse de libc++ et 44 tests de CI Linux ont été ignorés sur macOS. Ces résultats proviennent de l’essai initial. La récupération du 3 novembre a uniquement relu les artefacts.

## Passage au prochain contrôle du lot 0

Le worktree documentaire part de `main` à `b9d5cbdc`. Au relevé du 3 novembre, la PR 72 est ouverte à `6d828f3110da522418ee612151cf2891fccffca8`, une tête plus récente que celle de l’issue. Son diff paginé compte 243 fichiers, aucun sous `native/`. La PR 74 est ouverte, en conflit, à `ede7f3b8d8cbdc995726d48ffff50b276338c039`. Sur 315 fichiers, 12 touchent `native/`, dont le bridge, les arrays et MTP. Aucune des deux têtes n’entre dans la référence historique. [PR 72](https://github.com/0xZKnw/mlxl3/pull/72) · [PR 74](https://github.com/0xZKnw/mlxl3/pull/74)

Prochaine étape du lot 0. Figer la source et le binaire du contrôle de migration, l’identité des poids, la version exacte de macOS, le budget mémoire total, les gardes et le seuil d’acceptation avant mesure. Capturer ensuite AR B1 MAIN/HOLD et les fixtures de logits, routes et états sur cette référence. Les cinq pas de l’oracle retrouvé utilisent une séquence de tokens distincte de MAIN/HOLD. Le pilote ne quantifie toujours pas le temps GPU exposé qu’une tranche native pourrait supprimer.

**Clôture du 3 novembre.** Le contrôle de migration est terminé. La référence recompilée depuis main, les quatre fixtures MAIN/HOLD et le contrôle AR128 ont servi aux lots suivants. [Dossier de migration](../issue75-migration-20261103/README.md).
