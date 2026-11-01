# Optimisations après profilage — 27–28 octobre 2026

Pas de nouveau gain intégré. Flash-Next EXL3 2,05bpw, M5 standard24Gio, cache3Gio/53slots par couche, MLX0.32.2, moteur Rust/MLX. Les expériences sont préenregistrées dans `opti.md` et utilisent les mêmes binaires figés avant modification. Aucun modèle tiers exécuté.

| Essai | Résultat | Décision |
| --- | --- | --- |
| 01, opérations par buffer20/40/80 | Qualité7tests physiques passée. Pilote4processus/8mesures MAIN128 ; référence11,124→10,584tok/s, dérive−4,86%. C20moy10,980 ; C80moy10,469 | Non concluant, aucun gain établi ; défaut40 conservé |
| 02, seuil de ressources20/40/80, opérations40 | Qualité6tests physiques passée. Neuf processus alternés,2passages par candidat,1chauffe+1mesure MAIN128. DécodeMB20−1,18/−0,16% contre contrôles encadrants ; MB80−3,88/+0,31%, TTFTMB80+2,42% aux deux passages | Rejeté, aucun réglage intégré |
| 03, une seule soumission Hot en M1 | Régression de comptage/valeurs passe3cas, mais test d'overlap échoue : les routes attendent le Hot retardé. Pas de benchmark de vitesse | Rejeté ; deux fichiers restaurés byte à byte et moteur reconstruit identique |
| 04, recherche Sushi/Strata/TensorFold | [Rapport et classement](upstream/README.md),20sources épinglées/vérifiées ; aucune exécution amont | Recherche terminée, pistes non mesurées |

[Inspection01](pilot-inspection.json), [inspection02](resource-pilot-inspection.json), plans et logs dans ce dossier. Tous les processus lancés ont été joints ; gardes zéro nouvelle écriture swap, secteur et absence d'autres inférences/compilations pendant GPU. Les timings CLI sont collectés sans Instruments ; les12ms du diagnostic instrumenté ne deviennent pas une promesse de gain.

La qualité numérique01/02 vérifie cinq positions et248320logits complets par position contre la référence indépendante, états GDN capturés bitexact, composants et erreurs/layouts. Les réponses CLI128 vérifient texte/hash/counts au golden externe ; ce contrôle seul ne prouve pas128états numériques complets.

Pour03, fmt/lintstrict/buildrelease réussissent ; suiteMLX/chat278passés168ignorés etCPU/chat248passés6ignorés avant GPU. Les tests ignorés ne sont pas comptés exécutés. Le nouveau test échoue sur l'ancienne double soumission avec le message attendu ; son superviseur avait un champ `native_test` mal recopié, échec conservé. Kani : trois filtres sans entrée, puis vrai harnais appelant la régression et SsdMoe ; compilation échoue sur ICEintrinsics.rs243, aucune propriété GPU prouvée. Deux tests physiques du prototype exécutés :1passe/1échoue ; dix autres sélections et tous les débits03 ne sont pas exécutés après rejet.

Les fichiers `single-hot/*.rejected.txt` et les binaires rejetés sous `build/post-profile/` gardent la reproduction. Les sources retenues correspondent à `ultra-profile-20261027/v1-source.json` ; le CLI reconstruit a le même SHA-256 `a60eb4ada3a948110fc057d9551032b6d03598ab404201145f5a5410f26f8e75` que la baseline. CI inspectée, aucun run distant pour cet arbre local non poussé. Aucun changement à l'app installée ou à une release.
