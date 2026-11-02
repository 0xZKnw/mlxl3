# Publication structurée des travaux — 2 novembre 2026

Publication autorisée par « push bien dans plusieurs commits, separe bien le
travail, tout ce qu'on a fait hier ». Périmètre : travaux moteur/profilage de
ce chat et nouveaux candidats issus des issues66/67 ; aucun fichier Desktop
étranger à ce worktree n'est intégré.

Les commits distinguent profilage/préfill, Dynamic+ColdU activé, prototypes
prepared/ColdShared/DEC-ONE désactivés, rapports et recherches, nouveaux
candidats SSD, géométrie MB3/MB4, puis analyse des diagnostics. Les anciens
résultats négatifs, interruptions et limites restent conservés. Le journal
`opti.md` est consolidé avec les rapports. Les étapes intermédiaires utilisent
les snapshots historiques identifiés, sans modifier les fichiers du worktree ;
leur compilation n'a pas été rejouée. L'état final des sources a passé
compilationMLX/chat tous targets et Clippy strict le2novembre ; les nouveaux
tests/proofs/GPU sont différés à la demande de l'utilisateur.

`published-inventory.json` décrit3794artefacts légers sélectionnés (environ
159,56Mio) avant l'ajout de cet inventaire et des métadonnées finales. Les
rapports, résumés, scripts, identités, snapshots de source et logs sont inclus.
Les nouveaux fichiers de publication et de clôture sont ajoutés séparément.

`local-only-inventory.json.gz` recense11312fichiers conservés localement,
environ18,6Gio : captures `.gputrace`/`.trace`, exportsXML bruts, longues traces
JSON, fichiers dépassant2Mio ou formats binaires. Leurs liens dans les anciens
rapports peuvent donc viser des preuves locales **non publiées** ; aucune
suppression ni recompression des originaux. Les empreintes historiques déjà
présentes dans les rapports/identités ne sont pas remplacées. Cette sélection
de publication ne change aucun résultat. Aucun motif de credential des formats
contrôlés trouvé dans les nouveaux artefacts texte sélectionnés ; ce contrôle
ne constitue pas une garantie exhaustive.

Les expérimentations sont publiées sur une branche de travail. Pas de merge,
release, remplacement de l'application ou activation des candidats non
qualifiés. Aucun statut CI vert déduit de la présence des workflows.
