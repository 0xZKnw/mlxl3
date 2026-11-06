# AR41 / issue 150 P02 — GDN et routeur dans une commande

Sur Apple M5/24 Gio, Qwen3.8-Flash-Next EXL3 moyen2.05/H4, l’écran 128 tokens donne **9.51339 OFF / 10.11220 ON / 9.49465 OFF tok/s** : environ **+6.40 % face à la moyenne des témoins**, dérive témoins−0.20 %. Une requête warmup puis une mesurée par processus ; 128 générés/127 décodages, MAIN87, cache3GiB/53slots, quatre lecteurs, profondeur0, contexte4096, prefix0, T0/topk1/pénalité1. T05/AR2/13défauts conservés ; MTP/DFlashOFF. Un modèle, aucun compiler/prover concurrent. Température/fréquence non mesurées, aucune application fermée ni garde swap ajoutée.

P02 conserve les kernels et leur ordre : HC/GDN et P partagent un CB, H reste séparé et soumis avant la lecture CPU des routes. L’ancien producteur H/F est terminé/libéré avant le nouveau H. Le CB commun est vérifié avant publication de l’état GDN ; erreurs/Drop rejoignent les producteurs. Exports/captures gardent la voie synchrone par scope RAII. Le flag `MLXL3_NATIVE_GDN_ROUTE_CHAIN` reste **opt-in**, en attente de la composition avec P03/P08 ; aucune app installée ou release modifiée.

## Correction avant vitesse

[QA physique](quality.json) : un test explicitement sélectionné, 1 passé/0 ignoré/442 filtrés. Deux tokens/36 prises chacun et **36 commandes en moins par token**. Logits complets248320Half et hidden40960B exacts ;84tenseurs de continuation par cas, **57606144 mots finis**, mêmes noms/formes/bytes. Oracle initial indépendant : digest logits complet et premier tenseur archivé, pas84archives indépendantes. Préfixe retenu, erreurs P et MoE précédent, rollback puis retry exacts ; export sans prise P02 et restauration du flag vérifiés.

Les [échecs R1](r1/quality.json) (2/36 prises : garde Active) et [R2](r2/quality.json) (garde de contexte export manquante) restent conservés, `parity:false`, tous joints. Aucun débit mesuré sur ces versions. Les plans en identifient les empreintes ; le binaire R2 a été remplacé au gel correctif, il n’est plus disponible sous son ancien chemin. Le préflight d’import échoué avant GPU est conservé séparément.

## Vérifications ciblées

- Clippy strict all-targets/direct-chat et rustfmt passent ; libtest release et CLI release construits.
- 67 tests `flash_support` passent,0ignoré ;11tests de métadonnées/inventaire passent,2subtests. Première erreur de typage de test et premier échec du manifeste conservés.
- Kani0.68 : **1harness réel,247SUCCESS,4coversSATISFIED** ;6UNREACHABLE internesstd(fmt et opérations de pointeurs de taille nulle) inspectés. Domaine : cinq booléens de la vraie admission, sans hypothèses. Cela ne prouve ni les erreurs Metal, ni les durées de vie GPU, ni la concurrence.
- L’inventaire natif inclut les trois deltas AR41 et le delta LFM3 déjà présent sur main71530f09 ; aucun code LFM3 modifié.
- Suites générales et autres tests GPU non relancés, à la demande d’aller vite ; CI exacte de publication à lire dans la PR. Profil intrusif AR39 non relancé.

## Preuves

[Gel source/binaires/modèle](plan.json), [résumé](summary.json), [OFF avant](control-before/report.json), [ON](gdn-route/report.json), [OFF après](control-after/report.json), [QA brute](quality.log), [superviseur](quality.py.txt), [pilote des trois bras](speed.py.txt). Les trois bras passent le golden complet128 et rejoignent tous leurs processus. Le pilote réutilise le contrôleur AR39 qualifié, en ne changeant que l’environnement du nouveau flag ; les empreintes des deux helpers sont gelées. Source de base71530f09 plus modifications enregistrées dans le plan ; binaires gelés SHA et révision embarquée dans chaque ready. Pas de nouveau hachage intégral des poids, cinq métadonnées vérifiées.
