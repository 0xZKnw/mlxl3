# Préparation des cinq écrans72S — aucun lancement

Amendement prospectif [72S/73S](/Users/justin/.codex/worktrees/flash-splash60/mix-stq1_0/opti.md:14694) relu avant création.
Cinq plans neufs : late,bounded,unused,lfu,weighted sous speed-v3/CANDIDAT/screen.
Pré20, MAIN H-C, HOLD H-C, post30 ; CLI77v1 69b919ae31b906b8fedd20ec95403c390543ad5b5f99ab93826e38be801d323c.
H=legacy/pipeline1/lookahead2 ; C change seulement PREFETCH_VARIANT.
B2 WIDTH1/ADAPTIVE0/PRECAP0,3Gio53slots,allocator64Mio,128sorties127decode,
unwarmup128+unemesure128, timeout660s par enfant, secteur et stock256Mio/phase.

Les cinq run-v2 sont intacts ; seules CLI et destinations ont été substituées.
Les identités, environnement, programmes existants et golden60 OFF128 ont été
vérifiés avant création.165sources recontrôlées : seule la CI a changé depuis
plans-v3-source-identity.json ; cette différence est enregistrée, pas effacée.
Aucun poids n’a été chargé ni rehashé intégralement ; versions de checkpoints
existantes et chemins sont conservés. Aucun code exécutable, test, benchmark,
build ou proveur lancé par cette préparation ; wrapper commun référencé sanscopie.

Les six bridges43v3 antérieurs (ces cinq modes etearly-shared) restent
préparés/nonexécutés, aucun résultat n’existe à cette préparation. Le bridge
packed43 validé ne les remplace pas pour une promotion : voir amendement72S.
Les écrans servent à écarter rapidement les variantes lentes. Tous les warmups
et mesures doivent encore passer le golden60 texte/hash/comptes nonvides ;
cette comparaison ne prétend pas vérifier tous les logits/états longs.

Les budgetscompression77B échoués restent échoués. Aucun gain promu ou15tok/s
revendiqué ; valeurs privées et globales doivent rester séparées. Pas de reverse
créé ni de lancement automatique du candidat suivant. Le parent choisit après
lecture des résultats ; un candidat prometteur exige ensuite son propre43.

Manifest : screens-v3-72S-prepared.json ; chaque écran possède identity.json
etrun.json. SHA manifest : f2e232afdf08394ab9ee2ff8c7680b89440898323baee8fa2350038c57edcaba. L’ancien document de préparation
plans-v3-preparation.md reste historique (son prérequis43avantécran est amendé
par72S/73S, pas réécrit rétrospectivement).
