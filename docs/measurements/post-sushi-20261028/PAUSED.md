# Pause utilisateur avant les contrôles GPU

**Pause levée le 28 octobre par l'utilisateur : retour au domicile et tests
GPU/mesures autorisés.** Essai clos le 29 octobre après audit, correction C et
nouveau comparatif : variantes archivées et rejetées, moteur de référence
restauré et vérifié. Résultats : [README.md](README.md). Le texte ci-dessous
conserve l'état historique de la pause.

28 octobre2026. L'utilisateur teste Qwen3.8, hors domicile et sur batterie. Instruction : ne pas continuer le test, lire la nouvelle issue ; finir Sushi après son retour sur secteur, puis passer à l'issue. Ne lancer aucun GPU, build, proveur ou installation pendant cette pause. Ne pas arrêter son inférence.

POSTPROFILE05 est un prototype local non validé GPU. Les deux soumissions Hot et les barrières restent en place. Aucun test physique/benchmark05 lancé, aucune ancre créée. L'app installée n'a pas été remplacée.

Les phases CPU sont terminées/jointes : Clippy strict, formats, builds, suites MLX279passés168ignorés etCPU249passés6ignorés ; géométrie Kani146SUCCESS+6covers, sansassume. L'API MLX réelle reste non vérifiée formellement (ICE243). Le compilateur Metal CLI est un stub sans MetalToolchain ; le contrôle offline a échoué, aucune compilation/parité GPU n'en est déduite.

Gel : [identity-v1.json](identity-v1.json). CLI6743401124a4969f5b5c75176a52ea4499f66f416ef830e929b87f102e31eecf ; libtest21788afeafc6537b91e7b4d15dbc3d82c10efda7bdb96ee164423402168b3344. Les50cas produit/scales, poison,5refus, overlap puis oracles/modèle/erreurs/Wide/paires/MTP sont dans [quality-plan.json](quality-plan.json), non exécutés.

À la reprise : vérifier le signal utilisateur, secteur, absence de son test modèle/autres builds/proveurs et ressources ; recontrôler les hashes source/binaires. Créer la première ancre05 après prévol, sans réinitialiser une campagne en échec. Réutiliser le superviseur strict de POSTPROFILE01–03. Après qualité : mutation du vrai index MSL, détection numérique puis restauration/rebuild et seuls contrôles affectés ; vraie CLI/golden avant ABBA MAIN/HOLD128/1chauffe+3mesures. Qualification MTP/DF avant conservation si gain. Si rejet, archiver puis restaurer seulement les changements05 face à before-source.json ; préserver les modifications préexistantes.

La revue de l'issue58 est séparée : [triage](issue58-triage.md). Aucune de ses pistes n'est implémentée pendant la pause. Aucun commentaire GitHub, push ou publication.
