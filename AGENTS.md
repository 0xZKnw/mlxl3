# MLXL3 — consignes pour les agents

Ces règles s'appliquent à tout le dépôt, y compris aux scripts de benchmark,
tests, harnais de preuve, packaging et workflows CI. Préserver les modifications
locales existantes ; ne pas les réinitialiser ni les inclure dans une validation
de PR sans le signaler. Les instructions explicites de l'utilisateur priment.

## Journal obligatoire des optimisations

Ces consignes s'appliquent à tout le dépôt : moteur, kernels Metal, inférence,
quantification, CLI, GUI, mémoire, cache et téléchargements.

1. **Avant toute recherche, modification ou benchmark d'optimisation, lire
   `opti.md` à la racine**, puis les rapports historiques pertinents qu'il
   référence. Chercher aussi dans le code si la piste existe déjà.
2. Ne pas refaire un essai déjà documenté sans raison nouvelle et explicite
   (nouveau kernel, forme, modèle, matériel, version ou protocole corrigé).
   Une répétition de validation reste permise : noter l'essai précédent et
   ce que cette répétition doit vérifier avant de la lancer.
3. **Avant chaque nouvel essai, ajouter une entrée dans `opti.md`**, avec un
   identifiant, l'hypothèse, le changement, la baseline et le protocole prévu.
   Pour une série paramétrée, une entrée peut référencer une matrice détaillée
   conservant chaque variante et son résultat.
   Ce préenregistrement est une condition de démarrage : ne pas lancer le
   prototype ou la mesure puis reconstruire son protocole après les résultats.
   Une revue qui répète un contrôle d'optimisation doit identifier l'essai
   historique et la propriété qu'elle veut vérifier.
4. **Après chaque essai, mettre cette entrée à jour immédiatement**, même
   pour un échec, crash, interruption, régression ou résultat non concluant.
   Indiquer les résultats, les contrôles de qualité, les limites et la décision.
   Avant de s'arrêter ou de rendre la main, consigner l'état des essais en cours.
5. Enregistrer les commandes/options, modèle et quantification, versions,
   contexte/tokens, répétitions, alimentation/conditions connues, métriques
   avant/après avec unités, et chemins des preuves. Marquer « non mesuré »
   plutôt qu'inventer une valeur. Distinguer microbenchmark et gain réel,
   mémoire allouée et RAM processus, égalité du texte et validation numérique.
6. Conserver les résultats négatifs et les anciennes conclusions. Ajouter une
   révision datée si elles changent ; ne pas les effacer. Ne pas additionner
   des gains issus de tests incompatibles ni présenter du bruit comme un gain.
7. Le journal doit distinguer **validé**, **rejeté**, **non concluant**,
   **bloqué/interrompu** et **en cours**, ainsi que l'état d'intégration réel
   (prototype, code local, app installée, publication). Un benchmark réussi
   n'implique pas que l'app a été mise à jour.
8. Identifier précisément les sources et artefacts comparés : commit, état
   local modifié, empreintes des binaires, options et variables d'environnement
   pertinentes, modèle/checkpoint et dépendances. Une mesure d'un prototype
   retiré ne valide pas le code finalement conservé. Refaire uniquement les
   contrôles affectés après une modification et conserver les preuves anciennes.
9. **Valider la correction avant de mesurer la vitesse.** Utiliser des oracles
   indépendants et vérifier les formes, le nombre de valeurs et leur finitude ;
   deux sorties vides ou un filtre qui sélectionne zéro test ne constituent pas
   une parité. Pour un changement numérique ou de cache, vérifier aussi les
   logits/états pertinents et le parcours réel bridge/CLI, avec une tolérance
   explicitée ou une égalité bit à bit selon le contrat.
10. Séparer les mesures GPU des compilations, proveurs et autres inférences.
    Ne charger qu'un modèle GPU à la fois pour une campagne. Prévoir warmup,
    répétitions et ordre alterné adaptés ; relever les conditions connues et
    les contrôles inchangés. Une forte dérive, un changement de protocole ou
    une parité non vérifiée interdit de présenter un gain comme établi.

Les rapports détaillés peuvent rester dans `docs/` et les mesures brutes dans
leurs fichiers habituels, mais `opti.md` doit toujours contenir un résumé et
leurs liens. Si les logs temporaires ont disparu, le signaler : leur absence
n'autorise pas à refaire silencieusement un essai. Ne pas lancer d'optimisation
uniquement pour remplir le journal, ni publier sans demande de l'utilisateur.

## Implémentation et tests obligatoires

- Implémenter le comportement demandé de bout en bout. Après une première
  version fonctionnelle, entrer dans une phase de vérification : chercher les
  contre-exemples, corriger les échecs causés par le changement et relancer les
  contrôles affectés. Une demande de revue autorise l'analyse et les contrôles ;
  elle n'oblige pas à modifier le code de la PR pour masquer les constats.
- **Pour chaque changement de code exécutable, charger `$formal-proof-skill`
  et la référence du langage concerné.** Tracer les fonctions modifiées,
  leurs appelants, préconditions et effets visibles. Réutiliser les outils du
  dépôt ; ne pas ajouter une grosse dépendance de preuve sans besoin concret.
- Ajouter des tests de régression déterministes pour le comportement attendu,
  les frontières et les erreurs. Exercer l'entrée de production ; ajouter les
  tests d'intégration aux frontières touchées et les tests de bout en bout des
  parcours utilisateur affectés. Utiliser des données jetables et isoler les
  fichiers, registres, téléchargements et processus de test.
- Pour un correctif, reproduire si possible l'échec sur la version précédente.
  Confirmer qu'un oracle ou test non trivial détecte le bug initial ou une
  mutation temporaire pertinente, puis restaurer le code et relancer. Ajouter
  du property-based testing ou du fuzzing déterministe lorsque cela expose des
  cas supplémentaires ; ne pas remplacer un oracle par une copie du calcul testé.
- Les scripts de vérification et benchmark doivent aussi couvrir leurs erreurs :
  réponses invalides/incomplètes, divergence, processus interrompu ou silencieux.
  Borner les attentes et nettoyer/rejoindre les processus, y compris sur erreur.
  Sauvegarder les échecs avec un état explicite ; ne jamais laisser un rapport
  partiel annoncer une validation complète ou `parity: true` après divergence.

## Vérification formelle et Kani

- **Pour les changements Rust, tenter Kani sur l'implémentation réelle.**
  Ajouter ou adapter un harnais `#[kani::proof]` pour les propriétés touchées :
  bornes, arithmétique, formes, budgets, offsets, acceptation et transitions
  d'état selon le changement. Une compilation ou des tests seuls ne remplacent
  pas cette tentative.
- Utiliser des entrées symboliques ; limiter `kani::assume` aux préconditions
  réellement garanties par les appelants. Vérifier la portée des hypothèses et
  utiliser `kani::cover!` pour les branches/cas importants. Garder les assertions
  d'unwinding ; documenter les domaines et bornes de boucles/récursion.
- Exécuter d'abord le harnais ciblé, puis la suite pertinente du package lorsque
  praticable. Le dépôt utilise notamment `cargo kani --lib --no-default-features`
  pour les contrats CPU. Inspecter les obligations individuelles, couvertures,
  assertions inatteignables, fonctionnalités non prises en charge et timeouts.
  Ne pas réduire le domaine ou ajouter un stub injustifié pour obtenir du vert.
- Pour C/C++, Python et les autres langages, tenter le vérificateur source le
  plus fort praticable décrit par le skill (par exemple CBMC ou CrossHair),
  avec le vrai contexte de compilation et des abstractions explicites.
  Vérifier le code original dans son langage, sans le réécrire ailleurs pour
  prétendre l'avoir prouvé.
- Si le vérificateur est absent, incompatible ou dépasse la limite prévue,
  conserver le harnais utile, la commande et le diagnostic ; marquer la propriété
  **non vérifiée**. Un timeout n'est ni un succès ni un contre-exemple. Distinguer
  preuve déductive, model checking borné, recherche symbolique, exploration
  exhaustive finie et tests échantillonnés. Une preuve CPU ne prouve pas MLX,
  Metal, l'allocateur, la concurrence ou la génération complète.

## Contrôles locaux et CI

- Exécuter les contrôles applicables de format, lint strict, types, compilation,
  tests ciblés et suite pertinente complète. Vérifier la variante réellement
  livrée ; pour le moteur, un build sans `mlx,chat` ne valide pas la FFI MLX.
  Consulter `Cargo.toml`, les scripts et workflows actuels avant d'utiliser une
  ancienne commande de documentation. Un test ignoré, sauté ou absent reste
  non exécuté et doit être indiqué.
- Les tests nécessitant un GPU Apple physique doivent être sélectionnés
  explicitement et exécutés seuls lorsque leur état global l'exige. En l'absence
  du matériel/modèle requis, préciser les parcours non vérifiés ; la CI hébergée
  ne remplace pas les contrôles numériques physiques.
- Dans un dépôt GitHub, inspecter `.github/workflows/`. Pour une implémentation,
  créer ou réparer la CI nécessaire aux contrôles touchés : PR et pushes sur la
  branche par défaut, setup reproductible, dépendances/verifier identifiés et
  aucune credential de production. Pour une revue, signaler les lacunes observées.
- Quand l'accès est disponible, vérifier les runs GitHub du commit exact et
  leurs jobs. Distinguer succès, échec, annulation, contrôle absent et approbation
  requise d'une PR externe. Ne pas déclarer une CI verte à partir de l'existence
  du workflow ou des seuls tests locaux.

## Clôture et revues de PR

- Examiner le diff final et tous les fichiers affectés. Pour une PR, fixer les
  SHA base/tête, vérifier que le checkout testé correspond et distinguer les
  régressions introduites des problèmes antérieurs. Fonder les constats sur un
  scénario concret, leur impact et les lignes concernées.
- **Avant de rendre la main, actualiser `opti.md` pour chaque essai lancé** et
  fermer ou qualifier explicitement les essais encore en cours. Conserver les
  commandes/options, résultats bruts et liens vers le rapport ; signaler toute
  preuve temporaire manquante. Ne pas effacer un échec après une relance réussie.
- Rapporter ce qui a été testé, les nombres de tests passés/ignorés, les
  propriétés vérifiées et leurs bornes, les échecs/timeouts et les parcours
  restant non vérifiés. Ne jamais présenter des tests ou une vérification bornée
  comme une preuve complète. Une limitation externe doit rester visible et ne
  peut pas être transformée en statut « validé ».
- Aucun push, merge, publication, release ou remplacement de l'app installée
  sans demande ou autorisation correspondante de l'utilisateur.
