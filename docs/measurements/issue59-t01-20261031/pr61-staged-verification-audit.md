# PR61 — vérifications transférées et ajout documentaire restant

31 octobre 2026. Lecture seule de [la description courante archivée](pr61-current-body.md),
tête PR61 `a36ba5077bf386466660d6b478058fbe6880245c` inchangée. Aucun merge,
nouveau contrôle exécuté ou portage T04/T18 réalisé par cette revue.

## Quelle vérification de PR61 existe déjà localement ?

| Périmètre PR61 | État local et limite |
| --- | --- |
| Deux nouveaux oracles T17 | Les tests locaux ont été adaptés au **vrai normed F32**, avec smoke court, vraie API, 63 488 gates finies/ancres F32, états/tapes, modèle indépendant, MTP/DF et capture. L'oracle PR initial normed F16 ne qualifiait pas ce domaine ; le défaut `exp` local a été reproduit puis corrigé sans élargir les seuils. [Preuves T17](../issue59-t17-20261029/verification.md). |
| Deux nouveaux oracles T04 | Réutilisés sélectivement depuis PR61 puis renforcés : smoke8 cas, F16/**vrais F32**, lignes1/2/4/8/10/20/80, frontières de cast, H128 stock, down réelle, refus avant FFI, slots masqués/groupés et mutation d'arrondi. Passés sur le candidat puis **archivés après rejet de son intégration** ; ils ne sont pas remis dans le moteur courant. [Tests exacts archivés](../issue59-t04-20261031/candidate-source/native/src/ssd_moe/swiglu_down_tests.rs), [bilan](../issue59-t04-20261031/verification.md). |
| Oracle commun logits/états/rollback et oracles existants | Les chemins réels ont été qualifiés séparément pour chaque candidat, avec référence indépendante, paires/commits/continuations et MTP/DFlash. Ne pas remplacer cette qualification par le seul nouveau test combinant les flags T17/T04 de PR61 ; nos flags ont été isolés. Les dix qualités T04, six processus bridge et douze pilote restent attachés aux binaires **archivés**81bc/220. |
| Tests portables de file dynamique T18a et validation des traces | PR61 rapporte 4 084 cas par run ASan/UBSan/TSan et mutation missing-job en cloud, ainsi que validation Python statique/dynamique. **T18a n'est pas porté dans le moteur local** ; ces succès cloud ne qualifient pas GCD/preadv→buffers Metal/leases sur M5. Le lecteur local et ses tests existants restent ceux du chemin statique. Portage et oracles réels devront être préenregistrés ensemble si T18 est choisi. |
| T18b et T01 | PR61 les diffère explicitement. Aucun oracle de lancement préparé T01 ni de publication Cold progressive T18b n'est fourni ou validé par la PR. Le nouveau filtre T01 garde sa propre qualification et ne calcule aucune sortie GPU. |

Le code conservé après retrait T04 a ses propres résultats :
[CPU256/0/8, MLX293/0/185, bins202f/a751 pré-T04](../issue59-t04-20261031/retained-checks-inspection.json).
Ils ne réactivent ni ne requalifient le candidat T04 retiré. La PR précise
elle-même qu'aucune de ses qualités Apple/Metal/modèle n'avait été exécutée
en cloud ; son check croisé via manifeste temporaire n'était pas un build
FFI/Metal livré.

## Ajout visé : la procédure staged, pas réimplémenter T04

La section exacte à consulter est **« Faster optimization procedure
(documentation only) »**, dans le [corps PR61 conservé](pr61-current-body.md).
Elle organise trois étapes :

1. Smoke ciblé de correction, arrêt et diagnostic au premier écart pertinent.
2. Qualification des chemins de production affectés, fidélité numérique,
   états/rollback, MTP/DF et budgets, **avant** la vitesse.
3. Contrôles complets pertinents, CI finale et comparaison MAIN/HOLD équitable
   avant activation ou merge ; réutilisation explicitement justifiée des
   preuves que l'édition n'a pas affectées.

Cette méthode a déjà été appliquée de fait aux campagnes locales, mais
**la section de procédure courante du cloud n'a pas encore été ajoutée
comme telle à `opti.md` local**. L'ajout documentaire recommandé est sa
reprise prospective et identifiable par l'agent propriétaire du journal,
avec les liens vers les reçus locaux et sans effacer les protocoles/échecs
historiques. Cette revue ne modifie pas `opti.md`.

Le petit pilote16 tokens cité par la PR n'est pas une validation de gain
ni un remplacement automatique du bridge17 local : 17 exerce les compteurs
spéculatifs réels attendus ici. De même, ses nombres cloud81/245/312 et ses
matrices24processus ne remplacent pas les nombres inspectés localement.
AGENTS, tentatives de preuve réelles, garde swap, cache3 Gio, isolation et
inspection des sorties non vides restent applicables. **Ajouter les
vérifications ne signifie ni importer les trois options ni promouvoir un
résultat avant les tests physiques correspondants.**

Mise à jour après cette lecture : la procédure prospective a été ajoutée en
tête de `opti.md` par le propriétaire du journal le31 octobre. Elle conserve
les obligations AGENTS et les résultats historiques ; aucun code T04/T18a
n'a été réintroduit. [Bilan T01 et état conservé](README.md).
