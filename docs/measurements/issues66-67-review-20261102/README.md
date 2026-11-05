# Lecture des issues 66 et 67 — 2 novembre 2026

Les deux dernières issues créées sont [#66](https://github.com/0xZKnw/mlxl3/issues/66)
et [#67](https://github.com/0xZKnw/mlxl3/issues/67), ouvertes le 2 novembre 2026
à 08:45:12 et 08:45:21 UTC, sans commentaire au moment de la lecture. Les corps
intégraux sont conservés dans `issue-66.md` et `issue-67.md`. Les versions de la
recherche GitHub et de la récupération directe sont identiques.

Revue de main d5a776e53425bb4d0ba61c0774c5399ae7987c06 dans les issues, comparée
au même HEAD avec modifications locales conservées, branche
`codex/decode-profile-p09-20261101`. Lecture du journal et du code uniquement :
aucun prototype, test, build, profilage GPU ou publication. La comparaison #66
a été déléguée à un agent en lecture seule ; #67 et les conclusions ont été
recoupées dans le code par l'agent principal.

## Priorités retenues

1. **Attribuer l'attente Cold résiduelle sur la baseline Dynamic+ColdU, puis
   filtrer la priorité de lecture par multiplicité B2.** C'est la priorité SSD
   ajoutée en tête de #66 ; elle remplace explicitement l'ordre de son annexe.
   Les jobs locaux restent dédupliqués par slot dans l'ordre des premières
   occurrences (`ssd_moe.rs:1231`), puis `wait_ready(2)` attend deux jobs uniques
   (`ssd_moe.rs:471`). Un expert commun aux deux lignes peut débloquer deux
   occurrences pour les mêmes octets. Aucun tri par cette multiplicité dans
   le chemin examiné. Avant prototype : compter les doublons Cold utiles,
   l'ordre réel d'achèvement, la taille W1 et la queue exposée. Des doublons
   rares ou des données déjà masquées rendraient cette politique peu utile.
   Changer l'ordre de lecture ne doit pas changer l'ordre des réductions.
2. **Piste DFlash plus ambitieuse : ColdReady B3/B4, puis géométrie EXL3 MB3/4.**
   Les deux verrous de #67 sont encore actuels : `split_cold` impose `rows<=2`,
   pas de shared_exl3 ni d'épilogue pondéré (`ssd_moe.rs:1251`), et le layout
   Flash reference renvoie MB2 pour toutes les largeurs2..8
   (`contracts.rs:626`, utilisé par `linear.rs:687`). Une comparaison B2/B4
   change donc à la fois le nombre de propositions et la disponibilité des
   vagues Cold ; B4 ne partage pas automatiquement une tuile entre quatre
   lignes. Qualifier le scheduling B3/B4 seul avant la géométrie MB3/4 ; ce
   n'est pas simplement retirer une garde. Le shader référence réserve
   `MB*NT*SG*256` floats, soit 8/16 Kio pour MB2/MB4 avec NT1/SG4, auxquels
   s'ajoutent ses autres tableaux et registres. Gain global non mesuré.
3. **Callbacks MLX comme complément ciblé.** Le protocole de #66 sépare bien
   A, handlers regroupés avec les mêmes conteneurs, puis B, représentation
   compacte/déduplication. C'est plus discriminant qu'un patch mélangeant les
   deux. La donation, les buffers en vol, les erreurs et les événements restent
   les contrats centraux. Le coût propre n'est toujours pas isolé ; le total
   des soumissions P09 ne peut pas lui être attribué.

## Écarts main / travail local déjà terminé

- Dynamic+ColdReady est déjà compatible localement et Dynamic vaut 1 par
  défaut (`ssd_moe.rs:178`, `disk_read_batch.h:103`). Le diagnostic Dynamic
  antérieur est conservé dans le journal ; la nouvelle attribution doit viser
  seulement ce qu'il ne tranche pas, sans grand profilage répété à l'identique.
- Shared après le départ SSD a déjà son écran court : AR MAIN−0,150%,
  HOLD+0,334%, sans gain établi, flag OFF. Sa recommandation dans #66 précède
  cette clôture locale et ne justifie pas une répétition inchangée.
- L'exclusion de la tête MTP de DEC-ONE a déjà été implémentée et auditée,
  puis les résultats R3/direct ont été conservés sans gain stable établi.
  L'utilisateur a explicitement abandonné cette piste. L'ablation suggérée par
  l'annexe #66 ne la réouvre pas.
- B4/B8 et experts groupés ont déjà été essayés dans 65T/66F : résultats
  négatifs d'écran et limites de dérive conservés. La nouveauté de #67 est la
  composition avec ColdReady et le vrai partage dense MB3/4, pas « essayer une
  largeur supérieure » en soi. Pas d'attribution rétrospective des anciennes
  régressions à une garde introduite plus tard.
- PreDraftBudget stocke déjà des coûts séparés B2/B3/B4
  (`flash_draft_budget.rs:9`). Le contrôleur VerifyBudget emploie aussi une
  approximation linéaire : comparer le mécanisme existant avant remplacement.

## Autres constats de #66 confirmés statiquement

| Point | État local | Implication |
|---|---|---|
| Configuration effective | Hot/Wide/Unique par défaut0, benchmark tuned les force ; Swift hérite de l'environnement | Reçu requested/effective utile avant extrapolation à l'app ; application et binaire local qualifié restent distincts |
| Timeout disque | wait_ready détecte après30s, destruction appelle join avec attente FOREVER | Robustesse réelle à traiter séparément ; blocage effectif non reproduit, aucun TPS attendu, ne pas détruire les banques avec un worker vivant |
| Indices de réplication | routed_input recrée Vec et Array UInt32, contenu dépendant du nombre de lignes (`ssd_moe.rs:809`) | Petit candidat borné composable ; supprimer l'allocation ne supprime pas le take GPU |

## Portée des objectifs et suites possibles

B2 signifie ancre + une proposition, B3 deux propositions, B4 trois. Le débit
utile dépend des préfixes réellement acceptés et du coût du cycle entier,
y compris union d'experts, draft et commit. Le taux d'acceptation de la
première proposition ne fournit pas celui des suivantes. Le budget de15tok/s
est66,7ms/token utile ; celui de20tok/s est50ms. Ce sont des critères, pas des
gains démontrés. Les références release P09 des issues précèdent Dynamic+ColdU
local ; aucune comparaison fraîche n'a été mesurée pour cette revue.

La prochaine expérience nécessitera son propre préenregistrement, qualité
avant vitesse, mêmes options effectives et états/prefixes/erreurs contrôlés.
Les recommandations externes ne remplacent ni l'ancre ni les plafonds swap
explicitement autorisés dans la conversation. Aucun protocole GPU n'a été
lancé ou modifié ici. Les articles et performances amont cités dans #67 n'ont
pas été reproduits ou audités intégralement ; les constats locaux ci-dessus
sont fondés sur notre source et l'historique.

Statut : **revue close**, aucun essai en cours, aucune modification exécutable.
