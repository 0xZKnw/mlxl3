# Outillage A/B : validation CPU finale

Le driver `benchmarks/compare_native.py` accepte les ordres ABBAAB/BAABBA et un
budget global optionnel pour les attentes de chargement, échanges, sondes et pauses.
Après fermeture et join, il exige aussi un arrêt normal du bridge ; des réponses
valides suivies d’exit1 ou d’une terminaison forcée produisent `failed`, sans faux
`parity:true`, en conservant les réponses utiles. Aucun changement de
`native_json_process.py` ni dépendance ajouté.

## Vérification et régression

- Ancien vrai CLI isolé :2 régressions EOF rouges attendues (exit0/paritytrue).
- Source finale :110 tests pertinents passés,0 ignoré,51,26s ;13 cas affectés
  repassés,44 désélectionnés,14,82s. Driver, JsonProcess, ancien bridge et évaluateur
  qualité ; pas la suite complète du dépôt.
- Ruff strict, format, py_compile et diff-check passent.137PID de fixtures vérifiés :
  aucun vivant. Tous les sous-processus du superviseur sont joints.
- Les13 blocs shell du workflow Rust passent bash-n. YAML BaseLoader conserve `on` ;
  PR/push, permissions lecture, harnais API réel,7 tests CPU MLX avec sélections
  exactes et garde «1 passed» vérifiés. Le run GitHub du commit final reste à inspecter.

Le [manifeste final](benchmark-tool-verification.json) contient les commandes,
retours, bornes et SHA des sources. L’[archive brute](benchmark-tool-evidence.tar.gz)
conserve les versions précédentes, échecs, mutations, sorties finales, source
ancienne isolée et rapports des fixtures EOF. Chaque entrée extraite a été vérifiée
byte à byte. Les anciens budgets de fixtures0,4s pouvaient expirer avant ready sous
charge CPU :2 échecs conservés, budgets des tests ajustés avant reprise11/11, puis
13/13 avec EOF. Aucune adaptation de production n’a masqué ces échecs.

## Portée et limites

CrossHair avait exploré une postcondition du vrai helper `bounded_deadline`,
confirmée dans son modèle ; le helper final est byteidentique et ce résultat n’a
pas été relancé. Ce contrôle symbolique ne prouve pas l’I/O, l’horloge physique,
le cycle de vie des processus ou le GPU. Les nouvelles régressions EOF exercent
le vrai CLI et ses enfants ; ce sont des tests, pas une preuve complète.

Le cleanup existant laisse0,25s à l’arrêt EOF normal, puis terminate1s et kill/wait5s.
Une destruction saine plus lente que0,25s peut donc être rejetée par le driver ;
une terminaison forcée n’est jamais qualifiée comme arrêt sain. Le budget global
porte les attentes avant cleanup ; les écritures de rapport et l’ordonnancement
système ne bénéficient pas d’une garantie temps réel.

Aucun modèle GPU, benchmark physique, commit, push ou publication par ce sous-lot.
La parité texte/token/historique du driver ne remplace pas les contrôles numériques
fullvocab/états/routes décrits dans le [plan physique](physical-plan.md).
