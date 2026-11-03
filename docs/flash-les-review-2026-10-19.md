# Revue du prototype LES32 — 19 octobre 2026

Le changement ajoute une prédiction des experts de la couche suivante et quatre
lecteurs SSD non bloquants. Les poids achevés sont copiés seulement lorsqu’un
expert réellement demandé en a besoin. Le cache reste à 5 Gio, avec 4 992 048
octets de staging supplémentaires.

Portée : diff local sur `fb782d98`, moteur immuable `1821488c`, modèle
`65c89531` EXL3 2,05 bpw, un utilisateur et un modèle GPU sur M5/24 Gio.
Revue statique pendant les mesures32 ; aucun code mesuré modifié. Les chemins
de référence sont `qwen4::FlashNext::{forward_inner,verify_pair}` → `Layer` →
`SsdMoe::{forward_lookahead,fill_many,prefetch}` → `ReaderPool` et les copies FFI.
Les tests, les appels de projection, le routeur, les processus de benchmark et
les workflows PR/push ont également été inspectés.

## À corriger avant publication du prototype

1. **Des résultats terminés occupent encore les quatre lecteurs.**
   `native/src/ssd_prefetch.rs:118`, `native/src/ssd_moe.rs:158` et `:478`.
   Le polling a lieu avant les lectures de demande. Si les prélectures se
   terminent juste après ce polling, les quatre champs `pending` restent remplis
   au moment d’admettre les prédictions suivantes, même si les workers ont fini.
   Il faut récupérer les résultats périmés une fois avant la nouvelle cohorte,
   sans attente, et ajouter une régression déterministe couvrant cette séquence.
   Sans correction, le modèle reste exact mais perd une fenêtre de recouvrement.

2. **Le budget de prédiction peut être entièrement consommé par des experts chauds.**
   `native/src/ssd_moe.rs:327` et `:485`.
   En profondeur4, seuls les quatre meilleurs IDs sont lus ; les hits sont
   filtrés ensuite. Si ces quatre experts sont déjà en RAM mais le cinquième
   manque, aucune lecture n’est lancée malgré des workers disponibles.
   Chercher un ensemble fixé plus large, filtrer les hits, puis borner les
   nouvelles lectures à2/4 conserve le budget mémoire. Sans cela, on paie le
   routeur supplémentaire sans masquer les lectures des rangs suivants.
   Le gain de cette modification reste à mesurer.

3. **La prédiction MTP répète du calcul dont elle n’a pas besoin.**
   `native/src/ssd_moe.rs:330`, `native/src/router.rs:91`.
   Deux lignes entraînent deux appels de projection et deux top-k. Le top-k
   générique compare chaque expert aux512 autres, soit262144 comparaisons par
   ligne, puis produit des scores inutilisés. `Projection::forward` accepte
   déjà plusieurs lignes et `Array::argsort` est déjà exposé par la FFI MLX.
   Tester ces primitives uniquement pour la prédiction permet de conserver
   intact le routage numérique réel. Le gain de débit n’est pas encore établi.

Les cas1/2 sont des scénarios vérifiés par lecture du chemin de production ;
leurs reproductions exécutables ne sont pas encore lancées à cette révision.
Les262144 comparaisons sont un compte statique, pas un chronométrage GPU.

## Vérification et limites

LES32 a passé18 tests physiques, les logits248320 et états complets, les
commits MTP1/2 et le bridge réel. Les235 cas C++ passent ASan/UBSan et TSan ;
ce dernier contrôle les lecteurs/copies C++, pas la concurrence du pool Rust.
88 tests CPU,108 release et100 tests Python passent. Les tentatives Kani sur
les workers et MLX échouent dans le compilateur avant les obligations ; CBMC
et CrossHair sont absents. Le contrat CPU historique des intervalles passe
sur le domaine `usize`64 bits, avec ses trois couvertures, sans prouver le pool.

Le test à caches froids vérifie la correction ; ses taux de prélectures utilisées
ne décrivent pas un cache chaud en conversation. Le cribleOFF32 reste
non concluant : la variation des témoinsMAIN dépasse le seuil préenregistré.
La mesureMTP32 est en cours. Aucun gain32 publié, aucune app remplacée et
aucune preuve complète de concurrence/GPU annoncée.

Verdict : corriger1/2 et mesurer3 avant de publier cette optimisation.

Non vérifié : effet de ces corrections sur le débit réel ; objectif15tok/s.

## Révision après reproductions et correction locale (19 octobre)

Les cas1/2 échouent effectivement sur le code32 inchangé : résultats terminés
bloquant l’admission, puis quatre prédictions chaudes empêchant tout préchargement
utile. Les tests de régression33 passent ensuite. La prédiction33 utilise une
projection groupée1/2lignes et le tri MLX des16 meilleurs candidats, intercalés
par rang ; les hits et doublons sont filtrés avant le budget de2/4 lectures.
Les sources, empreintes, échecs et sorties sont conservés dans
[opt-33-les-admission](measurements/flash-next-ssd-impl/opt-33-les-admission/).

Un quatrième écart a été reproduit sur33 : quatre experts réellement routés
et déjà préchargés restaient dans le lot « froid ». Leur calcul attendait donc
les six autres lectures SSD. La sortieMoE était exacte mais le compteur du
chemin réellement soumis au GPU restait à0 au lieu de4. Le correctif34 importe
les seuls records prêts et demandés avant de séparer les calculs chauds/froids ;
les compteurs et la politique LFU conservent leurs contrats. Le test échoue avant
le correctif, puis passe avec les autres tests physiquesSSD. Voir
[reproduction34](measurements/flash-next-ssd-impl/opt-34-les-ready/reproduction.json).

Le cribleMTP32 achevé est favorable (+6,78%MAIN/+6,30%HOLD), mais sa confirmation
n’a pas été exécutée avant remplacement du prototype ; aucun gain32 publié.
Le diagnostic33 a passé ses contrôles numériques mais dépassé la limite mémoire
avec283,39Mio de swapouts système. Il était en modeLES0 avec la têteMTP et les
instantanés résidents ; ce relevé global ne permet pas d’attribuer tout le swap
au modèle. Campagne33 arrêtée, aucun débit33 mesuré. Le candidat34 est vérifié
avec un diagnostic séparé par prompt et par mode, sans instantanés massifs.
Le journal `opti.md` et les résultats34 font foi pour l’état final de validation.
