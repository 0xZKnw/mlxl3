# Expert partagé pendant les lectures SSD — essai 27

**Rejeté : aucun gain global, prototype retiré.** Le moteur retenu est HC20, publié dans
la PR34. Les essais22–25 restent rejetés. M5/24Gio, MLX0.32.2, modèle Flash
Next EXL3 2,05bpw, révision65c89531 ; mesures demandées sur batterie.

## Changement testé

Après le retour des routes, le moteur peut calculer les experts déjà en RAM
pendant le remplissage des slots froids. Le partagé résident était construit
après les lectures. Le prototype soumet son graphe exact avec les hits dans
un seul appel `Array::async_eval_all`. Il conserve les projections ligne par
ligne, les mêmes experts, les réductions et le budget de cache.

Cette variante concerne OFF, la vérification de deux tokens et la tête MTP.
Sans miss ou lorsque l'overlap est désactivé, aucune nouvelle soumission
asynchrone n'est ajoutée. Aucun poids supplémentaire n'est conservé en RAM.

La piste vient de la revue26 du chemin local et du profil21 : environ27ms
par pas pour les remplissages, vérification cible dominante en MTP. Ces
horloges instrumentées ne sont pas additives et ne prédisent pas le gain.
Elle diffère des essais de l'ancien projet soumettant le partagé avant le
retour des routes ou utilisant un second flux GPU ; un seul flux reste utilisé.

## Correction et première campagne interrompue

Le premier oracle compare90paires à des sorties MoE indépendantes, avec
10/20/143slots,1/2/4lecteurs et overlapON/OFF. Il couvre aussi18erreurs mixtes
hits/misses avec invalidation et reprise exacte. Une mutation supprimant
le partagé est détectée ; la source restaurée repasse le test.

Fmt, Clippy strict, buildMLX/chat,87testsCPU/2ignorés et107testsrelease/98ignorés
passent. Les11testsQwen, l'oracle tenu à part, les3testsSSD et le bridge réel
passent aussi, chacun dans un processus isolé. Tous ces contrôles ajoutent
zéro swapout système. Kani sur le vrai parcours (1/2lignes symboliques et
valeur i8/128, sans hypothèse) échoue dans le compilateur avant obligations.
Aucune preuve du GPU, de MLX ou de l'achèvement asynchrone n'est revendiquée.

Une revue supplémentaire révèle une erreur du prototype : un échec de
préparation du partagé après réservation peut laisser des slots non lus
marqués valides. Un nouveau test reproduit cette erreur sur le premier code.
La campagne commencée est interrompue : un témoin terminé, un candidat
interrompu,14bras non lancés. Aucun gain n'en est déduit.

La correction construit le partagé avant la réservation. Un échec de
soumission invalide le cache et termine toutes les sorties soumises avant
retour ; un échec de lecture fait de même, en conservant le diagnosticI/O.
Le nouveau test injecte une projection partagée invalide, vérifie les slots
encore présents, restaure les poids et contrôle la reprise exacte.
Les preuves initiales sont conservées ; les contrôles corrigés sont séparés
sous `repair/`.

[Premier code et identités](measurements/flash-next-ssd-impl/opt-27-shared-overlap/identity.json),
[mutation](measurements/flash-next-ssd-impl/opt-27-shared-overlap/mutation.json),
[régression reproduite](measurements/flash-next-ssd-impl/opt-27-shared-overlap/failure-reproduction.json),
[campagne interrompue](measurements/flash-next-ssd-impl/opt-27-shared-overlap/speed/result.json).

## Protocole de débit

H20 immuable contre candidat corrigé identifié, avec nouveaux témoins sur
batterie vérifiée avant/après chaque bras. Deux prompts MAIN/HOLD, sans MTP
puis avec MTP, deux tours H-C/C-H. Une chauffe128 et trois générations128 par
bras,127tokens décodés, médiane3. Cache5Gio/89slots par couche, allocateur512Mio,
lecteurs4, overlap1, limite de contexte4096 avec prompts courts. Applications
conservées ; un seul modèleGPU, aucun compilateur ou proveur pendant la vitesse.

Tous textes/hashes128 et préfixes64 doivent rester exacts. Chaque gain apparié
≥1%, étendue≤3points par cellule ; surplus du pic processus≤256Mio et swapouts
système≤256Mio par bras. Ratio intra-bras max/min≤1,25, délai360s et nettoyage
des enfants. Arrêt après8bras si un seuil de vitesse échoue, immédiatement
sur une erreur de qualité, mémoire ou alimentation. Aucune relance pour bruit.


La correction passe les18erreurs de préparation injectées et leurs reprises,
puis les11testsQwen, l’oracle tenu à part,3testsSSD et le vrai bridge en200,17s.
Les14processus ajoutent0octet de swapout. Fmt/Clippy/build,87CPU/2ignorés
et107release/98ignorés passent ; Kani échoue encore avant obligations.
Candidat corrigé3faac16c, tests e17266f8.
[Identités corrigées](measurements/flash-next-ssd-impl/opt-27-shared-overlap/repair/identity.json),
[qualité corrigée](measurements/flash-next-ssd-impl/opt-27-shared-overlap/repair/quality.json),
[contrôles corrigés](measurements/flash-next-ssd-impl/opt-27-shared-overlap/repair/checks.json).

## Résultat corrigé

|Mode/prompt|Témoin tok/s|Candidat tok/s|Écart|
|---|---:|---:|---:|
|baseline/MAIN|9.540|9.332|-2.181%|
|baseline/HOLD|9.192|9.029|-1.773%|
|mtp/MAIN|9.644|9.484|-1.660%|
|mtp/HOLD|8.880|9.042|+1.819%|

Arrêt prévu après8bras : trois cellules sous le seuil. Les32réponses complètes et compteursMTP restent exacts ; tous les enfants sont joints et les empreintes conservées. Le dernier brasMTPcandidat ajoute168493056octets de swapout (160,69Mio), les7autres zéro. Surplus maximal de pic processus4,30Mio ; pics candidats8,637–9,440Gio. Les8bras de confirmation suivants ne sont pas exécutés. Aucun gain intégré, SSD restauré exactement au code publié.

[Résultat brut](measurements/flash-next-ssd-impl/opt-27-shared-overlap/repair/speed/result.json), [inspection](measurements/flash-next-ssd-impl/opt-27-shared-overlap/repair/assessment.json).
