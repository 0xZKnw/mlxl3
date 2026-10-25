# 82F06 — relecture du préflight intégré et des plans

25 octobre2026. Lecture de W après applicationF06 sur62cae (moteur9dc, F01 retiré), des vrais tests intégrés et des quatre plans. Aucune édition W, compilation, preuve, exécution de test, modèle ou GPU par cette revue. Les petits contrôles de structure/hashes ci-dessous lisent seulement les fichiers et données.

## Conclusion

Aucun défaut de marqueur ou de sélection bloquant trouvé. Une incompatibilité de signature apparue pendant le remplacement `chunks_exact`→`as_chunks` a été signalée : l’itérateur reçoit `&[f32;512]`, alors que le callback attendait une slice. Le parent a modifié `expected_ids` en `&[f32;512]` ; cette correction a été relue dans W, sa compilation appartient aux contrôles du parent.

Les dix-neuf entrées qualité comprennent pré/post, neuf CLI invalides et huit invocations natives (composant, métadonnées, routes réelles, loader défaut/0/1, préfixes, MTP). Les noms exacts sont présents dans W et les marqueurs correspondent aux vrais println et compteurs256/384/6. Le loader absent retire aussi la variable héritée avec env-u ; aucune fixture Rust ne modifie l’environnement. Les huit invalidesASCII et byte0xff sont fixés au démarrage du processus. Le résultat non-Unicode reste à observer physiquement : une erreur de setup ou de lancement n’est pas un refus fonctionnel validé.

Bridge : ADAPTIVE1, WIDTH absent et retiré de l’environnement hérité ;43 attendu dans checks.json. Le marqueur n’est pas inventé dans stdout : il faut lire statut, parité, nombre de contrôles et états de jonction du vrai rapport. La limite des événements tardifs de fermeture de JsonProcess reste celle déjà documentée.

Écrans B2 : même CLI, H0/C1, MAIN etHOLD HC puisCH ; seule la variableF06 diffère dans les environnements. F01 absent du JSON et supprimé dans le préfixe des enfants. WIDTH1/ADAPTIVE0, cache3Gio,128warm+128mesure sont conservés. Outils et plans correspondent aux hashes de identity-template.json à cette lecture ; CLI/source intégrée/natif restent volontairement null jusqu’au gel.

## MutationGPU : elle échoue même si les deux chemins partagent le mauvais store

Le patch change le vrai store d’ID en `(tid+1)%EXPERTS`. Il affecte donc legacy et nouveau chemin, et une seule comparaison candidat==legacy serait insuffisante. Le composant ne repose pas uniquement sur cette comparaison : W/native/src/router/flash_top10_tests.rs:98–106 compare chacun des trois résultats à `expected_ids`, tri CPU indépendant par valeur puis indice.

Son **tout premier cas** est déjà discriminant : rows1, FP16, non stridé, fixture0, normalizefalse. Les512entrées sont exactement1, représentables enFP16 ; aucun seuil flottant fragile ou normalisation n’intervient. Avec le tie-break historique, les dix indices attendus sont :

`[502,503,504,505,506,507,508,509,510,511]`

Le shader mutant écrit :

`[503,504,505,506,507,508,509,510,511,0]`

La première valeur diffère nécessairement503≠502. Tous les slots restent initialisés et les stores restent disjoints ; ce mutant ne dépend ni d’un OOB, d’un output non initialisé, d’une course ni d’un NaN. L’échec attendu est donc l’erreur fonctionnelle `independent rank oracle rows=1 fixture=0`, dès la vérification du résultatlegacy. Il ne faut pas attendre le marqueur final256 sur un mutant qui échoue volontairement.

Commande recommandée après construction d’un **binaire mutant distinct**, hors de toute campagneGPU concurrente :

```text
<mutant-native-tests> router::flash_top10_tests::flash_top10_early_exit_matches_independent_ranks_and_legacy_scores --exact --ignored --nocapture --test-threads=1
```

Le chemin ci-dessus est un placeholder à remplacer par le vrai binaire figé, pas un exécutable déjà présent. Environnement fixé avant Popen, autres expériences désactivées ; aucun modèle/checkpoint n’est nécessaire pour ce composant. Exiger exactement1test sélectionné,1failed/0passed, exit101 et le message fonctionnel indépendant ci-dessus. Un timeout, une erreur Metal de compilation, un mauvais chemin, zéro test ou l’absence de ce message ne valide pas la mutation. Conserver les hashes du mutant et du positif ; restaurer puis relancer le composant normal avant le reste. La mutation démontre la sensibilité du tri indépendant, pas la performance du break ni une preuve exhaustive du MSL.

## Couverture et limites à conserver

- Les256cas ne sont pas vides : B1/2/4/8, deux dtypes, deux layouts, huit fixtures et deux normalisations ; chaque résultat a rows×10 IDs, formes/types/tailles vérifiés et scores comparés en octets. Les non-finis intentionnels ne sont pas annoncés finis. Le triCPU traite ±0 égaux et tousNaN au-dessus, avec indice pour les égalités, comme les clés historiques.
- Les384cas réels sont384entrées/layouts, chacune exécutée avec flagfalse puistrue, sur les48gates et deux captures. Ce n’est pas384prompts ni384états complets du modèle. Les vues stridées sont créées après le cast dans les deux dtypes.
- Les36préfixes comparent le vérificateur à une exécution séquentielle sous flag1, avec logits/captures/états ; ils ne sont pas une comparaison exhaustive H0/C1 de tous les logits. La comparaison directe des IDs/poids, les oracles indépendants et les réponses H/C golden60 sont des contrôles distincts. Ne pas leur attribuer la portée du différentiel completF01.
- Le CPUKani vérifie le vrai garde de domaine, pas le MSL. Le harnaisMLX symbolique ne devient pas une preuveGPU si la FFI est non prise en charge. Les oracles de correction ne peuvent pas prouver que le compilateur économise effectivement du travail : cela reste le rôle du débit mesuré/profil approprié.
- Les gardes/identités du JSON ne sont pas appliquées automatiquement par le runnerstock45. Le parent doit vérifier/fixer les SHA et satisfaire les préconditions avant lancement. Le composant positif/mutant/restauré est une étape séparée avant le lancement enchaîné de quality/run.json ; ne pas tenter d’insérer un build/proveur au milieu de la matriceGPU.
- Aucun élargissement de la suiteEXL3/F01 non affectée n’est requis par cette lecture. OFF/MTP1 de débit restent nécessaires avant promotion d’un gainF06, après sa qualité et ses paires B2.

État final de cette revue : analyse statique terminée ; tous les résultats physiques/formels/de débit restent ceux que le parent exécutera et inspectera, sans résultat nouveau revendiqué ici.
