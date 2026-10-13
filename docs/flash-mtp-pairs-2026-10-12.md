# Flash Next : calcul exact par paires MTP, essai 15

État : essai 15 non concluant, confirmation 15B rejetée, prototype T2 retiré. Le moteur conserve les quatre lecteurs SSD du commit `2e83f157`, validés par 14B. La piste 11, seulement préparée auparavant, a été implémentée et testée ici ; aucun résultat de 14B n'est réutilisé comme témoin chronométré de 15.

## Portée

Le candidat regroupe sur deux lignes les projections EXL3, les connexions HC, les projections indépendantes de GDN, les experts partagés et la tête de vocabulaire. Les poids sont ainsi réutilisés dans les projections. Les opérations récurrentes GDN, QSA et PLE restent séquentielles ; le snapshot après le premier token permet toujours de conserver exactement un ou deux tokens. Les petites projections denses restent calculées par ligne pour préserver l'arrondi GEMV.

Le kernel Metal existant permet déjà MB2. Le chemin Flash utilise sa partition de référence, quatre simdgroups et NT1, avec la réduction des splits effectuée séparément par ligne. La tête MTP `fc_hidden` conserve son chemin MB4. Aucun nouveau kernel, changement de précision, choix approximatif d'experts ou option de lancement n'est ajouté. Le prototype NT2 de l'essai 10 et l'ancien budget de 8 Gio ne sont pas réintroduits.

Sources du prototype : commit de départ `2e83f157d986f8c673790c3dc2325254b08cab2f`, puis modifications locales de `linear.rs`, `qwen4.rs`, `ssd_moe.rs` et ajout du contrat QMV au job Kani ciblé. [Identités exactes](measurements/flash-next-ssd-impl/opt-15-t2/identity.json) et [diff préparé](measurements/flash-next-ssd-impl/opt-15-t2/selected.patch). Les tests ajoutés et le harnais réel GPU sont conservés dans le patch archivé. L'implémentation et ces ajouts ont été retirés des sources actives après 15B.

## Vérification avant mesure

- Format, Clippy strict avec `mlx,chat`, build release : réussis. `cargo test --locked --no-default-features --features chat` : 87 tests passent, 2 ignorés. `cargo test --locked --release --features mlx,chat` : 107 passent, 97 ignorés avant sélection explicite des tests physiques. [Commandes et logs](measurements/flash-next-ssd-impl/opt-15-t2/build.json).
- Huit projections réelles : MB2 égale bit à bit deux appels T1, en FP16/FP32, avec entrées non contiguës et dimensions de tête supplémentaires. Formes, nombre de valeurs, finitude, résultat non vide et rejet des entrées vides sont vérifiés. Le test HC compare lecture, portes et écriture par paire, et rejette les formes incompatibles.
- Mutation réelle de l'ordre d'accumulation MB2 : divergence détectée sur `shared_expert.down_proj`. Après restauration, le test repasse et le binaire de tests retrouve exactement son SHA initial. [Échec attendu et restauration](measurements/flash-next-ssd-impl/opt-15-t2/mutation.json).
- 20 exécutions physiques réussissent, hors mutant : nouveaux tests, répétition après restauration, 12 tests Qwen, référence tenue à part, 3 tests SSD et oracle des slots. Les 16 positions indépendantes gardent leurs logits et états exacts ; les commits 1/2 et le token suivant gardent aussi leurs états. Les tests SSD comprennent 90 paires de cohortes avec caches froids/chauds, tailles 10/20/143, lecteurs 1/2/4, overlap et erreurs. [Qualification](measurements/flash-next-ssd-impl/opt-15-t2/quality.json).
- Le bridge réel passe les budgets 1/2/3/17 en OFF/ON/ON/OFF, les cinq rejets de configuration, les deux annulations avec reprise et le fallback de pénalité. [Rapport](measurements/flash-next-ssd-impl/opt-15-t2/bridge.json), [événements](measurements/flash-next-ssd-impl/opt-15-t2/bridge.log). Ces timings fonctionnels ne sont pas des mesures de gain.

Kani 0.68.0 vérifie dix harnais CPU réels : 1 121 obligations et 33 couvertures, dont trois obligations inatteignables dans les contrats existants. Le layout QMV couvre tout `i32`, avec booléens symboliques et indices valides ; les commits couvrent tout `i32`, les budgets tout `usize`, les offsets `u64`. La partition est bornée à 26 déroulements ; les contrats MTP à 4 ou 6, l'acceptation à 9 pour au plus sept propositions. Les assertions de déroulement restent actives. La tentative du parcours T2/MLX réel échoue dans le compilateur, `intrinsics.rs:243`, sans stub ; aucune propriété GPU n'est prouvée. [Commandes et diagnostics](measurements/flash-next-ssd-impl/opt-15-t2/proofs.json).

## Protocole de débit et de mémoire

M5 24 Gio, macOS 27.2, MLX 0.32.2, Qwen3.8 Flash Next EXL3 2,05 bpw, révision `65c895314393431c09050b2e04e250836b3a6eb4`. Applications de l'utilisateur ouvertes. Cache fixe de 5 Gio, 89 slots par couche, allocator MLX 512 Mio, quatre lecteurs SSD, contexte 4096, décodage greedy. Un seul modèle GPU, aucune compilation ou preuve locale pendant les mesures.

La [matrice préenregistrée](measurements/flash-next-ssd-impl/opt-15-t2/matrix.json) contient douze paires, 24 processus : H-C, C-H puis H-C, pour MAIN/HOLD et OFF/MTP. Chaque processus fait une chauffe puis une génération mesurée de 64 tokens, soit 63 tokens décodés. Toutes les réponses, y compris les chauffes, doivent retrouver le texte et le hash historiques. VM, alimentation et état thermique sont relevés avant et après chaque processus ; le pic provient du processus natif, pas des seules allocations MLX.

Le témoin est `bd89a7d6`, le candidat `4fc51f1e`. Les critères fixés avant les nouvelles données sont : chaque gain MTP apparié ≥3 %, minimum candidat MTP supérieur de ≥1 % au maximum témoin, chaque gain OFF ≥−3 % et médiane OFF ≥−1 %. Le pic candidat ne doit pas dépasser son témoin de plus de 256 Mio ; aucun bras ne doit écrire plus de 256 Mio de swap système. Toute divergence interdit une validation. Les mesures et critères de 14 restent séparés.

## Résultat 15 : non concluant selon son critère

Les 24 processus et 48 réponses sont complets et exacts, sans nouvelle écriture de swap système. Les sources et binaires n'ont pas changé. Toutes les limites de mémoire et les six contrôles de non-régression OFF passent. Les valeurs restent dans le [résultat intégral](measurements/flash-next-ssd-impl/opt-15-t2/result.json) et les [bruts](measurements/flash-next-ssd-impl/opt-15-t2/execution.json).

| Mode / prompt | Témoin médian (tok/s) | Candidat médian (tok/s) | Gains des trois paires |
|---|---:|---:|---|
| OFF / MAIN | 6,920 | 6,921 | −0,06 %, −0,39 %, +0,08 % |
| OFF / HOLD | 6,887 | 6,895 | +1,18 %, +0,88 %, −0,05 % |
| MTP / MAIN | 6,947 | 7,288 | +4,91 %, +4,79 %, +4,85 % |
| MTP / HOLD | 6,821 | 7,161 | +5,35 %, +4,99 %, +5,81 % |

La dernière paire HOLD monte ensemble à 7,246/7,667 tok/s. Le minimum candidat HOLD de 7,146 devient inférieur au maximum témoin de 7,246 : ratio 0,9862, sous le seuil préenregistré de 1,01. Malgré les six gains appariés ≥3 %, ce critère échoue. La cause du déplacement commun n'est pas établie. Aucun bras n'est exclu, aucun gain 15 n'est déclaré validé. Le [prototype complet avec ses tests](measurements/flash-next-ssd-impl/opt-15-t2/prototype-15.patch) est archivé ; le lecteur 14B reste le code publié à ce stade.

## Confirmation 15B : protocole corrigé

Une seule campagne indépendante est préenregistrée après cet échec. La séparation des enveloppes absolues de 15 est remplacée par deux exigences sur les paires adjacentes : chaque gain ≥3 % et étendue des quatre gains ≤2 points de pourcentage dans chaque prompt. Cela mesure la répétabilité relative des passages observés, sans supposer un débit absolu constant ni revendiquer un test statistique IID. Le résultat 15 conserve sa conclusion.

La [matrice 15B](measurements/flash-next-ssd-impl/opt-15-t2/paired-confirmation/matrix.json) prévoit huit paires MTP, 16 processus, quatre répétitions par prompt, en ordre H-C/C-H/H-C/C-H. Les données sont nouvelles, les binaires et tous les réglages inchangés. Les mêmes contrôles de texte, hash, mémoire et swap restent obligatoires. Les six contrôles OFF déjà réussis dans 15 ne sont pas répétés ; leurs débits ne sont pas mélangés à ceux de 15B. Tout échec des nouveaux critères entraîne le retrait du prototype T2, sans nouvelle relance de cette même idée.

### Résultat 15B : rejet et arrêt des passages restants

| Paire complète | Témoin (tok/s) | Candidat (tok/s) | Gain |
|---|---:|---:|---:|
| MAIN 1 | 7,035 | 7,326 | +4,15 % |
| HOLD 1 | 6,813 | 6,968 | **+2,29 %** |
| HOLD 2, ordre inversé | 6,800 | 7,107 | +4,51 % |

HOLD 1 échoue au seuil de 3 %. Le superviseur est arrêté proprement ; sept bras complets restent conservés, un est interrompu et huit ne sont pas lancés. Le candidat MAIN 2 a terminé à 7,058 tok/s, mais son témoin est interrompu : aucun gain ne peut être calculé pour cette paire. Les quatorze réponses des bras complets sont exactes et les nouvelles écritures de swap nulles ; cela ne valide pas la campagne complète. [Motif d'arrêt](measurements/flash-next-ssd-impl/opt-15-t2/paired-confirmation/stop-reason.json), [résultat](measurements/flash-next-ssd-impl/opt-15-t2/paired-confirmation/result.json), [tous les passages](measurements/flash-next-ssd-impl/opt-15-t2/paired-confirmation/execution.json).

Le regroupement est numériquement correct sur les contrôles exécutés, mais son gain ne satisfait pas les critères de rétention. Aucun nouveau gain MTP n'est livré. Les trois fichiers Rust et le workflow sont restaurés octet pour octet à `2e83f157`. La cible locale est reconstruite pour retirer aussi le binaire expérimental. Format et builds passent ; le binaire de tests retrouve exactement `b564c6dd`, déjà qualifié par 14/14B.

Le contrôle initial attendait également le SHA moteur `bd89a7d6` et échoue : le rebuild produit `4fa70acb`, car `build.rs` embarque maintenant `2e83f157d986-dirty` à la place de `bdc3889b6f3f-dirty`. Cet échec reste dans le [contrôle de restauration](measurements/flash-next-ssd-impl/opt-15-t2/restore.json). Une [inspection complète des octets](measurements/flash-next-ssd-impl/opt-15-t2/restore-identity-inspection.json) confirme que les seules différences sont les deux occurrences de cette révision, le UUID Mach-O et la signature générée ; tous les autres octets sont identiques, avec même taille et version CLI. Aucun nouveau débit n'est attribué à ce rebuild. Les timings du prototype retiré ne qualifient pas le moteur conservé.

## Limites

La cible de 15 tok/s n'est pas atteinte. Le résultat 15 et sa limite restent conservés. Les contextes QSA au-delà de 2048, le multimodal et une preuve formelle Metal restent hors de cette validation. La CI du témoin `2e83f157` passe les jobs Linux, macOS, contrats ciblés et desktop/protocole ; son Kani complet était encore en cours lors de l'inspection. L'application installée n'est pas remplacée.
