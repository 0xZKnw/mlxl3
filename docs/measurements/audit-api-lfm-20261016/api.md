# API : correctifs et vérifications locales

État : **code API et transport vérifiés sur fixtures CPU**, sans modèle GPU.
Le [manifeste final](api-verification.json) a été confirmé par l’agent API ; les
empreintes de ses quatre sources ont été recoupées avant la rédaction. Il ne
certifie pas la future release, la CI distante ou les optimisations LFM.

## Comportement corrigé

| Audit | Problème | Résultat local et contrôles |
|---|---|---|
| F01 | Budget de sortie rapporté comme arrêt naturel | Le bridge transporte une cause explicite `eos`, `max_tokens` ou `context_full`. API streaming et JSON mappent EOS vers `stop`, les deux limites vers `length`. Un tag absent/inconnu est une erreur de protocole, pas une cause inventée. |
| F02 | Requêtes accumulées derrière le mutex | Une seule admission de génération, obtenue avant lecture du corps ; saturation refusée avec429 et `Retry-After: 1`. Une lecture partielle ou lente expire puis libère la place. |
| F03 | Client parti avant le premier delta | Détection de fermeture indépendante des deltas, annulation du bridge, drainage de la bonne requestID et nettoyage avant la requête suivante. En cas de protocole cassé, session invalidée et503. |
| F04 | Producteur ou vrai client TCP qui ne lit plus | Taille de sortie et frames bornées ; délai du producteur séparé du délai d’écriture socket à travers le buffering Hyper. Un client TCP non lecteur est fermé avant de reprendre sa lecture dans le test. |
| F05 | Health positif après mort du bridge | Liveness/readiness tiennent compte du processus réel, de l’état occupé et d’une session invalidée. La récollection d’un enfant mort ne laisse pas un PID réutilisable être signalé ensuite. |
| F06 | Origin/Host/auth locale ambiguës | Vérification d’autorité loopback IPv4/IPv6, rejet des origines nulles/étrangères/malformées ou dupliquées, authentification et préflight cohérents. Hors loopback, clé requise. Aucune promesse de limite globale de connexions. |
| F08 | Capacités non alignées sur le validateur | `/v1/capabilities` décrit la voie texte/API réellement acceptée et filtre les métadonnées privées du ready. Les champs outils/vision/JSON non pris en charge restent refusés. |
| F09 | Conformité et erreurs de protocole insuffisamment exercées | Compteurs de tokens consommés strictement typés et bornés ; `stats:null`, champs manquants/booléens/négatifs ou dépassement de budget refusés au lieu d’un200 trompeur. Erreurs bridge longues tronquées sur frontière UTF-8 avant HTTP/SSE. |

Le cycle de vie inclut également : démarrage silencieux/ready invalide/pipes absents,
SIGTERM pendant chargement, EOF/JSON partiel ou événements anciens continus,
arrêt après ready, récollection de l’enfant et join du lecteur stderr.
Les sources réelles sont `native/src/openai.rs`, `openai_io.rs`, `openai_tests.rs`
et le point d’entrée d’intégration `tests/openai_transport.rs`.

Les limites de production lues dans `Limits::default` sont distinctes : corps 16 Mio,
événement bridge 64 Mio, sortie 16 Mio ; corps et commande 10 s, attente événement 300 s,
génération 1800 s, envoi/drainage 10 s. Les fixtures emploient des limites réduites pour
exercer ces transitions. Le timer socket démarre au premier `Pending` et n’est
réinitialisé qu’au flush complet ; un petit débit intermittent ne repousse pas
indéfiniment la même phase. Ce timer n’est ni la durée de vie absolue d’une connexion
ni un quota global de connexions HTTP.

## Résultats inspectés

- `cargo test --locked --no-default-features --test openai_transport` :
  **33 passés, 0 échec, 0 ignoré**, 0,52 s ; `final33-tests.log`.
- `cargo clippy --locked --no-default-features --test openai_transport -- -D warnings` :
  passe ; `final33-clippy.log`.
- Format des quatre fichiers API et contrôle de whitespace : passent selon le
  manifeste final. Aucun Cargo ou proveur relancé pour rédiger ce rapport.
- Régression F01 rouge avant correction : `stop` au lieu de `length`.
  Régression compteurs rouge : `stats:null` produit 200 au lieu de 502.
  Régression taille d’erreur rouge : corps 12055 octets dépassant l’oracle 5000 octets.
- Trois mutations détectées, puis sources restaurées octet à octet : fermeture
  client avant premier delta (500 ms, avant délai événement 10 s), backpressure
  (libération admission 1 s), timer socket (vrai client TCP non lecteur, borne 3 s).
  Les trois sorties d’échec sont conservées ; une comparaison de texte n’est pas
  utilisée pour valider ces propriétés de cycle de vie.

Les échecs de préparation restent visibles dans le manifeste : premier filtre
sélectionnant 0 test, diagnostic de syntaxe Kani `--test` au lieu de `--tests`,
fixtures Python dont un `print` interrompu tronquait une ligne JSON, première
oracle TCP observant un abort Body avant le timer socket, et diagnostics Clippy
corrigés. Le serveur rejetait correctement la ligne JSON tronquée ; cette erreur
n’est pas effacée comme si la première campagne avait réussi.

## Vérification source et limites

| Contrat réel | Domaine et résultat | Ce que cela ne prouve pas |
|---|---|---|
| `GenerationStopReason::from_round` | Deux booléens symboliques, quatre combinaisons, sans hypothèse : 15 obligations réussies et 4 couvertures satisfaites dans le contrôle ciblé root, 0,66 s. | Parsing/mapping des chaînes, sérialisation bridge, génération GPU. Ces raccordements sont testés séparément. |
| `TokenUsage::new` et `total_tokens` | Tous les `u64` prompt/completion et `Option<u64>` budget, sans `assume` ni boucle : **65 obligations réussies + 3/3 couvertures**, 0 échec, 2,77 s, superviseur 30 s. Prompt doit être positif, completion 0 admise, budget présent respecté ; total exact si possible et saturé à `u64::MAX` sinon. | Parsing JSON, entrée HTTP, signaux, Hyper/Tokio, concurrence, tous entrelacements ou génération complète. |
| Parseur Origin réel | Suffixe arbitraire de deux octets après `http://localhost`, unwind 64 : compilation puis CBMC atteint, **timeout 30 s** avec kill/join. | Propriété non vérifiée ; absence de résultat n’est ni succès ni contre-exemple. Ce harnais non concluant n’est pas ajouté à la CI. |

Ces résultats Kani portent sur les fonctions CPU réelles et leur domaine décrit,
pas sur un modèle réécrit. Les tests de transports sont des scénarios déterministes
concrets et ne constituent pas une preuve de tous les entrelacements réseau.

La CI PR/push existante découvre `openai_transport` ; le workflow de cette campagne
ajoute explicitement le harnais usage avec `--no-default-features --tests --harness
validated_usage_respects_budget_without_total_overflow --jobs 1`. Le contrôle Kani
`--lib` seul ne le sélectionnait pas. La CI du SHA livré doit encore être inspectée ;
modifier le workflow ne constitue pas une CI verte.

Les logs bruts et leurs empreintes sont conservés dans
`build/audit-api-lfm/api/` et `build/audit-api-lfm/root/`.
L’[archive API](api-evidence.tar.gz) (46 353 octets, SHA-256
`7ed305edc9ca567d3c7dd4440d62edbad38bc21a932519a35f7e9adc7639e78d`)
contient les quatre sources API vérifiées octet à octet contre le manifeste, ses
logs CPU, les échecs conservés et les logs parent F01. Les 43 entrées ont été
relues et comparées à leur contenu prévu, sans extraction. Elle ne contient ni
poids ni entrées privées ni binaire. Les logs parent F01 décrivent leur étape
historique mais ne figent pas une copie complète du code parent à cette date ;
l’archive ne revendique pas cette identité historique ni celle d’une release.

## Périmètre restant

Aucun poids, calcul de logits ou paramètre de quantification modifié par ce lot API.
Aucun KL/PPL ou gain LFM revendiqué. Aucun modèle chargé par les fixtures API.
Le build livré `mlx,chat`, les suites de dépôt, le vrai bridge avec modèle et les
runs GitHub du commit exact relèvent de l’intégration root, pas de ces33 tests.
L’observation EOF peut être retardée par des données HTTP/1 pipelinées ; les délais
globaux et le timer d’écriture restent les bornes de récupération documentées.
