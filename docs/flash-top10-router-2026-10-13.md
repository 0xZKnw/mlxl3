# Routeur SIMD top10/512 — essai 28

**Qualité validée, débit interrompu par changement d’alimentation.** Le prototype27 rejeté est retiré.
Le candidat28 seul est comparé au moteur retenu20 avec de nouveaux témoins
sur batterie. Aucun gain28 n’est encore établi.

Flash appelle le routeur legacy512/top10, qui classe chaque valeur en la
comparant aux512autres. La structure SIMD existante de MTPLX, adaptée dans
ce dépôt pour256/top8/FP16, n'est pas sélectionnée sur cette forme.
Le prototype l'étend séparément à512/top10, une ou deux lignes, F16/F32.
Il garde dix gagnants dans chacun des16SIMDgroups, puis fusionne ces160
candidats dans un seul SIMDgroup. Chaque lane gère cinq candidats.

Le contrat reste celui du routeur existant : clés FP32, NaN au-dessus des
nombres, ±0 égaux, plus grand indice gagnant lors d'une égalité. Les dix
sorties sont ordonnées par clé croissante, avec la même somme sérielle et
le même dtype pour les scores. Les autres formes restent sur leurs chemins
existants. Ce changement ne supprime ni ne remplace un expert sélectionné.

Tests exécutés :96combinaisons,192comparaisons avec tableaux ordinaires
et vues transposées réellement non contiguës sur deux lignes. Oracle de tri
CPU indépendant pour les indices, kernel legacy pour tous les bits des
scores ; cas finis, NaN, infinis, zéros signés, égalités, valeurs prochesFP32
et seeds fixes. Les nonfinis attendus sont distingués des cas exigeant des
sorties finies. Refus de métadonnées et replis sont aussi prévus, puis
mutation, logits/états complets, sessionsMTP et bridge réel avant tout débit.

Inspection des bornes, sans preuve GPU :512threads chargent512valeurs ;
16groupes écrivent160candidats ;32lanes×5 lisent ces160candidats ; les indices
produits restent dans0..512 et chaque ligne écrit exactement dix résultats.
La clé nulle sert de marqueur retiré, strictement sous les clés de toute
valeurFP32 valide, y compris−inf. La barrière précède la fusion des groupes.

Le protocole de vitesse prévu est celui de27, avec un candidat identifié et
une nouvelle baseline batterie issue du dernier moteur retenu. Un succès
isolé du kernel ne suffira pas : deux prompts, OFF/MTP, ordre H-C/C-H,
128tokens, trois répétitions, seuils débit/RAM/swap préenregistrés dansopti.md.

[Patch préparé](measurements/flash-next-ssd-impl/opt-28-router/candidate.patch),
[shader préparé](measurements/flash-next-ssd-impl/opt-28-router/flash_top10.metal),
[identités du stage](measurements/flash-next-ssd-impl/opt-28-router/identity.json).
L’attribution MTPLX/Apache2.0 est étendue dans le prototype ; elle sera
conservée si le kernel est retenu.

## Vérification exécutée

Le kernel passe192comparaisons exactes et les erreurs/replis. Une mutation
remplaçant le plus grand indice gagnant par le plus petit échoue sur
l’oracle CPU. La source restaurée repasse. Clippy signale d’abord
`chunks_exact_to_as_chunks` dans le nouveau test ; ce diagnostic est conservé,
le test corrigé utilise `as_chunks` et Clippy strict repasse.

Commandes inspectées dans `checks.json` : `cargo fmt --all --check`,
`cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`,
`cargo test --locked --no-default-features --features chat` (87passés/2ignorés),
`cargo test --locked --release --features mlx,chat` (107passés/99ignorés),
`cargo build --locked --release --features mlx,chat`. Deux tests routeurGPU
sont sélectionnés explicitement et passent hors de la suite par défaut.
Les tests ignorés de cette suite ne sont pas tous exécutés.

Puis11testsQwen, l’oracle indépendant tenu à part et3testsSSD passent,
en processus séparés. Le vrai bridge passe budgets1/2/3/17, cinq refus de
configuration, deux annulations/reprises et le repli avec pénalité.
Les14processus sont joints ; ils ajoutent au total29,294,592octets de swapout
système (27,94Mio), dont27,328,512octets pendant le bridge197,71s.
Ces valeurs de qualité sont séparées de celles des benchmarks.

Kani est tenté sur le vrai `topk`→MLX avec valeur i8/128 et booléen de
normalisation symboliques, sans `assume`,512valeurs égales et deux couvertures.
Il échoue dans `intrinsics.rs:243` avant obligations : ni sélection ni GPU
ne sont prouvés. `cbmc native/shaders/flash_top10.metal --unwind 11
--unwinding-assertions` est tenté mais CBMC est absent ; le MSL et les
opérations SIMD ne sont pas modélisés. Aucun résultat de preuve revendiqué.

Le superviseur reprend les gardes27 inchangées (égalité AST), avec une
réserve batterie≤5% avant nouveau bras. Huit cas frontières/erreurs de cette
garde passent. CrossHair est absent ; les gardes restent des contrôles testés.
La réserve interrompt la campagne sans convertir une mesure partielle en gain.

Candidat `248a35c3f2e668a4e157b527769b2103ce390d8d93731f6a6f62f5a19c17b2aa`,
tests `02b32302c6817a9ada98fa6832436591a9ee70bdd29103a40129ad8f9b3c703c`.
Sourcefb782d98+patch routeur/shader ; aucun code27 présent. Cache5Gio,
allocateur512Mio, lecteurs4, overlap1, MLX0.32.2, M5/24Gio et checkpoint
65c89531 EXL3 2,05bpw. La vitesse commence à18% de batterie, applications
conservées, après arrêt de toute compilation/preuve/autre inférence.

[Identités exécutées](measurements/flash-next-ssd-impl/opt-28-router/candidate-identity.json),
[contrôles](measurements/flash-next-ssd-impl/opt-28-router/checks.json),
[qualité modèle/bridge](measurements/flash-next-ssd-impl/opt-28-router/quality.json),
[mutation](measurements/flash-next-ssd-impl/opt-28-router/mutation.json),
[gardes](measurements/flash-next-ssd-impl/opt-28-router/supervisor-checks.json).

## Interruption de la campagne batterie

Le Mac passe de batterie18% à secteur18% pendant le premier témoin.
La garde termine la campagne avec zéro bras conforme,15bras non lancés
et aucun candidat mesuré. Les quatre réponses128 sont exactes et le
processus est joint, mais le débit mêle deux alimentations : aucune
comparaison ni gain n’en est déduit. L’utilisateur est sollicité pour
fixer le prochain protocole ; les données restent conservées.

[Interruption](measurements/flash-next-ssd-impl/opt-28-router/power-interruption.json),
[résultat brut](measurements/flash-next-ssd-impl/opt-28-router/speed/result.json).

Clôture le19octobre :28B échoue sur dérive du témoinHOLD, après une seulepaireOFFMAIN −0,474%. Aucun gain niMTP établi ; prototype retiré du runtime. Tous les résultats bruts restent sousac/speed.
