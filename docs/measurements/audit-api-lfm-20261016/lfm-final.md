# LFM V2 : code conservé, qualité et décisions

État : contrôles numériques et frontières réelles réussis sur les binaires identifiés ;
mesures de vitesse interrompues. Le chemin rapide reste **opt-in, OFF par défaut**.
Aucune publication ni installation n'est annoncée par ce rapport de qualification.

## Modèle et sources

Apple M5, 24 Gio, macOS 26.5, MLX 0.32.2. Modèle local
`LFM2.5-24B-A2B-EXL3-4bpw-v2`, quatre shards vérifiés contre `v2-manifest.json`,
12 614 197 321 octets. Les poids, tokenizer, précision et calibration ne changent
pas. Le manifeste de cette copie annonce 4,21 bpw ; les anciens rapports sur une
autre copie ne servent pas d'identité à cette campagne.

Le candidat conservé est le runtime release `mlx,chat` SHA256
`5f91a2044ca06282ef5b507eaae061a93b5087abf788e7f1692ba099cebdf1bc`, compilé avec
métadonnée `6e9cf753dab3-dirty`. Les empreintes de toutes les sources natives,
FFI et shaders identifient le code modifié, indépendamment de cette métadonnée.
Le test physique figé est
`f6725024a387721e8f42d5671406d8124e8f1d7b99d58682342f84dfb1ae447f`.
Les binaires de release ultérieurs doivent avoir leur propre identité ; ces SHA
ne sont pas ceux d'un DMG publié.

K03 regroupe les seuls FFN MoE par microblocs de huit tokens au plus. Attention,
convolution, normes, résiduels, gates et head restent en M1 canonique. La sélection
exige la configuration qualifiée et les métadonnées des 7 296 projections experts
K4/MCG ; `MLXL3_LFM_MOE_PREFILL=0` ou une valeur explicite invalide choisit la voie
sérielle. Le prototype5f91 sélectionnait le microbloc par défaut. Ce défaut a été retiré
avant livraison, faute de qualification de vitesse complète ; seule la valeur
explicite1 conserve le chemin qualifié. Les preuves de ce prototype restent
distinctes de la vérification du sélecteur final. Les autres géométries conservent
leur sélection existante.

## Qualité numérique avant vitesse

L'ancienne [matrice LFM-01](lfm-01.md) conserve ses 32 cas, T1..8/23/24/25/63/64/65/
128/266 × préfixes0/17 et capture passive128, sur un binaire antérieur identifié.
Le routeur est vérifié contre un tri CPU indépendant sur 1 365 fixtures finies.

Après retrait du prototype KV et raccordement du sélecteur unique, le nouveau
binaire passe **5/5 contrôles affectés** : T8/préfixe0, T266/préfixe17,
capture passive128 et T600/préfixes0/17, en 116,30s. Les **128 000 logits F16
terminaux**, **50 états nommés**, routes/affinités des **38 MoE** puis
**17 continuations forcées** sont bitidentiques, non vides et finis. Les formes et
les compteurs de microblocs sont contrôlés. Ces compteurs désignent des graphes
construits, pas des lancements GPU matériels. Tous les logits de chaque position
intermédiaire ne sont pas relevés ; le maximum configuré131072 n'est pas une
qualification numérique de cette longueur.

Le bridge release par défaut confirme les quatre prompts FR/code à266/600 tokens,
1 732 tokens de préfill et des appels groupés réellement construits, puis sort0
(25,42s). Le serveur HTTP réel passe **13 contrôles** par défaut (4,60s) :
JSON/SSE max_tokens1→length, fermeture après dispatch pendant une requête occupée,
récupération bornée, prochaine réponse égale au contrôle froid, SIGTERM et child
récolté. Ce scénario ne prouve pas l'absence d'un delta modèle avant fermeture ;
la déconnexion avant premier delta est couverte par les fixtures CPU et mutations
API distinctes. Tous les processus de ces campagnes sont terminés et joints.

Un contrôle antérieur sur le vrai CLI release sériel compare l'ancien moteur0beb
au prototype2e15 : huit pas imposés, 1 024 000 logits F16 et les50 états à chaque
pas, bitidentiques. Il ne mesure ni vitesse ni PPL.

## Tests et vérification source

- Source conservée : **118 tests Rust CPU réussis,3 ignorés** =71lib +14contrats
  +33API. Les33API sont inclus, pas à additionner. Sept oracles CPU LFM sélectionnés
  explicitement passent aussi. Format et Clippy strict `chat`/`mlx,chat`, build
  release et compilation du vrai binaire de tests MLX passent.
- Sélecteur : le test final détecte un défautON non qualifié, puis passe après
  restauration de l'opt-in. Les256 valeurs d'un octet sont vérifiées avec/sans qualification,
  ainsi que None, vide, chaînes longues et octets nonUTF8.
- Kani réel du sélecteur final opt-in : **33 obligations/5 couvertures**, deux booléens et quatre
  octets arbitraires, longueurs construites0..4, unwind6, aucune hypothèse. C'est un
  model checking borné du contrat CPU, pas une preuve de l'environnement ou de MLX.
- Propriétés antérieures inchangées : gardes routeur/préfill/géométrie121obligations/
  13couvertures ; cause terminale15/4 ; compteurs API65/3. Le profil mémoire/UI a sa
  campagne indépendante220/11. Les domaines et limites restent dans leurs rapports.
- Suite intégrée antérieure aux derniers changements Rust : **495 Python réussis,
  69 sautés**, puis Desktop complet (dont821assertions/180états de sampling,
  240cas mémoire). Les sources Python/Swift correspondantes sont inchangées ;
  les nouveaux chemins Rust sont recontrôlés séparément ci-dessus.
- Origin symbolique : timeout 30 s ; offsets généraux : timeout 120 s, non prouvés.
  Les limites de preuves du prototype KV retiré restent archivées : budget général
  timeout 120 s et CBMC C++ incompatible avec le parsing libc++ du SDK. Aucun succès
  de test ou de preuve CPU n'est une preuve complète de Metal, HTTP ou concurrence.

## K04 retiré

Le cache KV à capacité passe ses cinq primitives et20cas numériques du modèle sur
le binaire prototype précédent. Les identités après matérialisation montrent de
la réutilisation, sans prouver la donation dans le bridge réel. Ces résultats ne
sont pas une accélération.

Son crible FR266, K03 actif sur les deux variantes, ABBAAB/3paires,
warmup48/mesure48/contexte4096, termine en98,03s avec sortie identique :

| Métrique | KV OFF | KV ON | Écart ON/OFF |
|---|---:|---:|---:|
| Préfill |83,53tokens/s|83,13tokens/s|−0,479%|
| Décodage |53,60tokens/s|53,81tokens/s|+0,391%|
| TTFT |3,1849s|3,2002s|+0,481%|
| Durée totale |4,0618s|4,0814s|+0,483%|

Aucun bénéfice≥3% : **rejeté pour cette release**,1 314 lignes spécifiques retirées,
cache concat conservé, patch/preuves gardés dans l'[archive physique historique](physical-evidence.tar.gz)
et son [index SHA256](physical-evidence-index.json). Les autres formes de performance
KV n'ont pas été mesurées. Aucun gain RAM général n'est déduit de ce crible.

## Mesures de préfill : état et critères

La première tentative48 est interrompue par une erreur du contrôleur après FR266 ;
le driver utilisait bien4096, le contrôleur attendait131072 à tort. La série48
corrigée termine les quatre prompts en632,69s, avec préfill observé+103..107% et
textes identiques. Elle reste **non concluante** : dérive de contrôles courts>3%
et décodage code600 médian−3,421%. Ces données restent distinctes de la suite.

La nouvelle série utilise **le même binaire5f91** forcé sériel/microblocs,
**code266 et FR600**, warmup256 puis mesure128 par prompt/passe,
**ABBAAB puis BAABBA**, trois paires par ordre, contexte4096/cacheOFF/traceOFF,
pauses2s et budget900s par ordre. Qualification limitée aux documents266/600 :
dérives premier/dernier des débits préfill et décodage≤3% pour A/B, médiane des
rapports de paires préfill≥+3%, décodage≥−3%, toutes sorties identiques et contrôle
numérique préalable. Le prompt court27 reste une observation secondaire ; ses
dérives sont rapportées sans lui attribuer un gain qualifié. Aucun ancien essai
n'est requalifié avec ces nouveaux critères. Pas de relance supplémentaire si les
deux ordres restent non concluants.

Une première exécution de ce protocole est invalidée par la veille du Mac :
`pmset` confirme Deep Idle/Sleep et les horloges murale/monotone divergent de
plusieurs dizaines de minutes entre passes FR600. Les données sont conservées,
sans qualifier ce cas. La reprise à paramètres identiques utilise `caffeinate -i`
pendant chaque utilitaire, et un contrôleur testé qui refuse les suspensions ou
discontinuités d'horloge>5s et borne à900s les temps mural et monotone. Cela ne
modifie aucun réglage système permanent et ne prétend pas empêcher une veille
forcée. Dossiers `perf-k03-{abbaab,baabba}-awake`.

La première série protégée est elle aussi interrompue : un jeu et une forte
charge de rendu sont observés en parallèle. Le cas code266 termine, mais les
débits changent fortement et le cas FR600 reste partiel. Le coordinateur est
arrêté par SIGINT ; rapports failed/parity nontrue, modèle et assertion caffeinate
terminés. La confirmation BAABBA n'est pas exécutée. **Aucun gain de vitesse
n'est qualifié globalement ; la release conserve la voie canonique par défaut.**
Un créneau GPU libre est demandé ; aucune relance silencieuse n'a lieu. Les conditions
batterie/thermique/swap et valeurs individuelles restent dans les données brutes.
Température et fréquences physiques ne sont pas mesurées. Les pics lifetime MLX/
footprint incluent le warmup ; aucune réduction de RAM n'est présumée.

## Portée

Aucune nouvelle quantification, calibration, mesure BF16/KL/PPL ou publication HF.
L'égalité numérique vérifie la fidélité au chemin canonique de ces mêmes poids
EXL3 sur les entrées testées. Les fonctionnalités/R&D restantes des40propositions
restent dans la [matrice de l'audit](../../audit-api-lfm-status.md). L'app installée
est conservée selon la décision utilisateur ; un build ou une release ne vaut
pas installation.
