# 80M — préfixe EXL3 immuable, validation puis débit

Le runtime reconstruit un en-tête de 256 positions à chaque préparation de kernel EXL3 de référence. Cette variante l’embarque une seule fois comme chaîne statique. Les 1 006 octets émis restent identiques ; les kernels, leur géométrie et les sommes ne changent pas. Les trois callers utilisent la même fonction CPU. Aucun nouveau flag ou cache de factories n’est ajouté.

## État réel

Code local sur la base publiée `0a27e504` (runtime `89174fcc`), candidat `e65dd887…`, baseline `fe22f100…`. Le gain DFlash2 ci-dessous est répété ; les contrôles OFF/MTP1 sont terminés, sans régression observée. Aucune app installée remplacée. Aucun 15 tok/s annoncé.

## Comparaison DFlash2 B2

M5 10 cœurs GPU, 24 Gio, secteur, Qwen3.8 Flash Next EXL3 2.05bpw `65c89531`, draft Q4 `a22a1744`, ring actif. Cache experts 3 Gio / 53 slots, cache allocateur 64 Mio. GPU Hot et Hot Wide actifs, vérificateur multi-token et cohortes uniques actifs, prélecture/profondeur lookahead/LES/grouped EXL3/shared batch désactivés. B2 fixe : ADAPTIVE=0, WIDTH=1.

Chaque enfant effectue 128 tokens d’échauffement puis 128 mesurés (127 décodés). Ordre HC puis CH, processus et lecteurs joints, observations pré20/post30, mêmes scripts45/49 et limites historiques. Les deux binaires sont immuables et leurs sources sont identifiées dans evidence/.

| Prompt | Ordre | Témoin tok/s | Candidat tok/s | Écart |
|---|---|---:|---:|---:|
| MAIN | HC | 11,359882 | 11,463537 | +0,9125 % |
| MAIN | CH | 11,310677 | 11,429120 | +1,0472 % |
| HOLD | HC | 10,771303 | 10,791026 | +0,1831 % |
| HOLD | CH | 10,743984 | 10,967550 | +2,0809 % |

Rapport géométrique des deux paires par prompt : MAIN +0,9798 %, HOLD +1,1275 %. Deux paires par prompt ne constituent pas une preuve statistique ni une prédiction pour d’autres contextes. Les seize réponses warm/measure ont les hashes et textes golden60 exacts ; 128 sorties / 127 décodés, acceptation, largeurs vérifiées et octets de demandes SSD identiques H/C. Aucun gain des anciennes variantes72/73 n’est inclus.

Pic physique de vie observé : 7 851 546 960 B. Aucune nouvelle sortie de swap système dans ces deux campagnes ; compteur de swap-in propre au processus à 0 B. **Les caps historiques77B de compression courante/pic/vie restent dépassés**, avec des métriques sources valides. Ceci ne prouve pas que la compression est gratuite ou qu’il n’existe aucun swap ailleurs. Les mesures système n’attribuent pas exclusivement une pression à notre moteur.

## Correction et vérification

- Fmt, Clippy strict chat et mlx,chat, builds release CLI et test MLX passent.
- Suite CPU/chat : 116 tests bibliothèque passés, 6 ignorés ; 16 intégrations chat passées. Suite native MLX : 129 passés /146 ignorés initialement ; tests physiques sélectionnés explicitement ensuite.
- Le test CPU compare tous les octets au renderer historique et les 256 positions à un oracle indépendant. Mutation d’une position détectée, source restaurée et test repassé.
- Kani sur la fonction CPU réelle : 27 obligations réussies, deux couvertures satisfaites, booléen symbolique sans hypothèse. Portée : activation, longueur et octets de bord ; ni preuve des kernels Metal ni preuve complète de l’inférence.
- GPU : 144 cas EXL3 (formes, dtypes et strides), 36 commits/préfixes avec logits et états exacts ; bridge corrigé43/43, annulation et reprise incluses.
- Microbenchmark CPU séparé : 50 000 constructions historiques en0,226000667s, accès statiques en0,000017209s. Ce résultat mesure seulement la préparation d’en-tête, pas le débit global.

## Échecs conservés et limites

quality/ conserve l’échec de configuration du premier bridge : WIDTH=1 incompatible avec ADAPTIVE=1, zéro contrôle et enfant joint. bridge-v2/ retire seulement WIDTH et passe43/43. screen/ conserve le premier essai de vitesse interrompu au témoin par un changement d’alimentation, sans paire complète. screen-v2/ et confirm-v2/ sont les nouvelles mesures secteur ; aucun résultat interrompu n’a été requalifié.

Les binaires, versions, options, modèles, SHA source et sorties brutes sont référencés par evidence/*identity.json, les run.json et les rapports. Le Kani global de la CI publiée0a27e504 est en échec ; les jobs natifs Linux/macOS et Flash Kani ciblé passent, ainsi que les régressions. La cause du job global et le run du futur commit80M restent à qualifier ; aucun «CI verte» global annoncé.

## Contrôles OFF/MTP1 terminés

Même protocole secteur/cache3Gio, warm128 puis mesure128 pour chaque mode ; chaque processus alterne OFF→MTP au warm, MTP→OFF à la mesure. MAIN HC et HOLD CH, quatre processus joints avec les seize réponses gold60 exactes.

| Prompt | Mode | Témoin tok/s | Candidat tok/s |
|---|---|---:|---:|
| MAIN | OFF | 10,451473 | 10,715884 |
| MAIN | MTP1 | 10,889616 | 11,106460 |
| HOLD | OFF | 9,965367 | 10,237544 |
| HOLD | MTP1 | 9,679507 | 10,137812 |

Une seule paire par mode/prompt : ces contrôles étayent une absence de régression observée ; ils ne justifient pas de promouvoir leurs écarts comme boosts répétés. Le test MTP réel passe séparément sur budgets1/2/3/17, avec acceptation, rejet, états cible/tête, erreurs et reset. Ses métriques RSS système sont distinctes des statistiques mémoire des bridges. Post30, lecteurs et samplers joints ; zéro nouveau swap-out système et ledger propre0B dans les bridges. Détails dans controls/analysis.json.

Observations système/VM archivées sans perte en .json.gz ; roundtrip SHA-256 vérifié, manifeste evidence/observation-archives.json. Checks et rapports numériques non compressés.
