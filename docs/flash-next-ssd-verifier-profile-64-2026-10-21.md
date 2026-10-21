# Profil du vérificateur Flash SSD — expérience 64P

21 octobre 2026. **Diagnostic terminé, aucun nouveau gain de débit revendiqué.** Base publiée `2e7eefd9`/`ba7068fb`, instrumentation locale `cfg(test)`. Cible Qwen3.8 Flash Next EXL3 2,05 bpw `65c89531`, draft DFlash2 `a22a1744` Q4/ring/adaptatif7, M5 10 cœurs GPU/24 Gio, MLX 0.32.2, secteur et applications ouvertes. Cache **3 Gio/53 slots**, 4 lecteurs, Hot1/Wide1, vérificateur63 activé, lookahead0/LES0 et cache allocateur512 Mio.

## Données et correction

Sept processus isolés et joints, **40 générations, 1 594 141 événements**, 22 543 lignes de composants et 81 018 cohortes exportées. Six processus naturels couvrent OFF/MTP1/DFlash2 × MAIN/HOLD, avec 64 sorties/63 pas décodés : oracle OFF séquentiel, warmup, témoin avant, deux traces et témoin après. Un septième processus DFlash MAIN utilise 17 sorties et un passage avec barrières, distinct du parcours naturel.

Chaque phase compare les IDs non vides, les 248 320 logits finis et tous les états cibles finaux, octet par octet, à l’oracle séquentiel du même processus. Les états de tête MTP sont comparés entre répétitions MTP. Les textes et IDs concordent aussi avec le préfixe externe128. Le champ `mtp` désigne le mode demandé, y compris sur la phase nommée `sequential_oracle`, qui exécute bien OFF et dont les compteurs MTP sont nuls.

- [Clôture, couverture et mémoire](measurements/flash-next-ssd-impl/opt-64-verifier-profile/closure.json), [identité de 146 sources et du binaire](measurements/flash-next-ssd-impl/opt-64-verifier-profile/identity.json).
- [Phases : secondes, octets et tok/s diagnostics](measurements/flash-next-ssd-impl/opt-64-verifier-profile/phases.csv).
- [Composants : wall/self hôte et quantiles en ns](measurements/flash-next-ssd-impl/opt-64-verifier-profile/components.csv), [routes](measurements/flash-next-ssd-impl/opt-64-verifier-profile/routes.csv), [unions des cohortes découpées](measurements/flash-next-ssd-impl/opt-64-verifier-profile/split-unions.csv).

Les sous-dossiers D/M/O/F conservent les plans, commandes, logs, observations système, analyses et événements JSON compressés. Les archives source/binaire sont identifiées sous `ROOT/build/flash-opt-64/profile/`.

## Temps naturels

| Mode | MAIN, traces tok/s | HOLD, traces tok/s | Dérive des témoins MAIN/HOLD |
|---|---:|---:|---:|
|OFF |voir phases.csv |voir phases.csv |2,14 %/0,90 % |
|MTP1 |11,122/11,176 |10,861/10,827 |0,56 %/4,23 % |
|DFlash2 adaptatif |10,251/10,553 |11,146/11,169 |25,21 %/3,29 % |

Le témoin DFlash MAIN évolue de 9,170 à 11,482 tok/s : cette dérive interdit de qualifier un overhead ou un gain. Tous les échantillons sont conservés. Les largeurs choisies par la politique temporelle peuvent varier. Ces parcours courts et ces processus distincts ne remplacent pas la comparaison release128 requise pour qualifier DFlash face à MTP.

Sur les deux traces DFlash MAIN, le MoE hôte inclusif représente 72,3–75,2 ms par sortie. La frontière de lecture des routes coûte 23,6 ms, les appels de lecture 25,9–26,3 ms et la préparation Hot 14,3–14,6 ms de self hôte, incluant soumission et attente. La construction HC représente 2,58–2,67 ms, QSA environ 1 ms et GDN groupé 0,74–0,81 ms. **Les frontières peuvent payer du GPU amont ; lectures et Hot se recouvrent. Ces valeurs ne sont pas des temps GPU exclusifs.** Le draft coûte 12,9–13,2 ms par sortie dans ces traces ; sa chauffe diffère du contrôle final.

Les lectures retournent 11,40–11,61 Go de records sur 63 pas décodés, soit 9 132/9 303 jobs. Ce ne sont pas des nombres de syscalls ni des octets NVMe physiques. Les compteurs et routes sont vérifiés sur les 48 couches.

## Découpage et prochains kernels

Le découpage avant calcul de l’union était déjà identifié dans61R ; le profil le quantifie. **217/240 cohortes scindées MAIN et 127/144 HOLD tenaient dans 53 experts distincts**. Sur deux traces par prompt : 688/768, soit89,58 %. Ce pourcentage concerne les cohortes scindées, pas toutes les vérifications et encore moins un gain prédit. L’essai64U testera une réservation atomique fondée sur la capacité réellement disponible, avec un repli sûr. Aucun slot RAM supplémentaire.

Le [vérificateur Splash](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/Qwen4ExpTarget.cpp) appelle le MoE sur les lignes groupées ; notre limite de53 slots exige une adaptation spécifique. HEAD1482f6bb a été revérifié pendant64P. Les occurrences répétées représentent27–29 % des routes dans les cohortes observées, mais cela ne démontre aucun partage de déquantification GPU ; le prototype58 reste non qualifié.

Le passage avec barrières ralentit DFlash MAIN17 de12,46 à3,52 tok/s. Repères dans ce seul diagnostic : HC completion38,37 ms/sortie, EXL3 trellis30,72 ms, entrée Hadamard/échelles16,82 ms, sortie9,65 ms, scan GDN groupé2,89 ms. Dispatchs, attentes et changements de graphe sont inclus : **ni fractions du décodage naturel, ni temps GPU exclusifs**. HC et EXL3 méritent donc une étude après la réunification64U ; les~89 % en vérification ne sont pas une part GDN.

## Vérification et limites

Rustfmt, Clippy strict et build release `mlx,chat` passent. Suite native :143 passés/143 ignorés recensés ; les tests physiques sont sélectionnés et exécutés séparément. Python :98 tests ciblés passent ; une mutation rétablissant la limite1–2 lignes provoque6 échecs, puis les sources restaurées repassent. Les workflows PR/push couvrent déjà ces contrôles.

Kani a été tenté sur le traceur et le parcours réel avec `--tests` : ICE `intrinsics.rs:243` avant obligations, donc non vérifiés. Les premières commandes sans `--tests` ne sélectionnaient aucun harnais et ne valident rien. CrossHair est absent. Les tests ne constituent pas une preuve complète.

Précontrôle20 s et postcontrôle30 s par campagne ; **0 nouveau Swapout sur l’ensemble64P**,31 178 752 octets de Swapin, pic physique complet8 524 028 816 octets. Aucun autre modèle GPU, build ou proveur pendant les mesures. P0/Xcode reste exclu ; GPU exclusif, fréquence et NVMe physique par modèle restent non mesurés. Aucune installation ni release ;15 tok/s non atteints.
