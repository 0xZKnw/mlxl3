# F02 v1 — audit d’adaptation avant modèle

31 octobre 2026. Lecture seule des sources finales et du micro ; aucun nouveau
build, GPU, proveur ou changement du candidat pendant cet audit. CLI `6f35338c…`,
libtest positive `e5e46494…`, shader restauré `ac5dfbdc…` ; identités complètes
dans [build-identity.json](build-identity.json).

**Aucun défaut concret nécessitant un correctif immédiat identifié.** Les
captures, la couverture des projections et la qualité modèle prévues restent
nécessaires avant un pilote de débit. Le micro ne valide aucun gain modèle.

- `linear.rs:238–249` lit le flag strict seulement dans `from_checkpoint` et
  le stocke dans le Linear. Aucun accès à l’environnement dans `forward`, le
  helper de sélection ou le kernel. `new()` garde false ; les autres modèles
  gardent leur chemin.
- `linear.rs:382–410,572–574,699–701` garde référence/K4/MUL1/6144→2560/SG4/
  splits2/B1–2/F16–F32/sans biais. Les deux sites interviennent avant la somme
  stock. La sortie Half du helper reprend exactement le cast vers le dtype
  d’entrée et la forme de l’appelant. Les refus reviennent au stock ; aucun
  biais n’est sauté par le retour anticipé.
- `dense_epilogue.rs:70–101` ne construit aucun readback, eval ou synchronize.
  Il passe les partiels et SV existants à MLX, avec une nouvelle sortie Half.
  Le graphe MLX possède ces entrées ; aucun pointeur manuel ni banque SSD
  nouvelle. La capacité du cache d’experts reste inchangée.
- `flash_dense_epilogue.metal:20–91` conserve `(p1+0)+(p0+0)`, le cast Half,
  radix16→Half, radix8→Half, scale F32→Half et SV Half→Half. Trois barrières
  uniformes pour huit threads ; aucun échange intergroupe. Le B2 indexe les
  plans avec `row*5120`, la sortie avec `row*2560`. Le mutant qui supprimait
  le premier offset a été détecté physiquement sur B2, demi-mot2561 ; ce
  négatif ne remplace pas la qualification positive.

La lecture MLX0.32.2 confirme pour B1 et B2 contigus `non_col_reductions=1`,
réduction de taille2/stride2560, donc le même ordre d’addition. L’axe batch
extérieur change les offsets et la grille, pas les deux sommes. Les contrôles
finis passés sont byte-exact ; les nonfinis intentionnels vérifient classes,
signes et valeurs finies, sans contrat sur les payloads NaN.

Coûts résiduels et risques : un `CustomKernel` alloue sa sortie et peut copier
un input noncontigu ; les partiels de production et SV chargés sont contigus.
Le H128 stock peut donner un buffer : quatre primitives remplacées ne signifient
pas quatre allocations économisées. La création du custom garde ses wrappers,
chaînes/source hash, lookup de pipeline et bindings MLX. Les deux temporaires
`volatile` peuvent coûter des instructions/stack ; leur coût et les spills ne
sont pas mesurés ici. Le flag ON est stocké aussi sur les Linears inéligibles,
qui font alors quelques tests scalaires avant fallback ; un préfiltre au
chargement serait possible, sans preuve que ce coût soit significatif. Aucun
de ces points n’établit une cause de ralentissement ou ne justifie une édition
pendant la mesure actuelle.

[Micro brut résumé](light-v1/micro-summary.json), 12 rounds alternés par cellule :

| Projection | B | OFF µs | ON µs | Débit local agrégé |
|---|---:|---:|---:|---:|
| GDN0 out |1|312,60|283,46|+10,28 %|
| GDN0 out |2|386,81|382,95|+1,01 %|
| QSA15 out |1|282,65|283,25|−0,21 %|
| QSA15 out |2|325,69|327,14|−0,44 %|

Le premier OFF GDN B1 vaut507,12µs ; il reste inclus et empêche de traiter le
+10,28 % comme gain stable. Chaque échantillon exécute32 `forward+eval`
séquentiels, pas un graphe naturel de48couches. La comparaison reste utile pour
détecter un coût composant majeur ; elle ne tranche pas l’économie CPU exposée
du modèle complet. Poursuivre les captures et la qualité complète prévue,
puis un pilote modèle court est cohérent. Aucun gain général, retrait ou
activation n’est décidé par cet audit.
