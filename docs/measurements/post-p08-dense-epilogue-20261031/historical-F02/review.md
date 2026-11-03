# 82F02 — conception d’un épilogue S2 autonome, sans prototype

25 octobre2026. Revue autorisée pendant82F01. JournalPERF88,82T,81I et [complément81R/F02](https://github.com/0xZKnw/mlxl3/issues/45#issuecomment-6382947448) lus. **Aucun code moteur/prototype, W/index/opti, build, test, proveur, GPU ou benchmark modifié/lancé.** Cette note recommande une prochaine expérience ; elle ne valide aucun gain et n’autorise pas son exécution sans amendement.

## Domaine réellement présent

Inventaire en lecture des seuls headers safetensors, sans charger ou évaluer les poids :48 projections cible,36 `linear_attn.out_proj` et12 `self_attn.o_proj`, trellis `[384,160,64]` partout. Donc entrée6144, sortie2560, **K4**, pas K2 ; toutes ont SVF16 et aucun biais. Détails et empreintes des headers dans `inventory.json` ; les payloads de poids n’ont pas été rehashés.

Pour `reference_splits(384,160)`, le premier passage donne2 car160<8192 et384/2=192≥128 ; le suivant s’arrête car384/4=96<128. SG4/NT1 et les deux splits QMV restent inchangés. Le chemin mono-ligne produit `[2,2560]`, puis `sum(axis0)`. Le batch de référence63 produit `[B,2,2560]`, puis `sum(axis1)`. Il faut traiter les deux sites, pas seulement B1. En vérificationB2, GDN regroupe ses deux lignes ; QSA garde ses appels causaux mono-ligne. La tête vocabulaire, les experts routés et MTPfc_hidden ne sont pas ce domaine.

La chaîne publiée après QMV est : sommeFP32 → castF16 → H128F16 avec scaleFP32 → multiplication par SVF16 → cast au dtype de l’entrée → reshape/biais éventuel. F02 ne doit changer ni QMV, ni son scratch, SG/NT/FMA, ni l’activation/gate de GDN/QSA.

## Somme exacte des deux splits

Sources primairesMLX tag0.32.2, commit `1f8e74e3f12f31365464a6867c6579f0e9b29d85`, copies dans `upstream/`. Les quatre headers installés `reduce_col.h`, `ops.h`, `hadamard.h`, `metal_kernel.h` sont byte identiques au tag. Le header de version installé annonce0.32.2. Les fichiers hôte `reduce.cpp` et `hadamard.cpp` viennent du tag : leur choix de dispatch est une déduction de ces sources et des formes, **pas une trace du binaire exécuté**.

Pour ces Arrays contigus et une seule dimension réduite de taille2/stride2560 : plan `ContiguousStridedReduce`, `non_col_reductions=1`, puis `col_reduce_small` puisque2<32. `n_reads=4`, `threadgroup_x=32`, `threadgroup_y=2` (64threads, supporté sur ce GPU),20groupes par ligne. Aucun SIMDsum ne participe à la réunion des deux splits.

**L’ordre ne se résume pas à p0+p1.** Dans `reduce_col.h`, les deux lanesY initialisent chacune `totals=Sum<float>::init=+0F32`. LaneY0 calcule `op(p0,+0)`, laneY1 `op(p1,+0)`, puis laneY0 réunit `op(shared_lane1,total_lane0)`. Comme `Sum::operator(a,b)` retourne a+b, le graphe source est :

1. arrondir enFP32 `p0 + (+0)` ;
2. arrondir enFP32 `p1 + (+0)` ;
3. arrondir enFP32 `(résultat1) + (résultat0)` ;
4. convertir le résultat enF16 avant tout H128.

Conserver les deux additions à+0 et cet ordre d’opérandes. Deux partiels−0 ne doivent pas devenir une simple addition−0+−0 ; le pipeline source commence par les neutraliser avec+0. L’ordre peut aussi affecter les payloadsNaN. Les annulations, débordements, sous-normaux et zéros signés doivent être contrôlés contre le vrai `Array::sum`, pas uniquement une somme CPU ou une égalité tolérante. Le compilateur reste une limite : les options du customkernel utilisent `CompileOptions{}`/MathModeSafe, mais la présence textuelle d’une addition n’est pas une preuve physique de son maintien. Des temporaires explicites/volatile peuvent conserver les frontières de stockage ; leur effet doit être vérifié.

## H128 installé :16 puis8, pas un arrondi F16 à chaque butterfly

Le dispatcher décompose128 en m=1,n1=1,n2=128. Il instancie `hadamard_n<half,128,16,4>` : **8threads par groupe**, tableau partagé128half (256octets),16float temporaires par thread.

- Lecture des128 entrées déjà converties enhalf dans le scratch partagé.
- Un radix16 : quatre étages de sommes/différences enFP32, puis stockagehalf (première frontière d’arrondi interne).
- Un radix8 : trois étages de sommes/différences enFP32 à partir du scratchhalf, puis stockagehalf (seconde frontière).
- Multiplication de ce half par le scaleFP32, puis stockagehalf.
- Dans l’opération MLX suivante, multiplication de ce half par SVhalf et stockagehalf.

Le scale est la valeurFP32 produite par l’expression Rust existante `1.0/128.0f32.sqrt()` ; réutiliser ce scalaire, sans précombiner scale×SV. Le futur test devra aussi vérifier ses bits ; aucune nouvelle exécution Rust n’a été faite ici. Les trois barrières du H128 natif (chargement, après radix16, après radix8) restent uniformes. L’orientation de la moitié haute est low−high, pas high−low.

Le vieux corps `MLXL3_FUSE_OUTPUT` contient déjà un équivalent7étages avec conversionhalf après l’étage3 et avant la mise à l’échelle ; **il est désactivé dans les QMV actuels**. Son organisation128threads/deux tableauxfloat/14barrières n’est pas le premier choix pour F02. Réutiliser le schéma natif8threads évite d’ajouter cette coordination et ne modifie pas les QMV.

## Plus petit patch futur recommandé

Un helper d’épilogue dans `linear.rs` appelé aux deux sites juste après la production des partiels, avant la branche `sum`. Un shader autonome, une sortiehalf,20blocs128 par ligne ;8threads par bloc, comme le H128 installé. Les entrées sont les partielsFP32 originaux et SVhalf, sans repacking. Une vue `[B,2,2560]` peut unifier B1/B2, sans nouveau payload.

Remplacer uniquement le chargement initial du H128 : chaque valeur du bloc lit ses deux partiels, reproduit les trois additionsFP32 ci-dessus, puis écrit leur conversionhalf dans le scratch. Conserver les codelets radix16/radix8 et leurs stockageshalf. À la sortie, forcer le cast après la mise à l’échelle avant de multiplier SVhalf, puis écrirehalf. L’ancien cast au dtype d’entrée, reshape et biais restent à leur emplacement. Les références de partiels/SV restent possédées par le grapheMLX jusqu’à évaluation ; pas de nouvelle synchronisation ou de libération manuelle.

Flag strict chargé une seule fois, défaut0. Premier périmètre : checkpointFlash cible, référenceSG4, K4,6144→2560, splits2, B1/B2, entréesF16/F32, SVhalf et formes exactes. B3..8, K7, MTP, dense nonFlash, mappedexperts et toutes formes inhabituelles restent pipeline historique. Ne pas modifier `MLXL3_FUSE_OUTPUT`, NT, MB, SG, les deux QMV ou leur ordre de réduction. Aucune combinaisonF01/K1 présumée.

Un contrat CPU futur doit être le vrai prédicat d’éligibilité appelé par les deux sites, pas un miroir mathématique du shader. Kani sur dimensions/tailles/types symboliques et bornes/offsets réelles ; cela ne prouvera ni H128 niMetal. Une tentativeMLX réelle peut rester non vérifiée en cas d’ICE/FFI ; CBMC ne sait pas prouver MSL par simple changement de langage.

## Qualification discriminante avant débit

1. Composant indépendant à partir de **partiels identiques**, ancien `sum→cast→H128→SV` contre futur épilogue. Vérifier séparément sommeFP32, conversionhalf, après radix16, après radix8, sortie mise à l’échelle et sortieSV lorsque le diagnostic est nécessaire. Toutes formes/cardinalités/dtypes/bits, sans comparer deux tableaux vides.
2. B1/B2, layouts des deux sites, SV±0/±1/valeurshalf, partiels±0, annulation exacte, near-ULP et seuils d’arrondihalf, petitsFP32, grandsfinis, ±Inf/NaN intentionnels avec payload. La finitude reste obligatoire pour les cas de génération finis ; ne pas l’attribuer aux fixtures non finies intentionnelles. Le cas deux−0 doit détecter le mutant « somme directe p0+p1 » ; mutation de la frontièrehalf/radix16 et d’un offset de ligne doit également être détectée.
3.48 vraies projections et activations capturées, chemin entier mono/batch contre chaîne publiée, fastpath prouvé non vide. Refus/fallbacks et reprise valide ; flags via enfants à environnement fixé, aucune mutation env après loadMLX. Puis36préfixes DFlash/logits/captures/états, MTP/cache-only/session et bridge43 réel.
4. Si exacte : mesurer F02 isolément, même binaire0/1, B2 MAIN/HOLD HC puisCH, cache3Gio,128warm+128mesure/127decode, Hot/Wide/U/63, autres flags constants ; OFF/MTP1 frais avant promotion. Gardes45/49 et82B, identités gelées, unGPU et aucun build/proveur concurrent. Les caps compression77B échouées restent dans leur statut.

## Valeur attendue et limites

Le tableauFP32 de la somme vaut10240octets logiques par ligne/projection,491520 cumulés sur48 enB1 et983040 enB2. Le futur épilogue peut aussi éviter des intermédiaireshalf et passer de quatre primitives somme/cast/H128/mul à une, mais MLX peut déjà réutiliser/donner des buffers. Ces comptes ne sont ni un picRAM récupérable établi, ni des octetsDRAM/NVMe mesurés. Les deux tableaux de partiels restent nécessaires ; aucune lectureSSD d’expert n’est supprimée.

PERF88 avait essayé un épilogue séparé sur head≥65536, B6/B8, splits1. Correct numériquement, débit non concluant sous dérive puis retiré. F02 est un **nouveau mécanisme S2 sur2560/B1–2**, pas une reprise de NT8 ou de cette ancienne géométrie. Il n’existe pas de blocage statique identifié ; le vrai risque est que les chargements et additions supplémentaires ralentissent le petit H1288threads autant que les soumissions économisées. **Recommandation : composant S2 exact d’abord**, puis seulement campagne complète si ce composant passe. Gain, tempsGPU, RAM et tok/s non mesurés.
