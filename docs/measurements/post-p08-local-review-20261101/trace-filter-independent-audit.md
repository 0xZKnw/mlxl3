# Contre-vérification du filtre P08r3

1 novembre 2026. **Chiffres reproduits ; aucun gain ni temps causal établi.**
Relecture CPU des exports existants, sans réexport, modèle, GPU ou modification
des sources surveillées. Lecteur ElementTree distinct du parseur AST initial ;
résolution des références XML avec contrôle de type, colonnes issues du schema,
union/intersection et affectation des lectures recalculées indépendamment.
[Reçu détaillé](trace-filter-independent-audit.json).

## Vérifications concordantes

- SHA des cinq sources du [filtre initial](trace-filter.json) conformes.
  PID 27154, thread des `fstat` 291767 ; fenêtre XML
  `[10263744750,14446565041]` ns. Les événements natifs recalés sélectionnent
  exactement les pas 76–115, **40 sorties**.
- **5985 `preadv`, 7469351820 octets, 1696 lots** ; chaque lecture appartient
  exactement une fois à un lot non chevauchant, même nombre de jobs et
  1248012 octets par job. Les retours sélectionnés indiquent succès.
- Intervalle de correction natif→XML reconstruit par inclusion des lectures :
  **−28364 à −25614 ns**, milieu **−26989 ns**, demi-largeur **1375 ns**.
  Les spans natifs corrigés coïncident avec les 1696 lots du JSON. Ce contrôle
  vérifie le modèle de décalage constant existant, pas une nouvelle horloge.
- Frontière GPU fixe recalculée depuis les intervalles XML `Active` du moteur,
  avec le même traitement des lots dont Hot n’est pas prêt au retour.

| `fstat` dans `fill_many` | Nombre | Somme wall ns | Wall ms/sortie | Wall sans GPU Active ms/sortie |
|---|---:|---:|---:|---:|
| `TensorInfo::source` |5985|3722801|0,093070025|0,002815700|
| `Array::read_records` |5985|1323340|0,033083500|0,002339750|
| Total |11970|5046141|**0,126153525**|**0,005155450**|

Les unités proviennent des entiers XML en ns, divisés par `40×10^6` pour les
ms/sortie ; les chaînes `fmt` ne servent pas aux calculs. L’intersection
ci-dessus exclut **toute** activité GPU observée, pas seulement le moteur.
Les 40 autres `fstat` viennent de l’embedding et sont exclus de ce total.

Cold premier retour→dernier retour : **633920443 ns**, soit
**15,848011075 ms/sortie**. Partie après la frontière GPU fixe :
**621305929 ns**, soit **15,532648225 ms/sortie**. Ces deux résultats
reproduisent exactement le filtre initial ; 1381 lots ont plusieurs jobs.

## Interprétation et pièges

1. Supprimer les deux `fstat` par expert est une **faible priorité dans cette
   fenêtre** : seulement 0,126 ms de syscall wall au total, largement recouvert,
   et 5,155 µs/sortie sans GPU Active. Ce dernier nombre est un proxy temporel,
   pas une mesure causale du gain possible. Cela ne chiffre pas les allocations,
   validations Rust/FFI ni tout le coût de préparation des records.
2. La colonne `cputime` excède `duration` dans **8 appels de fill**, quatre
   dans chaque catégorie. Les sommes CPU correspondantes sont 0,09419295 et
   0,03561995 ms/sortie. Ne pas former une partition wall=CPU+wait avec ces
   champs, ni interpréter leur excès comme un coût moteur mesuré précisément.
3. Les 15,53 ms Cold sont une **avance possible du premier job fini**, pas celle
   d’une cohorte utile déjà publiée. Il reste à choisir un groupe de slots
   entièrement prêt et à payer publication, encodage/calcul, contention et
   fusion canonique. Une nouvelle soumission peut aussi déplacer la frontière
   GPU employée dans ce calcul ; ce n’est donc ni un gain ni un plafond causal.
4. Les XML partiels, le modèle de calage, l’observateur et l’absence de contrôle
   final P08 restent des limites. Aucune nouvelle analyse de sensibilité aux
   extrémités du calage n’est revendiquée ici.

Le premier filtre indépendant avait omis d’exiger la présence de `fill_many`
avant de classer `TensorInfo::source` : il y ajoutait les 40 appels d’embedding.
Cette erreur de mon audit, **pas du filtre initial**, est conservée dans le
[reçu r1](trace-filter-independent-audit-r1.json). Le prédicat corrigé reproduit
les quatre catégories, durées, intersections et compteurs CPU initiaux.
