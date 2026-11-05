# Issue 140 — AR1 : préparations et finalisations des cohortes Cold Ready

Les trois adaptations A/B/C et la répétition C-R1 sont closes, numériquement validées, **sans gain de débit établi sur MAIN128**. La répétition demandée explicitement par l’utilisateur place C entre ses deux contrôles frais, avec une dérive de +8,96 %. Aucun défaut de calcul ou de sélection supplémentaire n’a été identifié lors des revues finales.

Décision : **NON CONCLUANT ; runtime expérimental retiré, publication des résultats seuls**. Les sources restent dans les archives passives ; aucune option AR1 ni activation par défaut n’est retenue dans le moteur livré. Ce rapport conserve aussi les essais négatifs et les corrections d’adaptation. Il ne qualifie pas une app installée ni une release.

## Composition comparée et protocole

Base exacte : `10397e9899eb5ec6a19e6b6f6af179500c6d3770`, avec les treize défauts AR10 et T05 activés ; AR20/AR30/AR33 expérimentaux désactivés. Les variantes sont des états locaux modifiés de cette base, identifiés par les manifestes de sources et de binaires ci-dessous. Elles ne constituent pas une mesure de `main` ultérieur `6030540b5573185a4bf8b13207011872d94d0045`, qui a intégré PR141, PR139 et Desktop#74. Une publication de ce rapport contre ce nouveau `main` ne réattribue pas les mesures à cette composition.

Matériel et modèle : Apple M5, 24 Gio, macOS 27.2, deployment target 26.2 ; `Qwen3.8-Flash-Next-exl3-205-65c89531`, EXL3 moyenne 2,05 bits, H4/MTP2 dans le checkpoint. **Génération AR sans MTP ni DFlash.** MAIN contient 87 tokens de prompt ; chaque bras utilise un warmup puis une mesure, 128 tokens générés et 127 tokens de decode, contexte 4096, cache experts 3 Gio, quatre lecteurs, lecteurs dynamiques désactivés, profondeur de lookahead 0, cache de prompt 0, température 0/top-k 1/pénalité 1.

Les sources, exécutables et helpers sont figés avant chaque QA. La correction est vérifiée avant la vitesse ; un seul modèle GPU est lancé à la fois, sans compilation, Kani ou autre inférence pendant les bras. Brave et Deezer restent ouverts. Aucun refroidissement imposé, arrêt sur swap ou attribution causale à une application. La température et les temps GPU par kernel ne sont pas mesurés par ce protocole.

Les sorties warmup et mesurées sont complètes, non vides et comparées au texte et au hash du golden128 indépendant : `82cce9dd104da4a46919d50fecc3306c7a4cfce959b02cf4402802443b847794`. Tous les enfants des campagnes A/B/C sont joints. Le statut `validated_speed` des rapports signifie que la mesure et sa sortie sont valides, pas qu’un boost est établi.

## Adaptations successives

| Variante | READY_STAGES | READY_BATCH_BINDINGS | READY_SPLIT_FINISH | Changement |
|---|---:|---:|---:|---|
| OFF | 0 | 0 | 0 | Chemin stock T05 + AR10 |
| A | 1 | 0 | 0 | Cinq stages vectorisés, finish dynamique commun |
| B | 1 | 1 | 0 | Même GPU ; cinq appels natifs `set_buffers` |
| C | 1 | 1 | 1 | Même largeur dynamique ; PSO plain et SwiGLU séparés |

Les noms complets sont préfixés par `MLXL3_NATIVE_`. Dans les prototypes, les trois options étaient expérimentales, strictes 0/1, OFF par défaut et qualifiées sur l’Apple M5 exact. Leur présence dans cette matrice n’annonce pas leur intégration au runtime conservé.

**A — vectorisation.** Les trois QMV d’A06 étaient déjà vectorisés. AR1 regroupe seulement les cinq autres stages des cohortes déjà retournées N2..10 : GU prepare, gate finish, up finish avec SwiGLU, down prepare et down finish. Le nombre de dispatches passe de `5N+3` à 8 ; N1 conserve le fallback. Aucun délai pour grossir une cohorte, déplacement de frontière SSD/wave ou réassociation des sommes. T05 traite déjà les hits et reste distinct des compteurs AR1.

**B — correction des liaisons.** La revue source d’A a relevé 90 liaisons de buffers et cinq de paramètres fixes pour ces cinq stages, contre `25N` liaisons buffers/paramètres dans les cinq stages stock. Cette comparaison est un comptage source, pas un temps CPU mesuré. B remplace les 90 appels individuels `set_buffer` par cinq `ComputeCommandEncoderRef::set_buffers`, avec deux tableaux sur pile de 22 éléments maximum, sans ajouter de `Vec`. Les constructions de bindings déjà présentes dans A restent identiques. Les trois QMV, shaders, slots et offsets ne changent pas. Le helper réel valide les préfixes 12/22 et la longueur des deux tableaux avant de créer l’encodeur ; le paramètre reste au slot 12/22. L’API compute Metal 0.33 employée est celle d’`encoder.rs:1578`, avec `Option<&BufferRef>` ; celle de la ligne 1924 appartient à l’argument encoder.

**C — correction de la séparation des PSO.** Le stock utilisait déjà une largeur dynamique dans `flash_finish128` et `flash_up_finish_swiglu` : il n’existait pas de finish spécialisé 640/2560 à restaurer. Il séparait cependant les fonctions avec et sans SwiGLU, alors qu’A/B utilisaient un `if(p.z)` dans un PSO commun. C restaure uniquement cette séparation avec deux PSO distincts, aux mêmes bodies, largeur paramétrée, ABI, barrières et arrondis. B reste sélectionnable sans cette option. La présence de plus de code ou de registres dans le PSO dynamique est une hypothèse ; ni leur quantité ni leur coût n’ont été mesurés.

## Résultats MAIN128

Unités : tokens de decode par seconde. Les rapports liés conservent aussi TTFT, temps total, allocation Metal, commandes, environnement et sorties complètes. Chaque ligne représente sa propre campagne ; leurs niveaux de débit ne doivent pas être mélangés.

| Campagne | OFF début | Variante(s), dans l’ordre | OFF fin | Dérive des contrôles | Conclusion |
|---|---:|---|---:|---:|---|
| [A](speed128/report.json) | 9,232696241549288 | A : 9,213011778568672 | 9,1956779444778 | −0,400948 % | A entre les contrôles ; aucun gain établi |
| [B](b/speed128/report.json) | 8,735346513742357 | A : 8,473950544635693 ; B : 8,667048589102773 | 8,492074079894635 | −2,784920 % | B au-dessus d’A dans cette séquence, entre les contrôles ; aucun gain de baseline établi |
| [C](c/speed128/report.json) | 9,111739285466331 | B : 8,19004122228186 ; C : 7,821095418505616 | 8,226646085982807 | −9,713768 % | C sous les deux contrôles ; forte dérive, régression causale non isolée |
| [C-R1](c-retest01/speed128/report.json) | 8,155587612616042 | C : 8,75261214697099 | 8,886223876917963 | +8,958720 % | C entre les contrôles frais ; aucun gain établi |

C-R1 réutilise la QA C et le même binaire gelé ; il ne crée aucun nouveau prototype. La propriété demandée est de vérifier si le niveau bas de C se reproduit face à des contrôles frais. Le bras B est retiré de cette répétition. Les six sorties warmup/mesure sont exactes, les 288 pins et exécutables inchangés, tous les enfants joints. Le résultat ne reproduit pas une baisse sous les deux contrôles, mais sa dérive positive ne permet pas de conclure à un boost. Les résultats précédents restent conservés ; ni gain de baseline ni régression causale isolée n’est établi.

## Correction numérique et parcours réels

Chaque variante a passé trois gates physiques : composant, modèle et vrai bridge/CLI. Rapports : [A](quality/report.json), [B](b/quality/report.json), [C](c/quality/report.json).

- **Composant :** 14 cas N1/2/10, F16/F32 avec vues stridées, experts distincts/permutés/dupliqués. Sept intermédiaires réels, soit **2 119 680 octets**, sont comparés bit à bit aux anciens PSO inchangés ; **189 440 mots Half** de sortie sont non vides et finis. Le chemin réutilise les trois QMV canoniques ; il ne prétend pas disposer de deux implémentations QMV indépendantes. L’ancrage aux fixtures stock existantes reste vérifié.
- **Frontières et erreurs :** slabs empoisonnés, préfixes/suffixes conservés, offsets négatifs de la vue d’entrée, N1 fallback, destinations superposées ou alias de l’entrée, dépassement de destination et erreur tardive avant commit. Les mutants de bit d’intermédiaire et de mauvais record sont détectés. Ces injections Rust/shader ne sont pas revendiquées comme des fautes matérielles du driver.
- **Activité réelle :** le composant qualifie 12 cohortes vectorisées, 72 routes et 96 dispatches, avec `valid_counts=[12,12,12,12,12,12,72,96]`. B/C imposent `valid_bulk_counts=[60,1080,60]`, soit `[5,90,5]` par cohorte. C vérifie les objets PSO distincts et `valid_finish_counts=[0,24,12]`, soit deux finish plain et un finish SwiGLU par cohorte ; la variante dynamique compte `[36,0,0]`.
- **Modèle :** oracle de processus indépendant `171bea949c7c16f20c1e1affa26b70602676c73620fcbb380d430a7587835351`, 84 états initiaux, **57 606 144 mots d’état**, **993 280 logits** et **81 920 octets hidden**, avec continuation et rollback. A exerce dès le passage froid 48 cohortes/432 experts/384 dispatches, puis 176 cohortes/1 092 experts/1 408 dispatches au total. Les gates B/C ajoutent les deltas bulk/finish correspondants sur le vrai caller, séparément des soumissions H/T05.
- **CLI :** deux tokens golden, moteur direct sans bibliothèque MLX chargée, arrêt et enfants joints. Les bras de vitesse vérifient ensuite les sorties MAIN128 complètes. La CLI de vitesse n’exporte pas les compteurs AR1 : la distribution des tailles de cohortes pendant ces 128 tokens n’est donc pas une métrique mesurée.

Les revues ont relié les flags aux macros/PSO, les offsets GU/up/down aux layouts canoniques, et les bindings aux destinations du caller. Les mêmes clés NT2, grilles et PSO QMV sont conservées. Le CB garde ses références normales jusqu’à completion ; aucun wait/commit supplémentaire n’est ajouté. Aucun défaut concret supplémentaire n’a été trouvé. Restent des coûts source — validation des records, allocation/croissance des `Vec` de bindings, sélection des pointeurs par route et bindings paddés à dix — dont l’importance temporelle n’est pas établie. Ils ne justifient pas à eux seuls une nouvelle mesure ou une explication causale du résultat.

## CPU, Kani et limites

| Variante | Suite CPU | Contrôleur | Kani ciblé |
|---|---|---|---|
| A | 403 PASS, 98 ignorés | 15 PASS | 1 harnais, 116 obligations, 6 covers |
| B | 404 PASS, 98 ignorés | 17 PASS | 2 harnais, 158 obligations, 10 covers |
| C | 404 PASS, 98 ignorés | 19 PASS | 2 harnais réutilisés, 158 obligations, 10 covers |

Les checks locaux incluent Clippy strict `direct-metal,chat` et `mlx,chat`, builds release CLI/libtest, format Rust/Python, inventaire et compilation Metal des options concernées. Les commandes exactes et statuts par processus restent dans [A/cpu/checks.json](cpu/checks.json), [B/cpu/checks.json](b/cpu/checks.json) et [C/cpu/checks.json](c/cpu/checks.json). Les tests physiques ignorés dans la suite CPU ont été sélectionnés séparément ; les autres tests ignorés ne deviennent pas exécutés par implication.

Kani vérifie les fonctions de production de `ready_policy`, avec entrées `usize/u64` symboliques, sans `assume` ni boucle de harnais : admissibilité N2..10, tailles/prefixes, destinations et disjonction de plages ; B ajoute les préfixes de bindings 12/22, leurs capacités égales 22 et le slot de paramètre. Les sorties et covers sont conservées dans [A](cpu/kani-final.log), [B](b/cpu/kani.log) et [C](c/cpu/kani.log). C ne crée pas un nouveau sélecteur CPU à prouver : ses indices PSO sont 0/1 et ses rôles sont vérifiés physiquement.

Il s’agit de model checking des helpers CPU dans les domaines indiqués, pas d’une preuve de Metal, des pilotes, du compilateur shader, de l’allocateur ou de la concurrence SSD. CrossHair était absent de la venv : diagnostic conservé, pas de preuve Python revendiquée. Les nouveaux contrôleurs sont couverts par leurs tests déterministes, y compris refus d’un compteur nul/incomplet et nettoyage. Les CI hébergées du futur commit de publication ne sont pas déclarées vertes à partir des seuls checks locaux. Aucun gain statistique, mesure exclusive de kernel, registre GPU ou cause de chauffe n’est établi.

## Identités, archives et échecs conservés

| Variante | Pins source | SHA-256 CLI | SHA-256 libtest |
|---|---:|---|---|
| [A/freeze](freeze.json) | 282 | `f7b2838cbf6c9be527fde0d6e7ac820fa09a121fde20ddc3c5117d6182f2034c` | `14d63c62dc1bb107b3edbf66bedc55a9d3f18c6c95cf68610a0519fee7bcf92e` |
| [B/freeze](b/freeze.json) | 285 | `f35cb7453e16b5c487c4d98fa9f9049d7017ec4cf43ff7b60eddd894104633be` | `e137c00dced81de478a5ae50ae5c79b11cd903036c848d5a4e9502006bb3d922` |
| [C/freeze](c/freeze.json) | 288 | `d2d0774bba3bc9e6ee1f7d0ade83e32fe5d08f8a1afe345db73e4b4a9fa005f5` | `ebf8b6e524b629b344886a69b2c7db06c141dd4ae99164f6cf8d844aa3642017` |

A a été archivé **avant** les changements B : [sources](source-a.tar.gz), [index SHA-256 des 282 pins](source-a-index.json), [43 preuves brutes](raw-a.tar.gz), [index des bruts](raw-a-index.json), [reçu de roundtrip](evaluation-a.json). B a été archivé **avant** C : [sources](b/source-b.tar.gz), [index SHA-256 des 285 pins](b/source-b-index.json), [35 preuves brutes](b/raw-b.tar.gz), [index des bruts](b/raw-b-index.json), [reçu de roundtrip](b/evaluation-b.json). Après fermeture de C-R1 et avant retrait du code, C a été archivé avec sa répétition : [sources](c/source-c.tar.gz), [index SHA-256 des 288 pins](c/source-c-index.json), [48 preuves brutes](c/raw-c.tar.gz), [index des bruts](c/raw-c-index.json), [reçu de roundtrip](c/evaluation-c.json). Les roundtrips sont marqués vérifiés dans ces reçus ; aucune archive n’a été recalculée par la rédaction du rapport pendant un bras GPU. Les chemins absolus locaux des exécutables figurent dans les gels ; leur présence locale future n’est pas garantie par une PR de rapports.

Les échecs restent visibles : trois diagnostics Clippy d’A avant correction de style, erreur du Python système 3.9 avant assertions (`pytest` absent), inventaires corrigés et préflight du premier freezer contenant un ancien littéral `build/ar34`. Le [rapport QA refusé](quality-preflight-failed01/report.json) garde `parity:false`, zéro phase et tous enfants joints ; [son diagnostic](cpu/freeze-preflight-failed01.json) situe la correction de chemin, sans modification native. Ils ne sont pas effacés par les relances réussies. Les limites et conclusions négatives A/B/C sont conservées dans `opti.md` et les rapports liés ; C-R1 n’en remplace aucun.
