# Contre-revue qualité F05 — 2026-10-25T11:12:23.421925+00:00

Les 19 étapes terminées sont toutes conformes à leur résultat attendu, avec enfants et sampler joints. Aucun nouveau défaut numérique n’est révélé dans le domaine exercé. **La qualité ne constitue pas un gain de débit ; la mémoire dépasse encore les anciens plafonds77B.** Aucune compilation, preuve, charge modèle, mesure ou répétition lancée par le relecteur.

## Identité

86 sources, 8 pilotes, 4 plans et les 2 binaires rehashés : tous conformes au gel. CLI `bb7d25de91d4e5d78afd508aab4f96bb769a3262a3754a46af4ff275379fb4cb` ; tests `c11724af5a332d234e0023ee9f9d625821ed3c7bd52561de33bf82d3afd2c482`. Plan qualité `9033b8c4271426c0b431786a373bed323c3fd6bdc1bd034678b5a5c32a311409` identique au rapport. Source locale F05 sur base documentaire cba5166b/native62cae/moteur80M ; HEAD documentaire b5898866, F01/F06 absents du code compilé. Le manifeste préserve aussi leurs fichiers de tests détachés ; les 86 fichiers ne constituent pas 86 modules compilés. La copie du workflow correspond au patch F05 avec le harnais CPU ajouté ; aucun run CI futur F05 n’est certifié ici.

## Résultats contrôlés

| Contrôle | Résultat et limites |
| --- | --- |
| Suite CPU | **117 lib passés,5 ignorés**,16 intégration passés. Le résumé initial/gates indique 6 ignorés : erreur documentaire seulement, conservée et corrigée ici depuis le vrai log. |
| Suite MLX non ignorée | 130 passés,0 échec,150 ignorés. Contrairement au premier plan F06, l’entrée de suite n’a pas de métadonnées de test unique. |
| Sélections physiques | 15 invocations exactes, chacune 1 passé/0 échec/0 ignoré/279 filtrés ; ce ne sont pas 15 suites complètes. |
| Réductions de bord | 28 cas B2/B3/B4/B8×7 motifs F32, probabilités/IDs/poids aux mêmes octets. NaN/Inf intentionnels : comportement historique conservé, pas une revendication de finitude sur ces fixtures. |
| Vrais routeurs | 1004 cas sur 48 couches,404 rapides ; logits de chaque GEMV scalaire, probabilités et routes bitexacts. F16/F32 et strides, B1–B8 couche 0, B1/B2/B3/B4/B8 autres couches, porte résidente supplémentaire couche 0. Compte d’appels GEMV et branche groupée vérifiés ; tri CPU indépendant des probabilités finies pour les IDs. |
| MoE complet | 24 cas, caches 20/53 et 53+4 pending, cohortes anciennes/uniques, B1/B2/B3/B8, entrées F32, oracle indépendant des sorties capturées. Exactitude des sorties et propriété des slots vérifiées. **Seulement 4 fenêtres UNIQUE en débordement vérifient directement l’ordre Hot avant readback**, pas les 24. |
| Fenêtres historiques | 32 cas UNIQUE, largeur 6–8/caches 20,24,53 : repli 6+2/4+4, Hot préservé après éviction, Cold devenu hit recalculé, pending exclus, I/O échouée et reprise.32 signifie le nombre de cas, pas un test de dtype UInt32. |
| Chargeur | 9 processus à environnement fixé : défaut/0/1 (chacun 2 comparaisons de routes),6 littéraux invalides dont newline. Les invalides terminent le **test** avec exit 0 parce que l’erreur réelle de chargement est correctement assertée. CLI byte 0xff séparée : rejet réel exit 2. Aucun set_var/remove_var après MLX. |
| Préfixes DFlash | 36 couples largeur/rétention B1–B8,248320 logits/ligne, captures et états face au séquentiel, commit/pas suivant/invalides/abandon. Oracle NumPy capture 64000 valeurs à tolérance 0,004, résultat observé byte-equal. Flag F05=1 commun : le scalaire B1 reste au repli, le wide F32 utilise le groupement ; ce n’est pas une campagne cible intégrale flag0/1 séparée. |
| MTP | Budgets 1/2/3/17, 15 propositions/5 acceptées au total ; séquence, états cible/tête, alignement/budget/pending et invalides face au séquentiel. Acceptation et rejet réellement couverts. |

## Sensibilité et preuve

Le composant normal release passe avant les mutants puis après restauration. Le mutant «supprimer la normalisation» échoue sur les routes couche 0/B2/F32 ; le mutant «remplacer les GEMV scalaires» échoue sur le compteur de GEMV. Ce second résultat établit la détection du contrat structurel, **pas une divergence numérique mesurée du GEMM lui-même**. Les 2 mutants physiques sont des binaires debug jetables, exit 101 et joints, sans mesure de vitesse comparée au release. Mutant CPU admettant B1 détecté, puis test et Kani passent après restauration. Les 86 SHA confirment les sources normales figées.

Kani 0.68/CBMC 6.11 : **97 obligations SUCCESS, 5 couvertures SATISFIED**, aucun statut UNREACHABLE. Les propriétés nommées «unreachable» dans la stdlib sont des contrôles qui rapportent SUCCESS, pas des assertions du projet classées inatteignables. Le vrai helper d’admission reçoit 4 dimensions i32 symboliques, bool FP32 symbolique, rang usize supposé≤4 pour la slice de 4 éléments, unwind 5 ; couvertures B2/B8/B1/F16/rang4. Ce domaine est explicite, sans preuve de rang arbitrairement supérieur à 4, et ne prouve pas les calculs Metal/MLX, l’allocateur ou la concurrence.

Tentative réelle MLX : v1 échoue au filtre du test ignoré ; v2 sélectionne le wrapper`#[kani::proof]` dans le module cfg(test), compile avec`--lib --tests --exact`, puis ICE`intrinsics.rs:243`, exit1 en 4,624s, pic748535808B, enfant joint. **Propriété numérique MLX non vérifiée**, sans stub et sans nouvelle usine de harnais. Format et Clippy CPU/MLX ont leurs logs conservés ; leur succès ne remplace pas cette limite.

## Mémoire et fermeture

Pré20/post30 terminés,19/19 enfants joints, sampler 64 observations sans erreur et 45 échantillons VM. Toutes les frontières d’alimentation indiquent AC. Swapins système 56688640B et Swapouts 502988800B constants dans les phases et la courbe : **0 nouveau in/out global observé dans cette qualité**, sans attribution privée. Compresseur système pic 7984152576B, minimum libre échantillonné 109592576B ; ces niveaux restent consignés, sans changement de garde.

Le seul snapshot propre au processus est celui du test de préfixes : PID99457/start1459317546704, empreinte7367252160B, pic vie7401593024B sous 12Go, MLX actif6460126752B/cache67113764B/pic6783715688B. Compression logique courante3375710208B et pic3377102848B dépassent 512Mio ; **caps77B FAILED inchangés**. Compression vie 3422633984B et décompressions 200614 sont des cumuls, pas des deltas ; un seul point ne permet pas d’évaluer le plafond de delta vie 1Gio. Ledger cumulatif 0 observé ne prouve pas l’absence de swap-out privé.

Le processus QA conserve des oracles, captures et états CPU/fichiers temporaires. Sa compression ne représente pas automatiquement celle d’un processus de débit ; aucune cause exclusive n’est établie. Footprint, mémoire MLX allouée, compression logique et RSS d’enfants sont distincts. Le bridge réel et les futurs HC/CH demandent leurs propres observations.

L’hypothèse 82L d’un `selected.to_u32` attendant tout le stream est déjà résolue depuis 56 pour la source soumise et contiguë : attente de son événement, sans primitive Contiguous tardive. Voir [l’audit Hot/readback](/Users/justin/Documents/mix-stq1_0/build/flash-opt-82/F06/hot-readback-audit.md). Cette revue n’en fait pas un nouveau prototype ni un gain.

**Bridge43 lancé par le parent, encore hors de ce gel qualité ; aucun tok/s ni gain promu.** Données et empreintes complètes : [cross-review-quality.json](/Users/justin/Documents/mix-stq1_0/build/flash-opt-82/F05/cross-review-quality.json).
