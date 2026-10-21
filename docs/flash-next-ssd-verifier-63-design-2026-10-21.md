# 63V — prochain prototype du vérificateur Flash multi-lignes

**Conception uniquement, aucun code ni benchmark63 lancé.** La campagne62H v2 doit finir avant toute compilation locale. La base source est2eccb469 ; les réglages Hot seront fixés après la qualification complète62, pas choisis selon un seul prompt.

## Réemploi du moteur existant

La tape compacte n'existe pas dans le parcours Flash Next, mais le moteur possède déjà `gated_delta::step_with_gates_tape`, `RecurrentTape::replay` et les contrats CPU `gdn_tape_prefix`. Qwen35 utilise ces composants. Leur corps FP16/FP32 est compatible avec la famille de kernels déjà appelée par Flash ; cette compatibilité doit être vérifiée sur les formes Flash16 têtes clés/48 têtes valeurs/128 dimensions, sans copier le GDN BF16 de Splash.

L'autre pièce existante est `_qmv_tile_kernel.metal` : le kernel sait avoir plusieurs lignes et une accumulation de référence, mais `Exl3Linear::forward` resérialise les lignes Flash ordinaires et `forward_qmv_batch` n'active cette accumulation que pour le FC caché MTP. Une concaténation seule n'accélère donc pas ces projections. Une variante Flash ciblée NT1/MB2/SG4 doit conserver les splits, Hadamard, conversions FP16 et réduction finale de chaque ligne ; les autres modèles gardent leur chemin.

## Prototype envisagé

1. Préparer PLE et HC par position, dans leur ordre causal, puis assembler les entrées GDN indépendantes. Garder QSA inchangé pour ce premier essai.
2. Grouper les projections EXL3 compatibles avec l'accumulation de référence ; repli séquentiel pour les formats/formes non qualifiés. Tester le kernel avant de brancher le vérificateur.
3. Effectuer la récurrence GDN multi-token avec sa tape compacte existante ; produire toutes les sorties utiles, conserver l'état final et rejouer uniquement le préfixe retenu lors d'un commit partiel. Conserver aussi le bon historique de convolution et l'état PLE de chaque préfixe.
4. Partager la préparation entre `verify_draft` et `forward_pair` seulement après parité de la première entrée ; rétablir exactement les états après rejet, erreur ou annulation.

Une seule option expérimentale explicite doit comparer le chemin précédent et le prototype dans le même binaire. Défaut précédent conservé avant qualification. La préparation de la tête sera regroupée uniquement si son format réel et son accumulation sont qualifiés ; ne pas faire passer une simple concaténation pour un partage des poids.

## Contrôles requis avant le débit

- Comparaison physique indépendante du kernel avec les projections mono-ligne : B1..B8, formes réelles Flash, formats concernés, strides, dtypes, valeurs finies, mauvais layout/mauvais slot détectés. Aucun MB2 systématique des experts de48 : il s'agit ici de matrices denses partagées entre positions.
- GDN16/48/128, T1..T8, tous les préfixes : sortie, état final et replay bit à bit égaux à la suite des kernels actuels ; formes/dtypes/commits invalides refusés.
- Modèle complet,36 préfixes B1..B8, logits248320/captures/136 états, puis continuation ; MTP1 et DFlash2 réels, bridge/CLI, annulation, reprise et abandon.
- Contrats CPU symboliques pour géométrie/offsets/préfixes, Kani sur le code réel et tentative MLX ; conserver les échecs/ICE. Pas de prétention de preuve GPU à partir des contrats CPU.
- fmt, Clippy strict, build release livré, suites pertinentes et CI du commit exact. Protocole détaillé à préenregistrer dans `opti.md` avant le prototype.

L'ancien payload récurrent de sept états par36 couches représente792723456 octets pour B8 ; ce calcul géométrique n'est pas une mesure de RAM récupérable. Mesurer le footprint, les allocations, le coût du replay et le débit complet, avec les gardes0Swapout/12e9 octets. Garder une comparaison128 tokens puis une campagne512distincte, et les durées hôte inclusives séparées de tout temps GPU exclusif non disponible sans l'outillage correspondant.
