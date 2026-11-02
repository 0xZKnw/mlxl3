# T02 — lecture statique : fenêtre commune de codewords

État au 29 octobre 2026 : **lecture seule, aucun prototype, test, build, preuveur, benchmark ou modèle GPU lancé** pour cette étude. Le résultat ne qualifie aucun gain. `opti.md`, les archives pertinentes et les deux kernels actuels ont été relus avant cette recherche.

## Décision proposée

**Ne pas relancer la fenêtre commune K2 comme une nouveauté.** C’est déjà le mécanisme de Flash54Q sur nos formes d’experts et de HENK0O-QWEN27-02 sur un autre modèle. Le lecteur Turbo n’efface pas leurs résultats.

La fenêtre commune **dense K4** est une piste distincte des essais retrouvés, mais sa nouveauté au niveau des instructions reste **non vérifiée** : les deux groupes actuels de K4 lisent déjà la même paire de mots et le compilateur peut éliminer les doublons. Elle vient après T01/T04 dans le triage, sauf preuve concrète d’instructions ou de temporaires évitables. Aucun nouveau préenregistrement de mesure n’est effectué ici.

## Source primaire et correspondance du format

Source Turbo lue par le connecteur GitHub : [`src/kernels/lane/helpers.metal`](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/src/kernels/lane/helpers.metal), commit `345bee9e35920ee77292055d873fca891146529c`, blob Git `50cebdf16cb501693c7264ad0603102147793e7a`. La section 7 de [`docs/KERNELS.md`](https://github.com/davidtai/turboexl3/blob/345bee9e35920ee77292055d873fca891146529c/docs/KERNELS.md) décrit les plans/lecteurs et leur comparaison à `format.unpackTile` ; ce contrôle amont n’est pas une validation MLXL3.

Chez Turbo, `N` est le nombre de halfwords par tuile de 256 coefficients. Pour nos taux entiers, **N = 16 × K**, et le nombre de mots `uint32` vaut `N/2 = 8K`, comme notre `PACKED_U32`. L’étendue des huit fenêtres de 16 bits est `7K + 16` :

| Taux local | N Turbo | Étendue utile | Lecteur Turbo |
|---|---:|---:|---|
| K2 | 32 | 30 bits | fenêtre 64 bits, dont seule la moitié basse est utile |
| K3 | 48 | 37 bits | fenêtre 64 bits |
| K4 | 64 | 44 bits | fenêtre 64 bits |
| K6 | 96 | 58 bits | fenêtre 64 bits |
| K7 | 112 | 65 bits | fenêtre 96 bits |
| K8 | 128 | 72 bits | fenêtre 96 bits |

Le seuil général `n≤108` concerne aussi les taux fractionnaires pairs de Turbo. Notre chargeur admet seulement K entier de 1 à 8 ; les QMV batch/mapped refusent K7. Le chemin dense K7 dispose d’une autre branche. **Le pilote Flash K2/K4 n’a donc aucun besoin du lecteur 96 bits ni d’un changement de format.**

`exl3_tile_load` reconstruit un début de fenêtre depuis le plan lane, recule depuis la dernière fenêtre, puis charge trois mots pour le cas 64 bits, quatre pour le cas 96 bits. Les extractions `exl3_tile_cw` ont des décalages constants. Les mots inutiles peuvent disparaître par spécialisation ; compter les accès dans le source ne prouve pas leur coût matériel.

## Parcours et taux réellement présents

Les headers safetensors du checkpoint local `Qwen3.8-Flash-Next-exl3-205-65c89531` ont été lus, sans données de poids ni chargement de modèle. `quantization_config.json` annonce MUL1, bits moyens 2,05 et head_bits 4. La dernière dimension du trellis vaut 16K :

| Projection réelle | Entrée → sortie | Dernière dimension du trellis | Taux |
|---|---:|---:|---|
| Experts gate/up | 2560 → 640, deux projections | 32 | K2/MUL1 |
| Experts down | 640 → 2560 | 32 | K2/MUL1 |
| Couche0 GDN qkv | 2560 → 10240 | 64 | K4/MUL1 |
| Couche0 GDN z | 2560 → 6144 | 64 | K4/MUL1 |
| Couche0 GDN out | 6144 → 2560 | 64 | K4/MUL1 |
| Shared gate/up/down | 2560 ↔ 640 | 64 | K4/MUL1 |
| lm_head | 2560 → 248320 | 64 | K4/MUL1 |

Les signatures déjà capturées de POOL04 confirment les deux formes expertes K2, `CB=2`, slot-major, NT1/SG4 et réduction de référence. Il s’agit de lecture d’anciens artefacts, pas d’une nouvelle mesure.

Le kernel mapped actuel possède déjà `MLXL3_K3_WINDOW_DECODE` lorsque K=3 ; K2 conserve les deux groupes de quatre poids. Le dense `_qmv_tile_kernel.metal` conserve aussi les deux groupes de quatre. Les `Exl3Group` qui réutilisent le corps mapped forcent explicitement la macro K3 à zéro ; il faut distinguer ces appelants des experts K3.

## Pourquoi le K2 Turbo n’est pas un nouvel essai Flash

Pour une lane L, la dernière fenêtre K2 finit en `16L+528`. Le décalage vaut 16 pour les lanes paires, zéro pour les impaires ; huit codewords occupent 30 bits. Les seuls bits de la fenêtre 64 bits effectivement consommés tiennent donc dans son `uint32` bas. Après spécialisation, l’algorithme est celui de l’archive54Q : deux mots cycliques, un merge/décalage et huit extractions espacées de deux bits.

La différence de structure/template Turbo ne démontre pas une différence d’instructions. Une éventuelle reprise K2 devrait identifier une différence concrète de code généré, de durée de vie ou une nouvelle forme réellement exécutée, puis la préenregistrer comme telle ; le seul nouveau dépôt n’est pas une raison suffisante.

| Antécédent relu | Mécanisme et résultat | Portée pour T02 |
|---|---|---|
| HENK0O-QWEN27-02, 5 octobre | Une fenêtre K2 de huit états, variantes ulong puis uint32 ; 162 cas FP32 exacts, variante uint32 en recul de 3,4–8,2 % sur les trois microformes | Même principe K2 ; modèle/dispatch différents, pas une preuve contre tout K4 |
| Flash54Q, 20 octobre | Fenêtre K2 de huit codewords sur nos experts 2560→640 / 640→2560, couches0/23/47, B1/B2 ; qualité120formes +24 négatifs ; +0,289..1,509 % par cellule, trois cellules sous le seuil et contrôle A/A non qualifié | Déjà testé ici, retiré ; aucun gain modèle établi |
| Q27-15 et correction v2, 16 octobre | Funnels32 sur K1..4 dense/groupé, **garde les deux groupes de quatre** ; aucun filtre retenu, correction de NT faite sur les onze cellules affectées | Inclut K4 mais ne teste pas une fenêtre commune de huit ; ne pas refaire son remplacement ulong→funnel32 |
| PERF106, 20 septembre | Funnels32 QMV K6/MCG ; parité passée, decode −3,35 % | Autre taux et autre regroupement, rappelle que ulong peut déjà être mieux compilé |
| OPT-2026-09-07-10 | QMM générique, extraction64→funnel32 rejetée | Autre kernel ; ne valide aucune réécriture manuelle32 |
| K3 mapped courant / ancien grouped K3 | Spécialisation K3 déjà présente ; anciens grouped K3 non retenus | T02 n’est pas un manque général du moteur |
| PERF89 / audit79 | LUT de 2^K états invalide et retirée ; LUT MUL11021 distincte restée exploratoire | K est un taux de transition, le codeword reste sur16bits ; ne pas importer une LUT petite ou la carte différée Turbo |

Recherche ciblée du journal, des rapports qwen27/qwen38, des sources archivées54Q et des scripts funnel/v2 : **aucun essai dense K4 à fenêtre commune de huit retrouvé**. Ce constat est limité aux preuves disponibles et ne prouve pas une nouveauté d’ISA. Certains anciens prototypes HENK sont décrits comme temporaires ; leur absence actuelle ne justifie pas de répéter silencieusement une mesure.

## Conditions numériques et mémoire

- Garder le pointeur local de tuile et son layout `[tileK,tileN,packed]`, ou slot-major pour un futur essai expert. Ne pas copier le repacking global Turbo.
- Le codec local écrit les halfwords à `logical ^ 1` ; la vue UInt32 est déjà l’ordre attendu par les kernels. Aucun deuxième SWAP16, changement d’endian ou lecture `ushort` réordonnée.
- Tous les indices de mot restent modulo `8K`. À K4, les mots utiles sont `L` et `L−1` modulo32 ; la lane0 doit lire31 puis0. La fin commune est alignée32, le décalage commun est zéro ; les extractions ont les décalages28,24,…,0 et un masque16bits.
- Garder **le même `mlxl3_decode_codeword`**, donc les mêmes poids FP16 et la conversion float. Les helpers Turbo `exl3_mul1_t/th` et l’application différée de l’affine par sortie sont hors périmètre : ils modifient la chaîne d’arrondi.
- Garder la même suite de FMA pour chacun des huit accumulateurs, l’ordre tileK/split/SG, la réduction de référence `r puis group`, les casts et les Hadamard. Une permutation de calculs entre accumulateurs indépendants doit encore être vérifiée sur les sorties complètes.

## Plus petit prochain contrôle utile, si T02 est retenu

1. D’abord comparer le travail généré pour le **dense K4 actuel** et une fenêtre commune bornée à ce taux, sans modifier NT/SG/splits/MB ni importer la carte MUL1. Vérifier si les doubles chargements sont déjà éliminés et si la fenêtre commune augmente les registres/spills. Sans différence concrète, garder cette piste basse priorité.
2. Si un mécanisme réel reste, préenregistrer un candidat isolé du dense K4, avec extraction d’un codeword à la fois depuis la fenêtre, sans tableau de huit codewords vivants. Le couplage éventuel à T10 doit être identifié séparément ; ne pas masquer un changement de géométrie dans ce test.
3. Oracle d’extraction indépendant : `codec::unpack` sur des tuiles brutes adversariales et bit-basis, toutes les32lanes et les huit états, bornes/retour cyclique31→0, puis comparaison des valeurs FP16 décodées. Ce contrôle doit porter sur le vrai extracteur candidat et détecter une mutation d’offset.
4. Oracles GPU existants utiles : `linear::tests::flash_reference_batches_match_independent_single_rows` (vraies qkv/z/out/head, B1..9, FP16/FP32, strides), puis les cinq positions de référence indépendante `qwen4::tests::flash_matches_independent_reference_states_and_logits`. **Le différentiel batch/mono seul est insuffisant si les deux chemins partagent le même shader modifié** : figer un témoin non modifié ou utiliser l’oracle de format/trajectoire indépendant. Puis paires/MTP/DFlash et bridge réel selon les appelants affectés.
5. Seulement après correction, microbench des vraies projections et débit/TTFT complet alternés, sans transformer un gain du GEMV en gain du moteur. Aucun budget de gain n’est déduit du source.

### Limite T10 liée à ce code

Flash traite actuellement les lots logiques2..8 avec **MB=2**, plusieurs groupes, via `flash_reference_batch_layout`; ce n’est pas une tuile MB8. Ses `x_values[MB][4]` et poids décodés peuvent faire l’objet d’une étude de durée de vie à MB inchangé. Importer directement les chunks6–8 de Turbo changerait une géométrie déjà qualifiée et constituerait un autre essai. Aucun spill local n’a été mesuré ici.

## Identité locale lue

Checkout `7978fceac30d991a8c06d72114fc932b2443f555` modifié localement après retrait du pool ; les empreintes ci-dessous identifient les vrais fichiers lus. La comparaison publiée dans l’issue utilisait `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`, donc ces identités locales priment pour un futur essai.

| Fichier | SHA-256 |
|---|---|
| `_qmv_mapped_tile_kernel.metal` | `02f91b3a97be9c6323fbc0a4c9d4d1ef13f68504eef1065b60d2c50a7def98c8` |
| `_qmv_tile_kernel.metal` | `27e37f490799673a62f07138cb6fb8c5b3e787fc7fbc9168668a59cf1c15bc88` |
| `native/src/linear.rs` | `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e` |
| `native/src/codec.rs` | `202dbfaf1ca622ce1f3fea4082228de26beb54be834857d0b208ae62722f1c49` |
| checkpoint `quantization_config.json` | `c057f732f92379dac8b345ee065d3f07c72df77fead15ee4ec972e8e9831bef5` |

Chemins de code consultés : `linear.rs:368` (dispatch), `:550` (QMV batch), `:1359` (mapped), `:1512` (spécialisation K3), `:1643` (codebook), `codec.rs:34..123` (packing/cyclicité), `contracts.rs:626` (MB2 de référence). Aucun fichier exécutable n’a été modifié.
