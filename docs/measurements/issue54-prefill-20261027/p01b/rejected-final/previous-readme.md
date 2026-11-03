# Issue 54 — préfill Flash Next

Statut : **prototype local, vitesse non mesurée, défaut sériel conservé**. Aucun push, remplacement de l’app ou publication.

## Portée

Source : [issue #54](https://github.com/0xZKnw/mlxl3/issues/54), pistes P01/P02 inspirées de l’organisation aMference. Le moteur reste Rust/Metal/MLX ; les commandes de supervision utilisent les outils Python existants. MTP ne change pas de préfill.

Base Git : `7978fceac30d991a8c06d72114fc932b2443f555`. Le contenu native/src/ffi/shaders de origin/main `7678456da8ff163f0609ea4e709a6cf5d5a97895` était identique avant les prototypes. Les autres modifications de workspace sont préservées.

## Versions des prototypes

- **P01A** : blocs 2/4/8, préparation GDN multiligne, scan final sans tape/snapshots, MoE groupé existant, head final M1, captures causales pour DFlash et append row1. SHA sources/binaires dans source-v1.json puis source-v2.json ; patch de v2 dans prototype-v2.patch.
- **P01B en cours** : lots par couche jusqu’à 128, tuiles internes ≤4, entrée AR réelle du bridge avec taille de lot adaptée, drapeau d’annulation partagé. Configuration expérimentale 1/2/4/8/32/128, défaut1. Les preuves numériques A ne sont pas attribuées à B.

## Graphe des modifications

| Fonction | Appelants / contrat |
|---|---|
| flash_support::prefill_chunk | from_checkpoint ; littéraux stricts, rejet avant allocation du modèle |
| Gdn::forward_rows / Layer::prepare_rows | vérification : retain_prefixes=true ; préfill : false, aucun rollback/tape conservé |
| mean_streams | captures partagées avec le draft, formule et conversions inchangées |
| FlashNext::forward_prefill_block | lote borné, causalité, dernier head et état, reset sur erreur |
| FlashNext::prefill_tokens / forward_tokens | drapeau réel pour bridge ; API existante conservée pour CLI/calcul sans annulation |
| prefill_dflash2 | captures de tous les tokens, append du draft à un token par appel et position absolue |
| NativeChatModel::forward_many / prefill_chunk_size / préfill bridge | sélection effective de Flash et transmission de l’annulation |
| SsdMoe::forward_draft_rows | corps inchangé, réutilisé aussi sans feature chat |

## Résultats inspectés

- Rust CPU : `cargo test --no-default-features --features chat`, 247 passés et 6 ignorés ; le build sans MLX ne vérifie pas la FFI.
- Clippy strict MLX+chat et build release MLX+chat A passés. Compilation MLX sans chat avec direct-metal passée. `mlx` sans direct-metal et sans defaults échoue dans array.rs:492, configuration préexistante incorrecte, diagnostic conservé.
- Kani A : parseur UTF-8 symbolique 0..2 octets, unwind4 ; 228 obligations, 0 échec, 12 inatteignables, 2/2 couvertures.
- Kani B : parseur UTF-8 symbolique 0..3 octets, unwind5 ; 232 obligations, 0 échec, 8 inatteignables, 4/4 couvertures. Ceci est une vérification bornée du parseur, aucune preuve du GPU.
- Kani vraie entrée MLX A : ICE intrinsics.rs:243 avant obligations, propriété non vérifiée. Tentative B à faire après fin de l’autre chat.
- GPU A : scan GDN avec/sans tape et étapes sérielles, T1..8 ; 1 test passé, zéro nouveau Swapout.
- GPU A : 54 cas AR/DFlash sur 1/2/3/7/8/9/17/87/256, blocs2/4/8 ; 248320 logits, états, 12800 captures/token et KV draft byte-exacts, continuation3, erreurs/contexte/reset. Données finies pour logits/captures/états.
- Vérificateur A : 36 commits de préfixes, logits/états byte-exacts ; 64000 captures byte-exactes face à la fixture NumPy indépendante, RMSE relative0 pour tolérance historique0,004.
- Mutation A : répétition de la capture row0 ; 1 test exécuté, échec101 attendu sur les clés KV, source restaurée exactement. Les assertions de métadonnées/finitude KV de v2 commencent à être validées par le contrôle négatif, mais le groupe positif v2 a été interrompu par un build externe ; ne pas annoncer ce groupe comme passé.
- B : source en cours, premières vérifications Clippy/parseur passées avant ajout de l’annulation. Vérification physique B, suite finale et vitesse restent à exécuter.

## Interventions / limites

Des launches ont été refusés/interrompus par la garde : propre nom mlxl3-check au premier contrôle (corrigé par réemploi du nom prototype-engine), puis builds/tests lourds de l’autre chat. Tous les rapports d’échec restent présents. Le user a demandé d’attendre la fin de « Ajouter le mode agent et Lab » : aucun message envoyé, aucun de ses processus arrêté, aucun test GPU lancé en parallèle de ses essais Qwen.

La suite pytest générale a été lancée trop largement : 1849 passés, 64 sautés, 51 échecs liés au paquet ponyexl3 absent et à ses imports historiques, hors changement moteur. Unittest discover était un runner inadapté aux tests pytest : 7 erreurs de collecte. Rapports conservés ; ces sorties ne valident pas le moteur. Aucun correctif hors périmètre.

Les workflows PR/push main sont inspectés ; la CI inclut le harnais Kani du parseur. Aucun run distant ne teste les modifications locales non publiées.

## Campagne prévue

Modèle EXL365c89531, draft Q4/ring a22a1744, M5 standard24Gio, MLX0.32.2, cache3Gio/53slots, HotWide/Unique1, nofetch, B2 fixe, 4 readers, overlap1, allocator64Mio. Dépendances et JSON de checkpoints dans dependencies-v1.json ; poids complets non rehashés. MAIN/HOLD/LONG figés dans prompts-v1.json (35/30/194 tokens bruts, taille réelle avec template à relever).

Corriger/qualifier B et son bridge avant vitesse. H1/C128 sur un même binaire figé, puis ordre inversé si écran favorable ; warmup et 3 répétitions par prompt/bras, 128 sorties identiques. Seuils : ≥1% TTFT et temps complet pour chaque prompt, decode pas dégradé de >1%, stabilité/dérive≤3%, pic≤12Gio, extra compression C≤256Mio, zéro nouveau Swapout, secteur, surveillance des processus concurrents. Nouvelle ancre B après fin de l’autre chat. Statut final et tableau chiffré seront ajoutés après inspection des résultats.

## Dernier état P01B — attente RAM

B est compilé (moteur release MLX+chat, tests lib/main, MLX sans chat) ; Clippy strict a passé, 247 tests Rust CPU passés/6 ignorés, 16 régressions du superviseur passées. Kani vraie entrée B rencontre à nouveau l’ICE intrinsics.rs:243, aucune obligation GPU vérifiée. Les deux mutants B sont compilés dans des exécutables distincts et les sources correctes restaurées exactement ; ils ne sont pas encore exécutés.

Le premier groupe physique B s’est interrompu après 70 sous-cas (longueurs1..7), avec 132300800 octets de nouvelles écritures swap. Les 100 cas et contrôles d’erreurs finaux ne sont pas achevés, donc aucun groupe B complet validé. Aucun benchmark de vitesse n’a été lancé. Le user a été informé et la reprise dépend d’une libération de RAM confirmée.

Le harnais ajoute maintenant une trace MemoryStats par sous-cas pour diagnostiquer la pression système/croissance du processus. Binaire diagnostique compilé/non exécuté ; code de production inchangé (empreinte de préfixe contrôlée dans p01b/memory-diagnostics-source.json). Ancienne ancre et échec restent immuables dans p01b/anchor.json et model-short-plan-result.json. Une nouvelle ancre dédiée sera nécessaire après le changement de conditions, avec un nouveau nom de campagne. Le chemin sériel reste le défaut.
