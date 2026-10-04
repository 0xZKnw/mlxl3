# Audit Rust/Desktop — premier lot vérifié, 22 septembre 2026

> Révision du 23 septembre : un contrôle long contexte a révélé une dérive
> normal/DFlash. Le correctif local aligne désormais les chunks DFlash à **256**
> tokens, capture les queues de moins de 24 tokens en série et conserve les
> projections denses `a/b` mono-token pendant la vérification. Les passages
> historiques ci-dessous qui indiquent 128 tokens décrivent l'état du
> 22 septembre. Voir [l'audit decode/prefill actualisé](audit-decode-prefill-2026-09-23.md).

Implémentation de A01–A04 (diagnostic), P01/P02 (cache Qwen), P03
(prefill sériel Qwen/Ling), D04 (budget spéculatif), I01 (découpage Markdown),
début de I03 (compteurs finaux).
Ce rapport **ne clôt pas les 32 pistes** de l'audit fourni par l'utilisateur.
Source locale : `MLXL3_Audit_Prefill_Decode_DFlash2_UI_2026-09-22.md`.

## Mesures reproductibles

Mac M5, mémoire unifiée 24 GB, sur batterie (91 → 78 % pendant le lot),
MLX 0.32.2, SDK 26.5, build Rust release `fbfb357edffe-dirty`.
Un seul processus chargeant un modèle à la fois. Révision de travail locale,
pas une nouvelle release. Les petits écarts de decode restent sensibles à
la chauffe ; les temps froids de compilation sont séparés des comparaisons chaudes.

### Le gain DFlash dépend réellement du texte et de la longueur

Qwen3.6-35B-A3B EXL3 2.49 bpw, draft DFlash2 livré, greedy
température 0 / top-k 1 / pénalité 1, contexte 4096. Prompt :
« Explain lossless speculative decoding in one paragraph. » (20 tokens rendus).
Deux observations par mode, ordre ABBA après warmups, même binaire et sampler.

| Budget | Normal | DFlash2 | Écart | Acceptés / proposés |
|---|---:|---:|---:|---:|
| 48 tokens | 45,54 tok/s | 68,74 tok/s | +50,94 % | 38 / 45 |
| 128 tokens | 44,92 tok/s | 47,42 tok/s | +5,57 % | 92 / 175 |
| 256 tokens | 44,76 tok/s | 43,14 tok/s | −3,61 % | 181 / 380 |

Texte, empreinte des IDs et nombre de tokens identiques à chaque budget.
Il s'agit d'une **mesure de la baseline applicative**, pas de trois nouveaux
gains dus à ce lot. Elle n'est pas directement comparable à PERF-112 : son
prompt et son ancien sampler de benchmark étaient différents.
[Données brutes](measurements/audit-2026-09-22-bridge.jsonl).

### Réutilisation du préfixe au deuxième tour

575 tokens rendus, 48 générés, 512 tokens réutilisés / 63 recalculés.
Préchauffage du grand prompt **et de son suffixe**, puis OFF/ON/ON/OFF.
Les premières compilations de formes de la passe exploratoire sont exclues.

| Mode | TTFT sans cache | TTFT avec cache | Réduction | Surcoût MLX actif observé |
|---|---:|---:|---:|---:|
| Normal | 1,331 s | 0,275 s | −79,3 % (×4,84) | ~201 MB |
| DFlash2 | 1,783 s | 0,285 s | −84,0 % (×6,25) | ~151 MB |

Même texte et mêmes IDs, decode normal ~44,2 tok/s inchangé. Ce gain porte
sur le **TTFT moteur de ce deuxième tour**, pas sur tous les prompts, le decode
ou le délai jusqu'à l'affichage SwiftUI. L'empreinte physique avec l'allocateur
déjà chaud reste autour de 15,70 / 16,13 GB respectivement ; elle ne se déduit
pas de la taille du checkpoint retenu.
[Données brutes](measurements/audit-2026-09-22-prefix.jsonl).

Un test du vrai bridge avec `fixture.echo` local vérifie aussi les rounds MCP :
deux rounds, 202 tokens générés agrégés, délai outil contrôlé de 200 ms,
réponse finale identique cache OFF/ON. Après l'outil : **895 → 127 tokens
réévalués**, grâce à 768 tokens cachés. Exa est désactivé dans une configuration
temporaire ; aucune requête externe. Cette passe valide le fonctionnement,
pas un pourcentage de performance supplémentaire.
[Données brutes](measurements/audit-2026-09-22-mcp.jsonl).

Annulation réelle au début du prefill puis au premier delta : reprise sur
le même processus/conversation égale à la référence froide, normal et DFlash.
Le banc a d'abord attendu à tort un événement `error` ; corrigé pour exiger
l'événement de production `cancelled`, puis relancé avec succès.
[Contrôle de reprise](measurements/audit-2026-09-22-cancellation.jsonl).

### Prefill sériel sans projection vocabulaire intermédiaire

Sur Qwen et Ling, les forwards M=1 évaluaient les logits ensuite jetés.
Les tokens non finaux évaluent maintenant le même hidden/state, sans head.
La dernière projection vocabulaire, les formes des couches et le decode
normal restent inchangés. Aucun batching Ling ou LFM-MoE non validé n'est activé.

| Modèle | Longueur sérielle | Avant | Après | Réduction |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B 2.49 bpw | 23 | 447,03 ms | 380,99 ms | −14,8 % |
| Qwen3.6-35B-A3B 2.49 bpw | 25 | 487,59 ms | 416,01 ms | −14,7 % |
| Ling-3.0-tiny 4 bpw | 23 | 210,17 ms | 187,41 ms | −10,8 % |
| Ling-3.0-tiny 4 bpw | 25 | 228,34 ms | 203,23 ms | −11,0 % |

Quatre observations par variante, ABBAABBA. Tests 1/2/23/24/25 : logits
et états internes identiques octet par octet (80 tenseurs pour Qwen).
À un token, résultat neutre. Qwen utilise déjà un autre chemin batché à
partir de 24 tokens dans le bridge : ces chiffres ne sont donc pas une
accélération de son gros prefill. LFM/Gemma laissent déjà les heads jetés
non évalués dans leur graphe paresseux ; aucune modification spéculative
n'a été ajoutée à ces chemins.

### Dernier bloc DFlash limité au budget livrable

Le bridge transmet maintenant le nombre de tokens encore affichables. Le
dernier head/verify ne prépare plus cinq propositions quand il ne reste que
quelques tokens ; s'il n'en manque qu'un, une étape target normale suffit.
Le réseau draft conserve ses huit positions bidirectionnelles. Ceci ne prédit
pas les futurs EOS et n'est pas une politique adaptative de largeur.

Deux prompts (explication et fonction Rust), budgets 1..16 : IDs exactement
égaux au target normal indépendant ; pas de pending au-delà du budget, offset
final exact, zéro rejeté sans mutation, contextes larges et serrés. ABBAABBA
chaud, quatre mesures par mode, temps decode **hors prefill** :

| Budget livré | Sans borne de sortie | Avec borne de sortie |
|---|---:|---:|
| 2 tokens | 69,36–69,49 ms | 20,36–20,53 ms |
| 4 tokens | 69,27–69,46 ms | 44,88–44,94 ms |
| 6 tokens | 69,42–69,72 ms | 64,58–64,65 ms |
| 16 tokens | 279,39–280,59 ms | 205,08–221,40 ms |

Les plages sont les médianes des **deux prompts**, pas des intervalles de
confiance. Il s'agit de travail terminal évité, pas d'un nouveau +50 % de
decode général. Le bridge JSON réel passe aussi les 16 budgets, mais cette
seconde passe inclut des compilations de formes : elle vérifie les sorties,
pas les pourcentages de performance.
[Test GPU](measurements/audit-2026-09-22-tail-gpu.log),
[bridge](measurements/audit-2026-09-22-tail-bridge.jsonl).

### Préparation Markdown incrémentale

Le renderer et ses chunks `Equatable` sont conservés. Le cache reprend
aux deux derniers chunks, avec offset du **texte brut** et état des lignes
de tableau (les IDs de rendu comptent aussi les en-têtes répétés).
Un remplacement ou une troncature invalide la réutilisation.

`Swift -O`, 30 appends contrebalancés, aucun modèle ni compilation concurrente
pendant les mesures finales ; moyenne du coût CPU de découpage par update :

| Texte nominal | Prose complète → incrémentale | Tableau complet → incrémental |
|---|---:|---:|
| 64 KiB | 2,074 → 0,383 ms | 1,488 → 0,501 ms |
| 256 KiB | 8,418 → 0,756 ms | 5,928 → 0,892 ms |
| 1 MiB | 34,137 → 2,008 ms | 23,079 → 2,106 ms |

4 845 fragments Markdown donnent le même découpage complet (texte, IDs,
offsets et continuation), plus tests d'en-têtes >16k, pipes sans séparateur,
Unicode composé/décomposé, fences, mathématiques, remplacements et troncatures.
Les 1 194 checks de coloration et la suite Desktop passent aussi.
[Log complet](measurements/audit-2026-09-22-desktop.log).

**Limites :** ni FPS, ni rendu CoreText, ni débit moteur mesuré ici. Le
contrôle exact du préfixe reste O(n) en octets. Un fence ouvert géant sans
frontière reste un seul chunk Markdown : I02 n'est pas résolu par ce cache.
Les contenus stockés/envoyés au modèle ne reçoivent pas de fences artificiels.

## Contrats et limites du cache

- Un checkpoint de prefill Qwen, jamais un état de decode contenant des
  propositions non livrées. Target récurrent + KV et cache draft sont conservés
  ensemble ; aucun trim arbitraire de la récurrence.
- Frontières de chunks complètes (256 normal, 128 DFlash) pour préserver les
  formes et arrondis du recalcul froid. Comparaison exacte des IDs rendus.
- Même conversation, modèle/tokenizer/processus/contexte et mode draft.
  Changer de draft libère le checkpoint ; système/historique/outils modifiés
  ne réutilisent que des IDs strictement identiques.
- Un seul snapshot, pas de recherche multi-blocs ou de cache multi-sessions.
  Si le préfixe du snapshot n'est plus valide, recalcul complet. Au-delà de
  256 MiB de payload logique target, le snapshot n'est pas retenu (draft borné
  séparément à 48 MiB ; le backing et l'allocateur MLX peuvent être plus grands).
- `reuse_prompt_cache=false` désactive et libère le snapshot. Le modèle
  reste chargé. Les autres architectures n'ont pas encore ce cache.

## Compteurs corrigés

Le runtime annonce commit/dirty, profil, version MLX, chemin d'exécutable et
capacité DFlash réelle. Un événement par génération confirme le mode actif.
Un modèle renommé n'est plus exclu une fois ses capacités reçues.

`first_text_seconds` marque un fragment effectivement émis, pas la fin du
round. Les durées/tokens des rounds sont agrégés par sommes, pas par moyenne
des tok/s ; les statistiques par round restent disponibles. La préparation,
le chargement draft, les outils et la durée totale sont séparés.

La taille disque est `model_size_gb`, **pas un pic mémoire**. Les compteurs
MLX actif/cache/pic et l'empreinte physique moteur sont distincts. Le pic
physique processus est explicitement un maximum **depuis le lancement**,
contrairement au pic MLX réinitialisé pour la requête. Les historiques legacy
restent lisibles, sans prétendre connaître la sémantique de leurs anciennes mesures.

Les compteurs DFlash incluent acceptés/proposés/blocs et temps mural cumulé
des blocs. Ce dernier n'est pas un temps GPU par kernel : du travail paresseux
peut être payé au bloc suivant. L'UI affiche pour l'instant les compteurs finaux ;
progression périodique et retard réception→rendu restent à instrumenter.

## Vérification

Préfixer les commandes MLX par :

```sh
export MLXL3_MLX_ROOT="$PWD/.venv/lib/python3.12/site-packages/mlx"
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
```

- `cargo fmt --all -- --check` ; `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`.
- `cargo test --locked --features mlx,chat` : 42 réussis, 35 tests matériels ignorés dans cette passe générale.
- GPU explicites : `native_array_and_kernel_smoke`,
  `bridge_prefix_cache_replays_cold_logits_exactly`,
  `snapshot_restore_replays_logits_and_state_exactly`,
  `draft_prefix_cache_clone_keeps_all_kv_bytes`,
  `qwen_serial_prefill_skips_only_unused_heads`,
  `ling_serial_prefill_skips_only_unused_heads` ; lancer avec
  `cargo test --release --features mlx,chat NOM -- --ignored --nocapture --test-threads=1`, un à la fois.
- `dflash_output_budget_matches_target_and_skips_unused_work` : même commande
  GPU, deux prompts × budgets 1..16 et contrôle de latence terminale.
- `MLXL3_MACOS_SDK="$SDKROOT" scripts/check-desktop.sh` : lifecycle, contrat JSON,
  modèle renommé, stockage, Markdown/Unicode/long code, bridge et transport.
- `.venv/bin/python -m pytest tests/test_bridge_benchmark.py -q` : le banc rejette
  un runtime sans accusé de mode effectif, plutôt que publier un faux ratio.
- Banc réel : `.venv/bin/python scripts/smoke-dflash-bridge.py ENGINE DOSSIER_MODELE DOSSIER_DRAFT`
  avec `--tokens 48,128,256 --repeats 2`, ou `--tokens 48 --check-prefix-cache`,
  ou `--tokens 48 --check-mcp`. Ces derniers utilisent des fixtures locales,
  pas la configuration MCP de l'utilisateur.

Le skill `formal-proof-skill` a guidé les contrôles de source et les contrôles
négatifs, en complément des tests GPU. `inference-engineering` a imposé de
séparer micro-mesures, bridge réel et rendu, avec parité et ordres alternés.
`Ponytail` a conduit à réutiliser snapshots, calculs hidden et cache de chunks
existants, sans nouvelle dépendance ni renderer concurrent.

- Kani 0.68.0 / CBMC 6.11.0, `cargo kani --no-default-features --lib --harness greedy_acceptance_is_the_maximal_matching_prefix` :
  216 obligations, zéro échec, une inaccessible ; deux couvertures satisfaites.
  Domaine 0..7 propositions, IDs u32, unwind 9.
- `cargo kani --no-default-features --lib --harness cached_prefix_requires_exact_ids_and_chunk_boundary` :
  97 obligations, zéro échec, une inaccessible ; trois couvertures satisfaites.
  Longueurs 0..8, IDs u32 symboliques, chunk u8 y compris 0, unwind 34
  pour couvrir le `memcmp` de 32 octets. Unwind 10 échouait, et a été augmenté
  sans supprimer les assertions d'unwinding. Une mutation retirant la comparaison
  des IDs fait échouer le contrat ; original restauré et revérifié.
- CBMC direct sur la FFI C++ mémoire : bloqué par le parsing libc++.
  La FFI est compilée par Clang et exercée sur le vrai GPU, **pas formellement prouvée**.
- `cargo kani --no-default-features --lib --harness speculative_work_reserves_target_within_both_budgets` :
  28 obligations, zéro échec, trois couvertures satisfaites ; deux `usize`
  symboliques (y compris zéro et MAX), aucune boucle. Une mutation supprimant
  la borne de sortie échoue sur deux assertions ; original restauré et
  revérifié. [Log](measurements/audit-2026-09-22-tail-kani.log).
- Kani ne prouve ni Metal/MLX, ni les arrondis flottants GPU, ni SwiftUI,
  ni les échanges de processus Python. Ces chemins ont des tests différentiels
  et d'intégration, pas une preuve d'absence générale de bugs.
- CrossHair/Nagini non disponibles dans l'environnement du banc Python ; le
  chemin sous-processus/queue/signaux est exercé par le vrai bridge et un
  contrôle négatif CPU. Pas de vérificateur source Swift utilisé ; Swift 6,
  AppKit/SwiftUI et comparaisons incrémentales restent des tests finis.

Le détail chronologique, les erreurs intermédiaires et les validations en cours
restent dans [opti.md](../opti.md), entrées AUDIT-01 à AUDIT-05.

## État d'intégration et suite

Code local et moteur release reconstruits. Le CLI local pointe sur ce moteur.
SwiftUI compile et ses tests passent ; **le bundle installé n'est pas remplacé**,
aucun push ni nouvelle release de ce lot.

Restent notamment : préfill draft limité au suffixe utile (P04), batching Ling
avec contrôle indépendant (P06), fenêtre locale Gemma (P07), coût du contexte
long DFlash/target (D01/D02), adaptation fondée sur le coût
réel (D08), et grands blocs de code/métriques UI (I02–I04). Les autres tickets
conditionnels doivent être sélectionnés sur un profil réel, pas activés à l'aveugle.
