# Mémoire et MTP adaptatif — campagne des 9 et 10 octobre 2026

La combinaison compaction du cache + embeddings empaquetés économise **566 à 654 Mo de RAM processus** après réutilisation d’un prompt de 687 tokens sur Qwen3.8-27B. Les allocations MLX actives diminuent exactement de **555 735 040 octets** dans les deux ordres de comparaison. Les sorties et compteurs restent identiques. Aucun gain de vitesse n’est établi.

Code local sur `codex/adaptive-memory-qwen`, base `dee440230bd1e3662eb0dc8c1a619ab8406b5a69`, checkout modifié. Binaire testé : SHA-256 `35f1349b34567d1a14b4cdbbbb999c1517fae8f6b1df020bcaf072da3b643fb3`. **État à la clôture des mesures : app installée inchangée, aucun commit/push/PR/release à ce moment.** Les documents de recherche et modifications du journal présents à l’ouverture ont été préservés.

## Comportements

L’inspecteur propose un contexte prudent, à enregistrer manuellement. Le calcul compte le moteur observé, les états GDN fixes, les KV F16 futurs et une tête MTP compatible. Il réserve 512 Mio pour les caches et au moins max(4 Gio, 25 % de la mémoire physique) pour macOS et le calcul. Le maximum du modèle reste la borne. C’est une estimation ; les autres applications et les activations peuvent réduire la marge.

| Réglage moteur | Comportement / défaut |
|---|---|
| `MLXL3_PROMPT_CACHE_MIB=0..4096` | Budget global du checkpoint de préfill : cible, logits, tête/draft, IDs et conversation. **256 Mio par défaut** ; 0 désactive ce checkpoint. Les vues comptent tout leur backing, les alias peuvent compter deux fois. Ce n’est pas un plafond de RAM totale. |
| `MLXL3_CACHE_COMPACTION=0` | Désactive le contrôle de compaction. **Compaction conditionnelle ON par défaut** : détacher au checkpoint les vues retenant >2× leur payload et ≥64 Kio inutiles, par copie exacte des bits. Jamais dans les snapshots de rollback par bloc. |
| `MLXL3_ALLOCATOR_CACHE_MIB=0..4096` | Borne les buffers inutilisés de l’allocateur MLX ; les tableaux vivants restent conservés. Défaut MLX inchangé ; 0 réduit la réutilisation de buffers. |
| `MLXL3_MTP_ADAPTIVE=1` | **Expérimental, OFF par défaut.** Dans une génération où MTP est déjà activé : profondeurs D0..3, fenêtre de 8 blocs, deux premières observations de chaque profondeur exclues, marge de 5 %, exploration après 32 blocs. Coût mural par input engagé, pas temps GPU pur. D0 entretient le cache de la tête pour reprendre D1..3. Tune reste à profondeurs fixes ; clé runtime distincte. |
| `MLXL3_EMBEDDINGS_PACKED=1` | **Expérimental, OFF par défaut.** Empaquetage exact des 13 bits hauts et exceptions de 3 bits bas par bloc de 128. Gather GPU reconstruisant les 16 bits ; IDs sur GPU et checkpoint inchangé. Fallback dense si l’empaquetage ne réduit pas les allocations. Coût de chargement et débit non qualifiés. |

Capture ciblée : `MTL_CAPTURE_ENABLED=1`, `MLXL3_METAL_CAPTURE_PATH=/chemin/neuf.gputrace`, `MLXL3_METAL_CAPTURE_REQUEST=id`. Effectuer une génération de warmup, puis envoyer la requête NDJSON d’id choisi. Le préfill est exclu ; une capture par processus. La garde ferme sur succès, erreur ou annulation. Xcode complet est absent : les compteurs GPU ne sont pas analysés.

## Mesures de RAM dans des processus propres

M5 Air, 10 cœurs GPU, 24 Gio, macOS 27.2 ; MLX 0.32.2, SDK 26.5. Qwen3.8-27B EXL3 nominal 2 bits, tête MTP 4 bits. Greedy, contexte 4096, prompt synthétique 687 tokens, réponses de 32 tokens, préfixe réutilisé de 512 tokens. Un modèle GPU à la fois, sans compilation ni proveur simultané.

Même binaire, deux wrappers : A compaction=0 / embeddings=0, B compaction=1 / embeddings=1. Budget checkpoint **1024 Mio communs** pour que la référence sans compaction conserve aussi son cache ; allocateur=0 commun, MTP adaptatif désactivé. Driver existant `benchmark_idle_memory.py --short-only`, ordre AB puis BA, cas normal/MTP/cache/erreur/annulation/reprise. Les réglages et commandes exacts sont archivés.

| Après cache chaud | Contrôle A, RAM processus | Candidat B, RAM processus | Réduction |
|---|---:|---:|---:|
| Ordre AB | 10 100 479 064 octets | 9 446 365 832 octets | 654 113 232 octets |
| Ordre BA | 10 100 888 568 octets | 9 534 708 312 octets | 566 180 256 octets |

Allocations MLX actives dans les deux ordres : **9 855 382 460 → 9 299 647 420 octets**, cache libre 0 partout. **32 cas, 24 completions non vides, 4 erreurs attendues, 4 annulations ; quatre moteurs fermés et rejoints.** Hash IDs, texte, contexte, réutilisation et compteurs MTP identiques. [Résumé et autres cas](measurements/adaptive-memory-20261009/compact-packed-summary.json), [AB](measurements/adaptive-memory-20261009/compact-packed-ab.json), [BA](measurements/adaptive-memory-20261009/compact-packed-ba.json).

Ces mesures qualifient la combinaison sur ce prompt court, après réponse, modèle toujours chargé. Elles ne mesurent pas le pic au chargement, le grand contexte ou un gain général. Ne pas additionner ces résultats aux anciennes campagnes ni attribuer tout le footprint aux seuls embeddings. Malgré la déclaration utilisateur secteur, `pmset` observe Battery Power/discharging : timings diagnostiques, aucune accélération établie ; thermique non mesuré.

## Correction et prototypes rejetés

| Essai | Résultat / décision |
|---|---|
| Embeddings réels | 8 positions ×248 320 logits F16, 128 états et hidden exacts ; préfill 24/257 puis 3 pas. Table **2 542 796 800 → 2 241 331 200 octets**, soit −301 465 600 octets, même économie d’allocations MLX actives. Tous les 65 536 motifs F16 testés avec deux permutations sur 131 072 valeurs. Conservé opt-in ; C++/Metal non formellement prouvés. |
| Cache préfill 256 | 128 états/hidden/logits et pas suivant exacts. Payload 170 731 520 octets inchangé ; backing **425 197 568 → 170 928 128 octets**, soit −254 269 440 octets, sous 256 Mio. C’est la taille des buffers du snapshot, distincte du footprint processus. |
| Copie de vues | 8 cas ×65 536 valeurs F16/BF16/F32/U32, contigus/stridés ; bits/forme/dtype inchangés, backing réduit, parent libérable ; erreurs NULL et fallback dense. Contrôle désactivé détecté rouge, puis test ON passé. |
| Sessions MTP | 32 blocs, dont alternance 0/3/0/1/2/0/3/1. Oracle cible mono-token + entretien indépendant du cache tête ; IDs, 128 états, hidden et stockage/forme/dtype des KV exacts. |
| Bridge réel adaptatif, budget cache par défaut 256 Mio | **43 requêtes, 35 completions, 5 annulations + reprises, 2 erreurs attendues, cache chaud 256 tokens, miss conversation distincte.** Histogramme D0..3 : [30,46,57,53]. Tune : 4 profondeurs, chacune 190 decode tokens et 2 hashes identiques entre profondeurs. Profil natif fixed=153 954 304, cible=65 536 et tête=4 096 octets/token. |
| Frontières budget / capture | Budget prefix 0/bad/−1/4097/vide : 7 completions de 8 tokens et 8 erreurs attendues ; 687 tokens réévalués/cached=0 si budget=0. Quatre limites allocateur invalides refusées avant ready. Dix processus rejoints. Captures physiques petites : trois traces, finish/drop/refus overwrite. Capture bridge de 2 tokens après warmup 8 : sortie exacte face normal et répétition sans nouvelle capture. |
| KV8 G64 puis G32/F16 | RMSE 0,0320988575 puis 0,0309084293 au premier prompt de 493 tokens, au-delà du seuil préenregistré 0,03 : rejetés et archivés. |
| KV8 G32/F32 | Court 493/495 tokens, 16 positions : passe. Long **8 701 tokens / 4 positions : RMSE 0,0365358610**, maxabs 0,169921875, aucun changement greedy. **Rejeté au long ; tout le prototype KV8 retiré, KV F16 conservé.** Rollback quantifié, bridge et vitesse KV8 non exécutés après échec qualité. |

Le premier A/B embeddings à budget 256 Mio a échoué : la référence sans compaction ne réutilisait pas son cache ; B n’a pas été lancé, parité fausse. Les harnais embeddings avec gros Vec CPU avaient aussi un footprint confondu : ils ne prouvent pas une économie de 2,6 Go. Ces échecs et corrections restent archivés. Un audit additionnel exigeait à tort decode_tps>0 pour une seule sortie ; le contrat correct decode_tokens=0/decode_tps=0 a été vérifié sur les mêmes données sans répétition GPU.

Capture locale non publiée : `build/adaptive-memory/bridge-profile.gputrace`, **9 802 604 683 octets, 1 917 fichiers**, empreintés dans [le manifeste](measurements/adaptive-memory-20261009/bridge-capture-files.json.gz). Occupancy, bandwidth et temps GPU purs restent **non mesurés**, faute de Xcode complet.

## Fonctions modifiées et frontières

| Entrée → fonctions | Préconditions et effets visibles |
|---|---|
| ready → `Qwen35Moe::context_memory_profile` → `ContextMemoryProfile.recommendedTokens` → StudioModel/inspecteur/save/reload | Dimensions natives validées, tailles checked ; profil absent/invalide refuse conseil ; réglage appliqué par action utilisateur. |
| préfill / préfill_mtp → `PromptCache.capture/within_budget` → snapshots Qwen/DFlash/MTP → `Array.retained_bytes/compact_for_cache` | Snapshot à offset égal à l’historique ; budget 0 court-circuite avant copie ; comptabilité complète après ajout tête ; copie conditionnelle hors rollback, erreurs propagées. |
| `Array.initialize` → `mlxl3_set_cache_limit` | Variable entière bornée ; absence conserve défaut ; tableaux vivants intacts. Limites 0/1 vérifiées physiquement. |
| bridge_generate_round → `Session.advance` → `AdaptiveMtp.observe/next` | Cible+tête détenues par session, budgets existants ; profondeur 0 seulement en adaptatif ; cache tête entretenu avant target.forward. Statistiques additionnelles par profondeur, agrégées entre rounds. |
| Qwen.load → `Embedding.new` → `mlxl3_pack_embedding` ; take/mtp_embeddings → packed_embedding.metal | F16 rank2, largeur divisible par128, compte et offsets bornés ; IDs vocabulaire garantis par appelants privés ; reconstruction exacte et fallback si allocations non réduites. |
| bridge_generate_round → `capture_for_request` → `MetalCapture.finish/Drop` | Configuration explicite, chemin neuf/parent existant, sélection atomique unique ; tenter stop même après erreur sync. Pas une preuve de concurrence MLX. |
| Tune → runtime_key | Profondeurs fixes conservées, options adaptive/embedding-pack incluses dans la clé pour séparer leurs profils. |

## Vérifications et limites

- Variante livrée `mlx,chat` : format, Clippy strict, build et **82 tests passés, 71 ignorés**. Les tests physiques ignorés ne sont pas comptés comme exécutés : chaque sélection GPU de la campagne a son log/statut et un oracle non vide. Variante CPU `chat` complémentaire : **66 passés, 3 ignorés**, Clippy strict et format passent. Avertissement Cargo préexistant de future incompatibilité `block 0.1.6`, sans échec du lint actuel.
- Python **338 passés / 4 skips** : trois adapters EXL3 facultatifs absents et modèle Ling absent. Contrôles transport/driver répétés dans le contexte du nouveau harnais : **69 passés**, dont silence, interruption, réponses invalides et rapports incomplets. Ruff strict/format/pycompile du harnais passent ; CI raccordée à ces vérifications statiques.
- Suite Desktop complète, Swift strict, **240 oracles linéaires** de recommandation et frontières/overflow/coût tête/compatibilité ancien bridge ; parcours StudioModel→enregistrer→recharger. Pas de vérificateur source Swift applicable installé.
- Kani 0.68 : suite prototype **50/50 harnais**, 6 265 SUCCESS, 132/132 covers, 76 UNREACHABLE de stdlib/modèles Kani inspectés, aucun échec. Après retrait KV8, 49 contrats conservés inchangés ; méthode réelle AdaptiveMtp.next ciblée : 190 obligations /1 inatteignable stdlib/2 covers ; nouveau prédicat compaction : 6 obligations/2 covers, satisfaits. **Ce n’est pas une suite entière réexécutée sur le SHA final** : seuls contrôles affectés répétés.
- Domaines CPU : usize/u64 symboliques complets pour budgets/compteurs/coûts, tableaux de 4 profondeurs et 6 composantes ; unwind 34 pour memcmp de 32 octets, assertions d’unwinding conservées. Sélecteur capture borné aux IDs de 0..8 octets/unwind10. Pas de preuve de l’allocateur, MLX, Metal, concurrence ou génération complète. Mutants inclusion des coûts froids et refus du budget exact détectés sur copies isolées ; version intacte passe.
- C++ : Clang `-Wall -Wextra -Werror` passe sur notre source, headers MLX en `-isystem`. CBMC 6.11 trouvé dans Kani mais incapable de parser les vrais headers libc++ : copie/encodage **non vérifiés formellement**, sans stub. ESBMC absent. CrossHair tenté à nouveau sur le harnais Python : module absent. Tests de bits ≠ preuve déductive des kernels.
- CI GitHub inspectée : Rust/Kani/chat, build MLX0.32.2 sur macOS26 et Desktop ; nouveaux tests/harnesses automatiquement découverts, nouveau harnais physique linté/compilé seulement. Branche non publiée : **aucun run du nouveau code à annoncer vert**.
- Autres modèles/architectures, vrai modèle DFlash, qualité applicative sur grands contextes et pic préfill/chargement non qualifiés dans cette campagne. Profilage avancé bloqué par l’absence de Xcode complet ; vitesse non concluante faute de conditions secteur observées.

Le fichier global de conversations fait **259 661 octets** (métadonnée seule, contenu non lu). Debounce, sauvegarde et récupération existent. Une migration par conversation ne justifie pas son risque actuellement : piste optionnelle différée, aucune donnée utilisateur modifiée.

Commandes, sources/binaires/options/dépendances, checkpoints et échecs dans [les preuves](measurements/adaptive-memory-20261009/), [l’identité finale](measurements/adaptive-memory-20261009/final-identity.json), [la synthèse de vérification](measurements/adaptive-memory-20261009/verification-summary.json) et [le journal](../opti.md). Aucun essai, modèle de test ou proveur encore actif.

## Tester la copie locale de l’app

Une copie de test est construite dans `dist/MLXL3 Desktop.app`. Fermer l’ancienne app avant de lancer celle-ci pour éviter deux moteurs ou deux écritures sur le même historique. Cette copie utilise les préférences et modèles habituels ; elle ne remplace pas /Applications.

Depuis le dépôt, lancer le nouvel exécutable et forcer son moteur embarqué :

```bash
MLXL3_EXECUTABLE="$PWD/dist/MLXL3 Desktop.app/Contents/Resources/runtime/mlxl3" \
MLXL3_MTP_ADAPTIVE=1 \
MLXL3_EMBEDDINGS_PACKED=1 \
MLXL3_ALLOCATOR_CACHE_MIB=0 \
"$PWD/dist/MLXL3 Desktop.app/Contents/MacOS/MLXL3Studio"
```

Charger Qwen3.8-27B, activer MTP dans l’app pour exercer l’adaptation et ouvrir l’inspecteur pour le conseil de contexte. Enregistrer le contexte seulement pour appliquer ce conseil. Envoyer un prompt, puis le réutiliser ; surveiller la mémoire après la réponse, modèle chargé. Brancher le Mac pour comparer la vitesse. La capture GPU ne s’active pas avec cette commande.

Pour tester les nouveaux comportements par défaut, lancer la même commande avec seulement `MLXL3_EXECUTABLE`, sans les trois autres variables. Les options expérimentales sont héritées au lancement du processus ; un double-clic ordinaire ne les active pas. La procédure de compilation reproductible reste `scripts/build-macos-app.sh` avec MLX0.32.2, Cargo/Swift et SDK compatible configurés.

Logs volumineux et manifeste de capture compressés sans perte pour la PR : [index et empreintes](measurements/adaptive-memory-20261009/log-archives.json). Les originaux restent locaux ; chaque décompression a été comparée octet pour octet.
