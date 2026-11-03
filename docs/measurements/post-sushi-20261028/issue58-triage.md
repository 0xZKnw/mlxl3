# Issue58 — tri en lecture seule pendant la pause

[Issue58](https://github.com/0xZKnw/mlxl3/issues/58), ouverte le28octobre2026, modification10:05:07UTC, aucun commentaire au moment de lecture. Texte/sources consultés, pas de nouvelle expérimentation. [Empreinte du snapshot](issue58-source.json), copie intégrale locale sousbuild/post-sushi-20261028/issue58.md. Son snapshot publiéf3f2c1d5 peut différer du worktree7978fcea modifié ; chaque candidat doit être confronté à la source locale avant essai.

Le dossier07 correspond exactement à POSTPROFILE05 en cours. Finir cet essai avant d'en ouvrir un autre, comme demandé par l'utilisateur. Les cinquante dossiers sont des hypothèses, pas cinquante gains compatibles.

| Priorité après Sushi | Dossiers | Intérêt et premier filtre |
| --- | --- | --- |
| Forte, CPU d'abord | 12/13/15/49 | Pool global à capacité identique, admission prefill/decode et suffixes rejetés ; comparer séparément LFU/CLOCK/ARC. Reproduire le cache initial réel et conserver les leases. Un replay froid ne vaut pas une trace chaude ; pas de nouvelle allocation statique47/55. |
| Forte, test ciblé | 10 | Vrai top10 SIMD512 : le local SIMD couvre seulement256/top8, Flash garde le rang quadratique. Différence réelle avec l'early-exit F06 rejeté ; conserver clés64bits, ties, ordre, sommes et divisions. Le nombre de comparaisons n'est pas un facteurTPS. |
| Forte pour TTFT | 24/33 | Grouper la maintenance du cache DFlash du prompt et anticiper les lignes PLE connues, tout en gardant le tronc sériel. `prefill_dflash2` appelle effectivement append par token ; append accepte plusieurs lignes. Tester KV/ring/captures/annulation et bitparité avant tout chrono. |
| Cible CPU12ms, plus gros chantier | 01–06 | Composer un graphe stable et un segment aux entrées mutables explicites ; mesurer ce qui reste à reconstruire/encoder. Strata cache le plan mais réencode les commandes ; compilationMLX et cachePSO ne sont pas du replayMetal. |
| Filtre peu de code, mémoire à comptabiliser | 45 | Option locale RESIDENT_GATES déjà présente : qualification globale historiquement inachevée. +240Mio à inclure au footprint global sans les prendre au cache3Gio ; aucun gain complet déduit des−4,6ms de composant. |
| Conditionnel aux traces | 16/17/18/19 | Dernier expert lent, répartition dynamique des mêmes4lecteurs, contiguïtés réelles, métadonnées préparées. Localement les jobs sont bien distribués i=worker;i+=workers et le lot attend la fin ; ne pas annoncer ces mécanismes comme une économie des31ms entières. |

Points recoupés dans le worktree, lecture seule : router.rs garde la spécialisation SIMD256/top8 et le ranglegacy ; ssd_moe.rs possède déjà RESIDENT_GATES ; disk_reader.h utilise GCD et la répartition statique4workers ; flash_draft_runtime.rs appelle draft.append après chaque capture du prompt. Aucun de ces constats ne mesure une vitesse.

Pistes à laisser après les premiers filtres : 08/09/43 fusion des frontières HC/MoE avec arrondis exacts ; 11 expansion des activations seulement si une copie est effectivement matérialisée ; 35–37 sélection/KV pour contexte long ; 39–40 préfixes pour TTFT répété. Ce dernier groupe ne résout pas directement le decode court à11–11,5tok/s. Ne pas réactiver PLEcacheS1, layer-majorP01, K1, shared68 ou prévisions72/73 inchangés.

23 utilise un ancien temps inclusif éligible (~4,5ms) avec une réserve explicite. La capture fine récente montre que la plupart de l'attente route est du GPU occupé : reprendre ses fenêtres réellement idle avant un port rollbackGDN. Ne pas additionner cet ancien temps aux12ms, ni présenter12msinstrumentées comme une économie release disponible.

Ordre conservé : pause jusqu'au signal utilisateur/secteur → finir qualité et décision Sushi05 → pool/admission en replay et sélection d'un seul candidat suivant → tests numériques/bridge avant vitesse. Aucun nouveau kernel ou réglage issu de58 activé, aucune promesse15tok/s, aucune activité de mesure pendant le test utilisateur.
