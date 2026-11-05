# 82F05 — plans préparés, non exécutés

25 octobre2026. Écriture limitée à ce dossier ROOT/plans. La source F05 intégrée actuelle a été relue, ainsi que opti et les plans F06 qualitév2. Aucun build, preuve, test natif ou GPU n’a été lancé par cet auteur. Les résultats CPU rapportés par le parent restent distincts de cette préparation. Les SHA source/CLI/tests sont null jusqu’au gel du parent.

La base native est80M, avec les contrôleurs82B publiés et F05 isolé. F01 et F06 ont été retirés ; leurs variables MLXL3_SSD_QMV_HALF_OUTPUT et MLXL3_SSD_TOP10_EARLY_EXIT sont absentes du JSON et supprimées explicitement dans chaque enfant par /usr/bin/env -u. Même protection pour80Q/80G. Ne pas recopier les anciens binaires F06 ou le plan qualitév1 malformé. Le CLI et les tests normaux utilisent les features par défaut plus mlx,chat (direct-metal conservé), jamais --no-default-features. Le flag F05 est MLXL3_SSD_BATCH_ROUTES : H=0, C=1, défaut0.

## Gel avant exécution

Remplir identity-template.json et gates-before-gpu.json après inspection des sources, CLI, tests et options exactes ; rehasher les pilotes, plans et golden60. Les champs SHA nuls empêchent une qualification reproductible, mais le runner stock ne les refuse pas automatiquement : le contrôle manuel du parent est obligatoire. Vérifier les noms via --list, une occurrence de chaque sélecteur exact, arrêter/join compilateurs et proveurs avant GPU. commands.json propose les commandes ; aucune n’est lancée automatiquement.

Les garde-fous45/49 restent inchangés : secteur stable, un modèleGPU, aucune compilation/proof concurrente, pré20/post30, délais bornés, enfants/samplers joints. Conserver les échecs partiels, run-result/logs/system-samples/vm-curve ; ne jamais écraser une campagne existante.

## Qualité :19 phases

Pré20, suite MLX non ignorée, CLI externe byte0xff, quatre composants exacts, neuf loaders natifs séparés, préfixes DFlash, sessionMTP et post30. Cela fait15 invocations natives exactes plus une suite globale, un CLI et deux attentes. La suite globale ne porte PAS native_test : son résumé doit être inspecté et montrer un nombre non nul de tests, au lieu d’imposer running1test. Chaque invocation exacte porte native_test et exige un test réussi, pas zéro sélection.

Les quatre composants exigent28 bords,1004 gates/404 branches rapides sur48 couches,24 MoE dont seulement4 contrôles Hot avant readback, puis32 cas U historiques (15 formes réelles+16 fallback+1 récupérationI/O). Les marqueurs ne généralisent pas la chronologie aux24 MoE. Les bits non finis des28 cas sont un contrat de réduction intentionnel, aucune autorisation de sorties de génération non finies. Les logits/probabilités/IDs/poids, formes, finitude des gates réelles et sorties MoE doivent respecter les assertions de la source.

Les neuf loaders ont chacun un environnement fixé avant Popen : absence,0,1, puis six invalides ASCII (vide,2,01,true,espace1,1\n). Le test natif vérifie lui-même le refus des invalides et retourne Ok : TOUS attendent exit0 et 1passed. Aucun set_var/remove_var après initialisationMLX. Le loadonce est couvert par champ stocké/câblage et revue source, pas une prétendue mutation dynamique sûre.

Le CLI non-Unicode est distinct : JSON U+DCFF, Python POSIXsurrogateescape en byte0xff, exit2 avec environment variable was not valid unicode. Ce cas externe et les six invalides natifs ne sont pas interchangeables. Une erreur de lancement doit rester un échec explicite.

Après les composants :36 préfixes/captures/logits/états DFlash byte exacts et sessionMTP aux budgets1/2/3/17 avec acceptation et rejet non vides. La suite complète est celle de cette intégrationF05 ; aucune répétition F01/F06/EXL3 inchangée n’est ajoutée pour gonfler le compte. Les mutations GPU sont orchestrées séparément par le parent, détectées sur binaires jetables et restaurées avant ce gel normal.

## Bridge et débit

bridge : flag1, ADAPTIVE1, WIDTH absent du JSON ET supprimé de l’environnement hérité. Exiger checks.json status=complete, parity=true, checks=43, annulation/reprise/erreurs attendues et tous enfants/samplers joints. check_dflash2/JsonProcess contrôle les requêtes mais ne certifie pas tous les événements tardifs de fermeture : cette limite reste visible.

screen : MAIN H→C puis HOLD H→C. confirm : MAIN C→H puis HOLD C→H, conditionnel au signal et à la décision parent. Même CLI immuable, seul F05 change. B2 fixe WIDTH1/ADAPTIVE0, Q4/ring,128warm+128mesure/127decode, cache3Gio/53slots, alloc64Mio, Hot/Wide/63/UNIQUE1, prélecture/LES/shared/stable0. Les résultatsF01/F06 ne s’additionnent pas àF05. OFF/MTP1 frais requis avant promotion, pas une nouvelle campagne automatique ici.

Inspecter validated/reaped/reader_joined et les réponses non vides, texte+hash golden60,128/127, modes, acceptations, largeurs et octets SSD logiques. Ces octets ne sont pas les lectures physiques NVMe. Conserver chaque paire, dérive warm→mesure et effet d’ordre ; aucune causalité matérielle sans mesure.

Mémoire : footprintprocess≤12Go, entrées/sorties système séparées du ledger/logicalcompressed/decompressions privés. Viser zéro nouvelle sortie système ; la garde stock256Mio reste un garde d’arrêt, pas un certificat zéro. Les capscompression77B historiques restentFAILED s’ils sont dépassés ; ledger0 ne prouve pas l’absence d’écritures swap propres. RSSRUSAGE_CHILDREN n’est pas le footprint. Aucune accélération, temps GPU exclusif ou réduction mémoire n’est validée par la cohérence statique de ces plans.
