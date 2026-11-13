# Documentation MLXL3 — point d’entrée

Cet index couvre les documents versionnés à la racine de `docs/` et les fonds
d’archives ci-dessous. Les noms de fichiers et les chemins historiques sont
conservés : les rapports, le journal et les liens externes peuvent les citer.

## Parcours pour une IA

1. Lire [les consignes du dépôt](../AGENTS.md) et le [README du projet](../README.md).
2. Pour une optimisation, lire d’abord [le journal `opti.md`](../opti.md),
   puis les rapports et mesures qu’il référence, avant de proposer un nouvel essai.
3. Choisir une rubrique ci-dessous ; rechercher un modèle, une version, une
   issue ou une date dans les noms, puis dans le texte (`rg -n 'terme' docs/`).
4. Vérifier la date, le commit, le protocole et les statuts du rapport ; un
   résultat historique, un prototype ou une release ne décrit pas forcément
   le comportement actuel. Les résultats négatifs et les limites comptent aussi.
5. Pour les preuves brutes, suivre les liens du rapport ou utiliser les index
   d’archives. Ne pas interpréter un fichier de mesure isolé comme une validation.

## Archives et preuves

- [Index des campagnes et fichiers de mesures](measurements/README.md) : entrées
  de premier niveau, puis résultats, logs et captures propres à chaque campagne.
- [Couverture native](native-coverage/README.md) : inventaire et preuves des
  parcours natifs ; vérifier sa date et voir aussi ses sous-dossiers par composant.
- [Anciennes notes de version](releases/v0.3.2.md) : notes conservées dans
  `releases/` ; les autres notes de version sont indexées ci-dessous.

## Documents par sujet

Les documents sont rangés par sujet principal dans cet index ; un même rapport
peut concerner plusieurs sujets. Les noms incluent souvent une date ou version.

### Moteur, validation et configuration

- [engine-v1.2.1-validation.md](engine-v1.2.1-validation.md)
- [engine-v1.3.1-validation.md](engine-v1.3.1-validation.md)
- [engine-v1.4.1-validation.md](engine-v1.4.1-validation.md)
- [engine-v1.4.2-80tps.md](engine-v1.4.2-80tps.md)
- [engine-v1.4.2-pipeline.md](engine-v1.4.2-pipeline.md)
- [engine-v1.4.2-research.md](engine-v1.4.2-research.md)
- [engine-v1.4.2-upstream.md](engine-v1.4.2-upstream.md)
- [mtp-dense-1.4.0-validation.md](mtp-dense-1.4.0-validation.md)
- [v1-validation.md](v1-validation.md)
- [v2-mlx-2026-11-10.md](v2-mlx-2026-11-10.md)

### Desktop, interface et expérience utilisateur

- [app-fluidite-implementation-2026-10-16.md](app-fluidite-implementation-2026-10-16.md)
- [app-fluidite-tokens-2026-10-16.md](app-fluidite-tokens-2026-10-16.md)
- [desktop-beta6-studio-verification.md](desktop-beta6-studio-verification.md)
- [desktop-experience-20261102.md](desktop-experience-20261102.md)
- [desktop-notifications-flash-startup-20261102.md](desktop-notifications-flash-startup-20261102.md)
- [desktop-project-features-2026-10-18.md](desktop-project-features-2026-10-18.md)
- [desktop-text-size-20261103.md](desktop-text-size-20261103.md)
- [desktop-theme-startup-20261102.md](desktop-theme-startup-20261102.md)
- [desktop-updater-html-agent-verification.md](desktop-updater-html-agent-verification.md)
- [desktop-v1.1.3-validation.md](desktop-v1.1.3-validation.md)
- [desktop-v1.2.0-validation.md](desktop-v1.2.0-validation.md)
- [desktop-v1.2.1-validation.md](desktop-v1.2.1-validation.md)
- [desktop-v1.2.2-validation.md](desktop-v1.2.2-validation.md)
- [desktop-v1.3.0-validation.md](desktop-v1.3.0-validation.md)
- [desktop-v1.4.8-verification.md](desktop-v1.4.8-verification.md)
- [desktop-v1.5.0-verification.md](desktop-v1.5.0-verification.md)
- [desktop-v2-beta-review-20261113.md](desktop-v2-beta-review-20261113.md)
- [desktop-v2-beta3-verification.md](desktop-v2-beta3-verification.md)
- [desktop-v2-beta5-verification.md](desktop-v2-beta5-verification.md)
- [file-import-validation-2026-10-04.md](file-import-validation-2026-10-04.md)
- [mtp-panel-layout-20261103.md](mtp-panel-layout-20261103.md)
- [pr-desktop-experience.md](pr-desktop-experience.md)
- [pr-desktop-theme-startup.md](pr-desktop-theme-startup.md)
- [settings-redesign-20261106.md](settings-redesign-20261106.md)
- [table-breaks-2026-11-06.md](table-breaks-2026-11-06.md)

### Mémoire, démarrage et compatibilité matérielle

- [apple-silicon-stability-20261030.md](apple-silicon-stability-20261030.md)
- [cold-ready-m5-20261101.md](cold-ready-m5-20261101.md)
- [memory-saver.md](memory-saver.md)
- [model-idle-unload.md](model-idle-unload.md)
- [ram-idle-context.md](ram-idle-context.md)

### Qwen

- [adaptive-memory-qwen-20261009.md](adaptive-memory-qwen-20261009.md)
- [pr-qwen27-lossless-ram.md](pr-qwen27-lossless-ram.md)
- [qwen27-defaults-audit-20261016.md](qwen27-defaults-audit-20261016.md)
- [qwen27-m5-optimization.md](qwen27-m5-optimization.md)
- [qwen27-m5-round2.md](qwen27-m5-round2.md)
- [qwen27-max-optimisations-20261016.md](qwen27-max-optimisations-20261016.md)
- [qwen27-optimization-20261016.md](qwen27-optimization-20261016.md)
- [qwen27-ram-audit-20261016.md](qwen27-ram-audit-20261016.md)
- [qwen27-ram-implementation-20261016.md](qwen27-ram-implementation-20261016.md)
- [qwen27-smallm-m5.md](qwen27-smallm-m5.md)
- [qwen27-speed-20261008.md](qwen27-speed-20261008.md)
- [qwen27-vers-15-20261016.md](qwen27-vers-15-20261016.md)
- [qwen36-issue144-performance.md](qwen36-issue144-performance.md)
- [qwen36-max-profile-2026-11-13.md](qwen36-max-profile-2026-11-13.md)
- [qwen36-native-20261104.md](qwen36-native-20261104.md)
- [qwen38-decode-local.md](qwen38-decode-local.md)

### LFM, Ling et Gemma

- [gemma-decode-investigation.md](gemma-decode-investigation.md)
- [gemma-sdpa512-investigation.md](gemma-sdpa512-investigation.md)
- [lfm25-calibration-v2.md](lfm25-calibration-v2.md)
- [lfm25-calibration-v3.md](lfm25-calibration-v3.md)
- [lfm25-measurements-2026-10-14.md](lfm25-measurements-2026-10-14.md)
- [lfm25-moe-quantization.md](lfm25-moe-quantization.md)
- [lfm25-quality-audit-2026-10-14.md](lfm25-quality-audit-2026-10-14.md)
- [lfm25-quantization-research.md](lfm25-quantization-research.md)
- [lfm25-router38-results-2026-10-15.md](lfm25-router38-results-2026-10-15.md)
- [lfm25-v2-results-2026-10-15.md](lfm25-v2-results-2026-10-15.md)
- [lfm26-local-quantization.md](lfm26-local-quantization.md)
- [lfm3-integration-2026-11-05.md](lfm3-integration-2026-11-05.md)
- [lfm3-kernels-2026-11-06.md](lfm3-kernels-2026-11-06.md)
- [ling-local-quantization.md](ling-local-quantization.md)

### Flash, DFlash et stockage SSD

- [dflash2-exact-optimisations-2026-10-08.md](dflash2-exact-optimisations-2026-10-08.md)
- [dflash2-key-view-tps-2026-10-09.md](dflash2-key-view-tps-2026-10-09.md)
- [dflash2-key-view-tps-isolated-2026-10-10.md](dflash2-key-view-tps-isolated-2026-10-10.md)
- [dflash2-stable-validation.md](dflash2-stable-validation.md)
- [dflash2-web-optimisations-2026-10-08.md](dflash2-web-optimisations-2026-10-08.md)
- [flash-gdn-prepare-2026-10-13.md](flash-gdn-prepare-2026-10-13.md)
- [flash-hc-fusion-2026-10-13.md](flash-hc-fusion-2026-10-13.md)
- [flash-hc-gemv-2026-10-12.md](flash-hc-gemv-2026-10-12.md)
- [flash-hc-injection-2026-10-12.md](flash-hc-injection-2026-10-12.md)
- [flash-hc-pair-2026-10-13.md](flash-hc-pair-2026-10-13.md)
- [flash-kernel-cache-2026-10-12.md](flash-kernel-cache-2026-10-12.md)
- [flash-les-followup-review-2026-10-19.md](flash-les-followup-review-2026-10-19.md)
- [flash-les-gpu-review-2026-10-19.md](flash-les-gpu-review-2026-10-19.md)
- [flash-les-review-2026-10-19.md](flash-les-review-2026-10-19.md)
- [flash-les-upstream-audit-2026-10-19.md](flash-les-upstream-audit-2026-10-19.md)
- [flash-mtp-pairs-2026-10-12.md](flash-mtp-pairs-2026-10-12.md)
- [flash-next-drafts-implementation-2026-10-20.md](flash-next-drafts-implementation-2026-10-20.md)
- [flash-next-drafts-paper-audit-2026-10-20.md](flash-next-drafts-paper-audit-2026-10-20.md)
- [flash-next-ssd-audits-synthesis-2026-10-20.md](flash-next-ssd-audits-synthesis-2026-10-20.md)
- [flash-next-ssd-cache-width-64-2026-10-21.md](flash-next-ssd-cache-width-64-2026-10-21.md)
- [flash-next-ssd-dflash-splash-60-2026-10-21.md](flash-next-ssd-dflash-splash-60-2026-10-21.md)
- [flash-next-ssd-early-hot-64-2026-10-21.md](flash-next-ssd-early-hot-64-2026-10-21.md)
- [flash-next-ssd-fixed-width-66-2026-10-21.md](flash-next-ssd-fixed-width-66-2026-10-21.md)
- [flash-next-ssd-grouped-exl3-65-2026-10-21.md](flash-next-ssd-grouped-exl3-65-2026-10-21.md)
- [flash-next-ssd-hot-demand-64-2026-10-21.md](flash-next-ssd-hot-demand-64-2026-10-21.md)
- [flash-next-ssd-hot-prefetch-review-69-2026-10-22.md](flash-next-ssd-hot-prefetch-review-69-2026-10-22.md)
- [flash-next-ssd-implementation.md](flash-next-ssd-implementation.md)
- [flash-next-ssd-prefetch-precap-69-70-2026-10-22.md](flash-next-ssd-prefetch-precap-69-70-2026-10-22.md)
- [flash-next-ssd-profile-53-2026-10-20.md](flash-next-ssd-profile-53-2026-10-20.md)
- [flash-next-ssd-roadmap-54-55-2026-10-20.md](flash-next-ssd-roadmap-54-55-2026-10-20.md)
- [flash-next-ssd-shared-expert-68-2026-10-22.md](flash-next-ssd-shared-expert-68-2026-10-22.md)
- [flash-next-ssd-splash-61-audit-2026-10-21.md](flash-next-ssd-splash-61-audit-2026-10-21.md)
- [flash-next-ssd-splash-64-research-2026-10-21.md](flash-next-ssd-splash-64-research-2026-10-21.md)
- [flash-next-ssd-splash-strata-followup-70-2026-10-22.md](flash-next-ssd-splash-strata-followup-70-2026-10-22.md)
- [flash-next-ssd-stable-kernels-67-2026-10-22.md](flash-next-ssd-stable-kernels-67-2026-10-22.md)
- [flash-next-ssd-strata-56-2026-10-20.md](flash-next-ssd-strata-56-2026-10-20.md)
- [flash-next-ssd-strata-splash-59-2026-10-21.md](flash-next-ssd-strata-splash-59-2026-10-21.md)
- [flash-next-ssd-unique-cohorts-64-2026-10-21.md](flash-next-ssd-unique-cohorts-64-2026-10-21.md)
- [flash-next-ssd-verifier-63-2026-10-21.md](flash-next-ssd-verifier-63-2026-10-21.md)
- [flash-next-ssd-verifier-63-design-2026-10-21.md](flash-next-ssd-verifier-63-design-2026-10-21.md)
- [flash-next-ssd-verifier-63-splash-map-2026-10-21.md](flash-next-ssd-verifier-63-splash-map-2026-10-21.md)
- [flash-next-ssd-verifier-profile-64-2026-10-21.md](flash-next-ssd-verifier-profile-64-2026-10-21.md)
- [flash-next-ssd-wide-hot-62-2026-10-21.md](flash-next-ssd-wide-hot-62-2026-10-21.md)
- [flash-profile-2026-10-12.md](flash-profile-2026-10-12.md)
- [flash-profile-after-hc-2026-10-12.md](flash-profile-after-hc-2026-10-12.md)
- [flash-shared-overlap-2026-10-13.md](flash-shared-overlap-2026-10-13.md)
- [flash-ssd-readers-2026-10-12.md](flash-ssd-readers-2026-10-12.md)
- [flash-ssd-upstream-2026-10-12.md](flash-ssd-upstream-2026-10-12.md)
- [flash-top10-router-2026-10-13.md](flash-top10-router-2026-10-13.md)
- [issue46-flash-architecture.md](issue46-flash-architecture.md)
- [recherche-flash-next-ssd-205-2026-10-11.md](recherche-flash-next-ssd-205-2026-10-11.md)

### Performance, kernels et audits transversaux

- [audit-api-lfm-status.md](audit-api-lfm-status.md)
- [audit-bridge-prefill-2026-09-22.md](audit-bridge-prefill-2026-09-22.md)
- [audit-decode-prefill-2026-09-23.md](audit-decode-prefill-2026-09-23.md)
- [audit-runtime-2026-09-04.md](audit-runtime-2026-09-04.md)
- [audit-v1-2026-09-06.md](audit-v1-2026-09-06.md)
- [decode-investigation-2026-09-10.md](decode-investigation-2026-09-10.md)
- [decode-profile-p09-20261101.md](decode-profile-p09-20261101.md)
- [decode-roadmap-2026-09-04.md](decode-roadmap-2026-09-04.md)
- [general-performance-2026-09-07.md](general-performance-2026-09-07.md)
- [issue52-throughput.md](issue52-throughput.md)
- [kernel-port.md](kernel-port.md)
- [metal-quantization-optimization.md](metal-quantization-optimization.md)
- [prefill-investigation-2026-09-11.md](prefill-investigation-2026-09-11.md)
- [research-app-engine-20261009.md](research-app-engine-20261009.md)
- [tool-prefill-rd-2026-09-05.md](tool-prefill-rd-2026-09-05.md)
- [tune-feedback-20261010.md](tune-feedback-20261010.md)

### Audits, revues et suivi des issues

- [issue162-download-integrity.md](issue162-download-integrity.md)
- [issue163-agent-reconstruction.md](issue163-agent-reconstruction.md)
- [issue163-buffer-api-reconstruction.md](issue163-buffer-api-reconstruction.md)
- [issue163-ci-followup.md](issue163-ci-followup.md)
- [issue163-converter-reconstruction.md](issue163-converter-reconstruction.md)
- [issue163-coverage-matrix.md](issue163-coverage-matrix.md)
- [issue163-desktop-runtime-reconstruction.md](issue163-desktop-runtime-reconstruction.md)
- [issue163-download-recovery.md](issue163-download-recovery.md)
- [issue163-draft-gpu-followup.md](issue163-draft-gpu-followup.md)
- [issue163-draft-storage-reconstruction.md](issue163-draft-storage-reconstruction.md)
- [issue163-functional-download.md](issue163-functional-download.md)
- [issue163-functional-native.md](issue163-functional-native.md)
- [issue163-functional-structured.md](issue163-functional-structured.md)
- [issue163-gpu-reconstruction.md](issue163-gpu-reconstruction.md)
- [issue163-incremental-lexer.md](issue163-incremental-lexer.md)
- [issue163-metal-recovery.md](issue163-metal-recovery.md)
- [issue163-native-reconstruction.md](issue163-native-reconstruction.md)
- [issue163-physical-io-cancellation.md](issue163-physical-io-cancellation.md)
- [issue163-protocol-reconstruction.md](issue163-protocol-reconstruction.md)
- [issue163-recovery-hub.md](issue163-recovery-hub.md)
- [issue163-recovery-ui.md](issue163-recovery-ui.md)
- [issue163-recovery.md](issue163-recovery.md)
- [issue163-responsiveness-completion.md](issue163-responsiveness-completion.md)
- [issue163-responsiveness-reconstruction.md](issue163-responsiveness-reconstruction.md)
- [issue164-native-progress.md](issue164-native-progress.md)
- [pr161-fixes-2026-11-08.md](pr161-fixes-2026-11-08.md)
- [pr161-regolith-audit-2026-11-08.md](pr161-regolith-audit-2026-11-08.md)
- [pr161-upstream-r4-2026-11-08.md](pr161-upstream-r4-2026-11-08.md)
- [pr72-cloud-integration-20261103.md](pr72-cloud-integration-20261103.md)
- [review-pr23-2026-10-05.md](review-pr23-2026-10-05.md)

### Versions du moteur et validations de release

- [release-engine-v1.2.1.md](release-engine-v1.2.1.md)
- [release-engine-v1.3.0.md](release-engine-v1.3.0.md)
- [release-engine-v1.3.1.md](release-engine-v1.3.1.md)
- [release-engine-v1.4.0.md](release-engine-v1.4.0.md)
- [release-engine-v1.4.1.md](release-engine-v1.4.1.md)
- [release-engine-v1.4.2.md](release-engine-v1.4.2.md)
- [release-engine-v1.4.3.md](release-engine-v1.4.3.md)
- [release-engine-v1.4.4.md](release-engine-v1.4.4.md)
- [release-engine-v1.4.5.md](release-engine-v1.4.5.md)
- [release-engine-v1.4.6.md](release-engine-v1.4.6.md)
- [release-engine-v1.4.7.md](release-engine-v1.4.7.md)
- [release-engine-v1.4.8.md](release-engine-v1.4.8.md)
- [release-engine-v2.0.0-beta.1.md](release-engine-v2.0.0-beta.1.md)
- [release-engine-v2.0.0-beta.2.md](release-engine-v2.0.0-beta.2.md)
- [release-engine-v2.0.2-beta.1.md](release-engine-v2.0.2-beta.1.md)
- [release-engine-v2.0.3-beta.1.md](release-engine-v2.0.3-beta.1.md)
- [release-engine-v2.0.4-beta.1.md](release-engine-v2.0.4-beta.1.md)
- [release-engine-v2.0.5-beta.1.md](release-engine-v2.0.5-beta.1.md)
- [release-engine-v2.0.6-beta.1-validation.md](release-engine-v2.0.6-beta.1-validation.md)
- [release-engine-v2.0.6-beta.1.md](release-engine-v2.0.6-beta.1.md)
- [release-engine-v2.0.7-beta.1-validation.md](release-engine-v2.0.7-beta.1-validation.md)
- [release-engine-v2.0.7-beta.1.md](release-engine-v2.0.7-beta.1.md)
- [release-engine-v2.0.8-beta.1.md](release-engine-v2.0.8-beta.1.md)
- [release-engine-v2.0.9-beta.1-validation.md](release-engine-v2.0.9-beta.1-validation.md)
- [release-engine-v2.0.9-beta.1.md](release-engine-v2.0.9-beta.1.md)

### Versions Desktop et validations de release

- [release-v0.3.1.md](release-v0.3.1.md)
- [release-v0.4.0.md](release-v0.4.0.md)
- [release-v0.4.1.md](release-v0.4.1.md)
- [release-v0.4.2.md](release-v0.4.2.md)
- [release-v0.4.5.md](release-v0.4.5.md)
- [release-v0.4.6.md](release-v0.4.6.md)
- [release-v1.0.0.md](release-v1.0.0.md)
- [release-v1.0.1.md](release-v1.0.1.md)
- [release-v1.0.2.md](release-v1.0.2.md)
- [release-v1.0.3.md](release-v1.0.3.md)
- [release-v1.1.0.md](release-v1.1.0.md)
- [release-v1.1.1.md](release-v1.1.1.md)
- [release-v1.1.2.md](release-v1.1.2.md)
- [release-v1.1.3.md](release-v1.1.3.md)
- [release-v1.2.0.md](release-v1.2.0.md)
- [release-v1.2.1.md](release-v1.2.1.md)
- [release-v1.2.2.md](release-v1.2.2.md)
- [release-v1.3.0.md](release-v1.3.0.md)
- [release-v1.4.0.md](release-v1.4.0.md)
- [release-v1.4.2-validation.md](release-v1.4.2-validation.md)
- [release-v1.4.2.md](release-v1.4.2.md)
- [release-v1.4.3-validation.md](release-v1.4.3-validation.md)
- [release-v1.4.3.md](release-v1.4.3.md)
- [release-v1.4.4.md](release-v1.4.4.md)
- [release-v1.4.5.md](release-v1.4.5.md)
- [release-v1.4.6.md](release-v1.4.6.md)
- [release-v1.4.8.md](release-v1.4.8.md)
- [release-v1.5.0.md](release-v1.5.0.md)
- [release-v2.0.0-beta.1.md](release-v2.0.0-beta.1.md)
- [release-v2.0.0-beta.10.md](release-v2.0.0-beta.10.md)
- [release-v2.0.0-beta.11.md](release-v2.0.0-beta.11.md)
- [release-v2.0.0-beta.12.md](release-v2.0.0-beta.12.md)
- [release-v2.0.0-beta.13.md](release-v2.0.0-beta.13.md)
- [release-v2.0.0-beta.14.md](release-v2.0.0-beta.14.md)
- [release-v2.0.0-beta.15.md](release-v2.0.0-beta.15.md)
- [release-v2.0.0-beta.16.md](release-v2.0.0-beta.16.md)
- [release-v2.0.0-beta.17.md](release-v2.0.0-beta.17.md)
- [release-v2.0.0-beta.18.md](release-v2.0.0-beta.18.md)
- [release-v2.0.0-beta.19.md](release-v2.0.0-beta.19.md)
- [release-v2.0.0-beta.2.md](release-v2.0.0-beta.2.md)
- [release-v2.0.0-beta.20.md](release-v2.0.0-beta.20.md)
- [release-v2.0.0-beta.21.md](release-v2.0.0-beta.21.md)
- [release-v2.0.0-beta.3.md](release-v2.0.0-beta.3.md)
- [release-v2.0.0-beta.4.md](release-v2.0.0-beta.4.md)
- [release-v2.0.0-beta.5.md](release-v2.0.0-beta.5.md)
- [release-v2.0.0-beta.6.md](release-v2.0.0-beta.6.md)
- [release-v2.0.0-beta.8.md](release-v2.0.0-beta.8.md)
- [release-v2.0.0-beta.9.md](release-v2.0.0-beta.9.md)
- [release-validation-v2.0.0-beta.11.md](release-validation-v2.0.0-beta.11.md)

### Autres sujets

- [discord-release-bot.md](discord-release-bot.md)
- [native-runtime-readme-before-audit-2026-10-16.md](native-runtime-readme-before-audit-2026-10-16.md)

## Entretien de l’index

Ajouter tout nouveau document racine à la rubrique la plus pertinente et toute
nouvelle entrée de premier niveau de `measurements/` à son index. Garder les
chemins existants, les données brutes et leurs limites ; ne pas déplacer ou
effacer des preuves sans traiter toutes leurs références et usages.
