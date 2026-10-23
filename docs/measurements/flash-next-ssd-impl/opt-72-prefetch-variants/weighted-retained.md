# Prélecture weighted — essai retiré

Le23octobre2026, l’utilisateur a retiré cette piste après le vrai A/B contreB2sansprélecture. Il souhaitait conserver weighted parce que les premières comparaisons semblaient accélérer le meilleur mode ; ces premières paires comparaient en fait weighted à l’ancienne prélecturelegacy.

Le vrai lotHC72W-R donne MAIN11,301206→11,125227tok/s (−1,557172 %) et HOLD10,716598→8,775646 (−18,111644 %). Un seul écran ne démontre pas une régression universelle, mais aucun gain contre la meilleure référence n’est établi. La confirmationCH est annulée et n’a pas été lancée, à la demande utilisateur. Les variations ne sont pas la cause de cette nouvelle décision.

## Historique conservé

Weighted additionne surGPU les probabilités prédites des deux lignes et choisit16experts, avec un résultat contigu. Face à la prélecturelegacy, ses quatre paires étaient positives : MAIN+19,95/+16,51 %, HOLD+6,78/+12,25 %. La consigne72W-KEEP avait accepté leur variation ; elle est désormais révoquée. Ces données ne sont ni effacées ni additionnées aux résultats contre le mode sans prélecture.

## Correction et mémoire

Le [bridgeweightedv4](bridge-weighted-v4/summary.json) a passé43/43 contrôles, paritétrue,43requêtes terminées/activeNone,33completeevents/6erreurs/2annulations prévues, avec pré20/post30 etsamplerjoints sursecteur. Les112casMoE,288contrôlespréfixes/logits/captures/états et96formesShared gardent leur portée finie ; ils ne prouvent pas Metal/MLX/macOS. Le retrait est une décision de performance et d’intégration, pas un échec numérique.

Weighted était autour de7,85Go pendant ses débits ; aucun nouveauSwapoutglobal/ledgerpropreobservé. Les caps de compression77B échoués restent signalés, sans requalification ni attribution abusive des écrituresSSD au modèle.

Le [bilan7](bilan7-72S-73S.md) et le [vraiA/B72W-R](speed-72W-R/README.md) conservent tous les résultats, options et limites. Le parent retire les sourcesprototypes72/73 versHEAD89174 après archivage ; aucunepublication de gainruntimeweighted ni remplacementd’appinstallée. Les [archives de reconstruction](reconstruction/README.md) identifient les prototypes réellement mesurés et le CLIgelé69b919 ; la publication prévue est documentaire uniquement.
