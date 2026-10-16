# Audit API / LFM — campagne du 16 octobre 2026

Cette campagne part de `c9aa43a312c63246eeb4b5a680ada26092c65a1c`, avec les
correctifs UI/mémoire/packaging locaux de `2eae717f`, sur `codex/audit-api-lfm`.
Le journal [opti.md](../../../opti.md) conserve les hypothèses, préenregistrements,
échecs et décisions. La [matrice des40 propositions](../../audit-api-lfm-status.md)
distingue ce lot des propositions déjà présentes et des pistes R&D non qualifiées.

- [API : comportements corrigés,33 tests et domaines Kani](api.md).
- [Manifeste API final inspecté](api-verification.json) et
  [archive des sources API et preuves CPU](api-evidence.tar.gz).
- [Plan physique LFM et commandes de qualification](physical-plan.md) :
  référence sérielle, fullvocab/50 états/routes, puis3 paires alternées ; un seul
  modèle/processus GPU et aucune compilation/preuve simultanée.
- [Correctifs antérieurs MTP/LFM et mémoire logique](../lfm-mtp-ui-20261016/README.md) :
  indépendants de la qualification des nouvelles optimisations.

Le [rapport du code LFM conservé](lfm-final.md) distingue les résultats physiques,
les sources/binaires, le prototype KV retiré et les campagnes vitesse encore en
cours. L'[archive physique historique](physical-evidence.tar.gz) et son
[index SHA256](physical-evidence-index.json) préservent les contrôles du prototype
et les échecs ; les preuves du code conservé seront jointes séparément.
Aucun gain de vitesse ou de qualité BF16 n'est déduit des seuls tests CPU.

Le driver existant `compare_native.py` est adapté aux 3 paires et à un budget global
optionnel ; sa parité texte/token/historique ne remplace pas les logits/états.
La revue EOF ajoute une garde du code de sortie du processus joint. Les deux faux
succès sont reproduits sur l’ancien CLI isolé ; 110 tests puis 13 cas affectés
passent sur la source finale. Voir le [rapport et les preuves versionnées](benchmark-tool.md).

La suite intégrée `scripts/check-e2e.sh` passe : 495 tests Python, 69 sautés pour
conditions ou dépendances optionnelles, puis tous les contrôles Desktop. Les
journaux détaillés et l’identité du runtime candidat sont conservés sous
`build/audit-api-lfm/full-e2e/` et `candidate-identity.json` ; ils seront archivés
avec le rapport final. Cette étape ne valide pas les variantes GPU ultérieures.

Publication autorisée par l’utilisateur après vérification : nouvelle PR, merge
lorsque les contrôles exacts sont bons, puis sous-versions Desktop et moteur avec
changelogs/empreintes distincts. Aucune installation dans `/Applications` n’est
impliquée : l’utilisateur a demandé de conserver l’app actuelle.


Les [résultats de synthèse](verification-summary.json), les [preuves du code conservé](retained-evidence.tar.gz)
et leur [index](retained-evidence-index.json) conservent les contrôles CPU, numériques
et les frontières réelles. Les [essais de vitesse interrompus](performance-evidence.tar.gz)
et leur [index](performance-evidence-index.json) restent publiés sans qualification
rétroactive. Les contrôleurs, journaux et échecs sont archivés ; les gros dumps CLI
restent locaux avec leurs SHA dans les rapports. Les originaux des inventaires
processus restent locaux ; seules les copies publiques sont expurgées.
