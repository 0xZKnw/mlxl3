# Qwen27 small-M: guide de revue

Cette branche fournit des outils de benchmark et les preuves de la campagne
M5 des 6–7 octobre 2026. **Elle n'active aucun nouveau kernel dans le moteur.**
Les résultats MB3 et GDN permettent de poursuivre une recherche ; ils ne
démontrent pas une accélération stable des générations longues.

## Parcours de revue

1. Lire le [rapport et ses décisions](../../qwen27-smallm-m5.md).
2. Consulter les [80 géométries EXL3 et 21 lignes GDN](inventory-table.md).
3. Examiner les cinq drivers `benchmarks/benchmark_smallm*.py`, leurs tests
   dans `tests/test_smallm_benchmark.py` et leur couverture dans le workflow
   `.github/workflows/rust.yml`. Le correctif de `tests/test_qmm_tiles.py`
   concerne uniquement le délai de démarrage des codecs de test.
4. Utiliser les [commandes et options originales](protocol.md), puis les
   [empreintes des artefacts](artifact-index.json) et la
   [provenance modèle/source](campaign-identity.json) pour auditer les preuves.

Les sorties machine JSON, logs et stderr sont repliées par défaut dans la
revue GitHub ; elles sont conservées en entier. Le protocole, les tableaux,
le rapport et le patch MB3 restent visibles. Les espaces et nouvelles lignes
des logs bruts sont intentionnellement préservés pour garder leurs empreintes.

## Ce que les artefacts établissent

| Ensemble | Résultat | Limite à conserver dans la PR |
|---|---|---|
| `grouped-mb3-{screen,confirm}.json` | Parité des cas micro, deux fenêtres de microkernels positives | Chaîne de projections avec feedback synthétique ; pas de tokens/s modèle |
| `mb3-{rollback,recursive}.log` | Contrôles physiques exacts du prototype natif | Trajectoires testées, pas une preuve de toutes les entrées |
| `mb3-verify-isolated.log`, `mb3-verify-summary.json` | Verify court apparié, +4,70 % observé | Dérive du contrôle ; gain soutenu non établi |
| `mb3-bridge-256.json` | Quatre sorties complètes identiques, puis deadline | Campagne incomplète : `failed`, `parity:false` |
| `mb3-bridge-ac-256.json`, `mb3-bridge-ac-summary.json` | Huit sorties complètes identiques | Alimentation variable et contrôle instable ; pas de promotion |
| `gdn-columns-screen.json` | 162 cas FP16 exacts | Filtre micro seulement, pas d'intégration modèle |
| `tensor-bm16-screen.json` | 598 mots FP16 finaux divergents dès M2 | Rejet ; aucun timing après divergence |
| `inventory-timings.json` | 80 géométries, oracle M1/NT1 exact et fini | Les deux séries chronométrées utilisent le même contrôle ; leurs ratios ne sont pas des gains |
| `inventory-initial-sg-diagnostic.json` | Premier inventaire conservé | SG incorrect pour K2 separate ; remplacé par l'inventaire final, sans effacer l'original |
| `metal-capture.json` | Capture stock M3 produite localement | Capture non analysée ; compteurs et temps GPU pur non mesurés |
| `kani-{grouped,all}.log` | Sélecteur Rust du prototype vérifié, suite CPU 35/35 | Ne prouve ni Metal, ni MLX/FFI, ni la génération complète |
| `final-python-tests-complete.log` | 258 tests Python passés | CrossHair absent ; aucune preuve formelle Python revendiquée |
| `final-native-tests.log` | 62 tests Rust passés, 55 ignorés | Les tests ignorés restent non exécutés dans cette suite |

Les logs rouges, les premières validations et les tentatives interrompues
restent disponibles. `artifact-index.json` décrit les preuves de campagne
originales ; ce guide de revue, ajouté ensuite, n'en fait pas partie.

## Reproduire sans dépendre des chemins de l'auteur

Utiliser Python 3.12 et, pour les mesures physiques, MLX 0.32.2 sur un Mac
Apple Silicon. Le workflow fixe les dépendances des contrôles CPU ; le
rapport distingue les versions locales réellement utilisées. Remplacer les
chemins modèle, tête et sortie par les siens, avec un nouveau dossier de
résultats : les drivers refusent d'écraser un rapport existant.

Exemple d'inventaire, sans lancer de mesure GPU :

```sh
python benchmarks/benchmark_smallm.py \
  --checkpoint /chemin/vers/Qwen3.8-27B-exl3 \
  --inventory-only \
  --output /chemin/vers/nouveaux-resultats/inventory.json
```

Les essais GPU doivent suivre le protocole d'isolation du rapport : un seul
modèle, aucun build/proveur/test concurrent, warmups, ordre alterné et
conditions d'alimentation relevées. Une nouvelle campagne doit conserver
les résultats précédents et justifier son protocole dans `opti.md`.

Le [patch MB3 archivé](mb3-prototype.patch) est fourni pour examen et
reconstruction explicite dans un checkout distinct. Pour le reconstruire,
partir de la base `5de518e0fbe7d3367416b9bc5999c7bd1f92c74d`, puis appliquer
le patch. Son flag reste OFF par défaut. Le driver bridge nécessite ce
prototype : le moteur final de cette branche n'implémente pas le flag MB3,
et le driver ne détecte pas à lui seul la prise en charge du flag par un
binaire arbitraire. Le binaire et le `.gputrace` historiques restent hors Git.

Les sources de la campagne étaient celles validées au commit `25e1fef` ;
leurs empreintes et résultats restent conservés tels quels. Les
[correctifs des validateurs PR27](../pr27-fixes/verification-summary.json)
vérifient désormais la finitude de chaque étape dépendante et les trois
compteurs MTP entre variantes. Les nouveaux fingerprints ont sept entrées,
les quatre premières gardant le format historique. Ces correctifs ne
constituent pas une nouvelle mesure des performances de la campagne.
L'absence de push/PR décrite dans les preuves correspond à sa clôture
historique ; elle ne préjuge pas d'un envoi ultérieur de la branche.
