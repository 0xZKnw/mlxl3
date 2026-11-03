# Preuves de conversion et mesures LFM2.5 du 14 octobre

Les rapports JSON restent directement lisibles. Les logs, stderr/stdout,
événements NDJSON et l'extrait MSL diagnostique sont conservés **octet pour
octet** dans [raw-evidence.tar.gz](raw-evidence.tar.gz).
[L'index](raw-evidence-index.json) fournit le nom, la taille et le SHA-256
original de chaque entrée, ainsi que le SHA de l'archive. La décompression
et chaque hash ont été vérifiés avant l'ajout à Git.

Lire un journal sans extraire tout le dossier :

```sh
tar -xOf raw-evidence.tar.gz biased-router-kani.log
tar -xOf raw-evidence.tar.gz gguf-paired.stderr
```

Les copies non compressées sont aussi conservées localement dans
`/Users/justin/Documents/mix-stq1_0/build/lfm25-eval/raw-evidence` et dans
ce dossier ignoré. L'archivage évite de modifier les espaces/carriage returns
des sorties originales pour satisfaire un contrôle de whitespace.
L'extrait MSL est le corps utilisé par le kernel ; la tentative CBMC refuse
son format, aucune preuve formelle de GPU n'est annoncée.

La synthèse de conversion de 17 Mo est conservée séparément dans
[conversion-pipeline_summary.json.gz](conversion-pipeline_summary.json.gz),
avec [manifeste et empreintes](conversion-summary-archive.json).
Son original reste dans le work-dir de conversion. Les modèles, le backend
GGUF téléchargé et les matrices/dump de logits restent exclusivement dans
les dossiers locaux `models`/`build`, sans ajout à Git.

Protocoles, résultats et limites : [rapport](../../lfm25-measurements-2026-10-14.md),
essais14/15/16 de [opti.md](../../../opti.md).
