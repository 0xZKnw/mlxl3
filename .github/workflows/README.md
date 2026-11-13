# Workflows GitHub Actions

- `check.yml` : régressions Python, inventaire natif et contrôles Desktop/protocole.
- `rust.yml` : tests et lint Rust, contrôles Metal et harnais Kani répartis par lots.
- `apple-compatibility.yml` : contrats de compatibilité Apple.
- `profile.yml` : compatibilité des outils de profilage Flash.
- `quantization.yml` : contrats de conversion et quantification.
- `discord-release.yml` : notification liée aux releases.

Consulter chaque fichier pour ses déclencheurs, environnements, versions et jobs exacts. Les contrôles hébergés ne remplacent pas la validation numérique sur GPU physique.
