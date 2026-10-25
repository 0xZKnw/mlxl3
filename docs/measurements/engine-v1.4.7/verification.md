# Moteur1.4.7 — publication vérifiée le16octobre2026

Source du moteur : **957b19469e2a4ed0cb788046cad8e75c100c583d**, build propre release/mlx,chat, MLX0.32.2, macOS26.2/arm64. Desktop reste1.4.6/build28 et latest ; aucune installation utilisateur.

- Intégration locale :179testsRust passés/85ignorés,442Python, fmt/Clippy strict ; troisKaniCPU79obligations/11couvertures, sansassume/stub/obligation inaccessible. Sources natives identiques au candidat Q27-19 qualifié.
- Paquet :signature ad-hoc et arm64 des troisMach-O, cinq membres exacts du TAR, tailles/hashes et manifeste vérifiés, archive reproductible, runtime-info1.4.7/commit957b propre, ciblemacOS26.2.
- Vrai installateur :installation dans un store temporaire, sélection depuis Desktop1.4.6 d'un moteur1.4.7 différent, exécution relocalisée puis rejet/fallback. L'app et le store utilisateur ne sont pas modifiés. Un défaut de type Data du harnais Swift a été corrigé, log initial conservé ; aucun code Swift de production modifié et aucune preuve formelle Swift revendiquée.
- Vrai bridge du paquet signé :sixbudgets, deuxpréfixes, deuxannulations/reprises, troisrefus, fallback ;238,35s/code0/joint. Comparaison aux références FR130/code256 avec défaut livré puis lane1 :quatre sorties exactes en IDs/texte/historique/count/compteurs,79,87s/deuxprocessus code0/joints. Ces durées sont fonctionnelles, pas un benchmark de vitesse.
- CI source957b inspectée :Linux etmacOS réussis, nouveauCPUstep layout/Tune passé ; Desktop/protocole etconversion/packing réussis. ÉtapeKaniusageAPI réussie ; suiteKani complète encore en cours au relevé de publication, sans succès global proclamé. Les commits de documentation ultérieurs peuvent déclencher une nouvelle CI ; la qualification et le tag moteur restent attachés au SHA source ci-dessus.
- Publication :tag engine-v1.4.7 au SHA exact, quatreassets distants tailles/digests vérifiés ; dernierDesktop toujours v1.4.6. Aucun nouveau débit, RAM, quantification ou qualité globale revendiqué au-delà des mesures Q27 historiques décrites dans le changelog.

Commandes : cargo fmt --all --check ; cargo clippy --locked --all-targets --features mlx,chat -- -D warnings ; cargo test --locked --release --features mlx,chat ; pytest sur les douze fichiers affectés etpackaging ; cargo kani ciblé sur les trois vrais contrats de disposition ; cargo build --release --locked --features mlx,chat depuis source propre. Les commandes exactes, environnement, délais et sorties sont dans les archives/manifestes joints.

[Release](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.7) ; [changelog](../../release-engine-v1.4.7.md) ; [validation complète](MLXL3-Engine-v1.4.7-validation.json) ; [logs et harnais](MLXL3-Engine-v1.4.7-verification.tar.gz) ; [preuve de publication](publication.json).

Limites :APIArrayKani échoue par ICE connue, MLX/Metal/concurrence et Swift non prouvés formellement ; longcontexte et autresMac non mesurés. La disposition lane1 reste OFF par défaut ; MLP2/4 automatique. Le paquet a une signature ad-hoc et aucune notarisation Apple.
