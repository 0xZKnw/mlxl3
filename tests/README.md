# Tests et contrôles de parcours

- `test_*.py` : régressions Python sur le moteur, la conversion, les scripts et les rapports ; certains nécessitent des dépendances optionnelles.
- `*-check.swift` : contrôles Desktop et interface ; `fake-*-engine.py` fournit des moteurs simulés à certains parcours.
- `physical_*.py` : harnais destinés à des contrôles explicites sur matériel et modèles compatibles ; leur présence ne signifie pas qu’ils sont exécutés par défaut.
- `*.rs` et `quantize_packing_proof.cpp` : contrats et preuves ciblés côté natif.
- [fixtures/](fixtures/README.md) : corpus et données de test conservés.

Les commandes réellement utilisées sont dans [check.yml](../.github/workflows/check.yml), [rust.yml](../.github/workflows/rust.yml) et [scripts/check-e2e.sh](../scripts/check-e2e.sh). La suite E2E lance des tests Python puis Desktop ; consulter ses dépendances et les conditions du matériel avant de l’exécuter. Les tests ignorés restent non exécutés.
