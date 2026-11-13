# Implémentation GPU native

Modules Metal du moteur : mémoire et vues, projection, attention/QSA, GDN, MoE, routage, MTP, préchargement et opérations de modèle. [gpu.rs](../gpu.rs) déclare les modules, les types et la sélection des kernels ; les sources Metal sont dans [shaders/](../../shaders/README.md).

Les sous-dossiers [moe/](moe/README.md) et [qwen/](qwen/README.md) complètent les modules correspondants. Les `*_tests.rs` du dossier incluent des contrôles ciblés ; lire leurs annotations avant d’exécuter les tests GPU physiques.
