# Interface C++ et pont MLX

`mlx_bridge.cpp` relie le moteur Rust aux opérations MLX ; les en-têtes voisins portent notamment les contrats de lecture disque, files de travail, cache de kernels, capacités Metal et mémoire de processus. `process_stats.cpp` complète les statistiques de processus. Les déclarations et appels Rust se trouvent sous [native/src/](../src/README.md), avec des contrôles C++ dans [native/tests/](../tests/README.md).
