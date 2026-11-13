# Expert partagé MoE sur GPU

[shared_block.rs](shared_block.rs) traite les lignes de l’expert partagé avec des poids immuables ; il ne modifie pas le routeur, le cache ou l’état. Il complète [moe.rs](../moe.rs) et [moe_block.rs](../moe_block.rs) ; les kernels correspondants sont dans [shaders/](../../../shaders/README.md).
