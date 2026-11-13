# État et MTP Qwen sur GPU

[kv.rs](kv.rs) gère le stockage d’attention avec préfixe initialisé ; [mtp.rs](mtp.rs) complète le chemin MTP. Ces modules sont raccordés par [qwen.rs](../qwen.rs) et s’utilisent avec les configurations et états du modèle dans [native/src/](../../README.md).
