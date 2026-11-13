# Compléments du modèle Qwen Flash

[memory.rs](memory.rs) applique les politiques mémoire aux frontières de chargement et de requête ; [profile.rs](profile.rs) fournit le profilage associé. Les fichiers d’oracle et de test vérifient notamment GDN, HC, convolution préparée et prefill MTP. Le modèle principal est dans [qwen4.rs](../qwen4.rs).
