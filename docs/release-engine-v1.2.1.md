# MLXL3 Engine v1.2.1

Le moteur accélère l'entretien du cache de la tête MTP de Qwen3.6-35B-A3B. Lors du préfill et après une proposition acceptée, il calcule les clés/valeurs nécessaires à la prochaine prédiction et évite les sorties d'attention, MoE et projection finale immédiatement jetées.

- Mesure locale M5 24 Gio, Qwen3.6-35B-A3B EXL3 2,49 bpw avec tête MTP 4 bits, greedy : **57,22 → 63,89 tokens/s**, soit environ **+11,65 %**, sur le prompt de calcul matriciel. Comparaison A/B/B/A, deux répétitions par passe après warmup, 128 tokens. Les résultats court et long sont non concluants au seuil préenregistré ; ce chiffre ne constitue pas un gain général.
- Les tokens, textes et compteurs d'acceptation restent exacts dans les cas testés. Contrôles physiques du cache K/V, de la prédiction suivante, de la référence MLX-LM et du bridge : budgets courts, préfixes, annulation/récupération et pénalité.
- 54 tests Rust et 26 harnais Kani réussis localement avant préparation de la release. Kani vérifie notamment les positions i32 du cache et les refus de lignes vides/dépassements ; il ne prouve pas le GPU ni le FFI. Les tests GPU ignorés hors matériel physique et les autres architectures restent hors de cette validation.

Cette mise à jour du **moteur** est compatible avec **MLXL3 Desktop 1.2.0**, Apple Silicon et macOS 26.2 ou supérieur. Elle conserve le protocole bridge 1 et MLX 0.32.2. L'archive contient le runtime et ses empreintes, sans poids de modèles. Signature ad hoc, comme la version précédente.

Dans l'app, rechercher les mises à jour puis installer la mise à jour du moteur. La version de Desktop reste 1.2.0.

Les mesures et leurs limites sont conservées dans [le journal](../opti.md), [la décision MTP-03](measurements/mtp-03-decision.json) et [les mesures brutes](measurements/mtp-03-summary.json).
