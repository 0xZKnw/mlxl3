# MoE avec experts sur SSD

[policy.rs](policy.rs) résout les politiques de lecture, préparation et résidence ; [grouped_micro.rs](grouped_micro.rs) et [row_oracle.rs](row_oracle.rs) complètent le module parent [ssd_moe.rs](../ssd_moe.rs). Les tests couvrent entre autres indices, préchargement, expert partagé et scénarios à froid. Le transport de lecture est documenté dans [ssd_reads/](../ssd_reads/README.md).
