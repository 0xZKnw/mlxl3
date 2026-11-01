# Audit de l’adaptation P2

Le ralentissement combiné P2 ne suffit pas à rejeter les deux changements. Le diagnostic P1 a mesuré des fenêtres hôte, pas un calendrier GPU corrélé. Une soumission W1 après fin des lectures peut encore recouvrir la construction CPU de W2 et du résultat final. Qualifier toute soumission tardive d’inutile était une erreur d’interprétation.

Les variantes U et D diffèrent uniquement par les deux constantes privées PREPARE_COLD_REST et DEFER_COMPLETE_COLD. Les archives du fichier ssd_moe.rs le confirment. U conserve la soumission W1 et supprime le snapshot supplémentaire en release ; D garde la préparation W2 après join et ajoute seulement la décision de différer. Les binaires, archives, options, modèles et contrôleurs sont figés dans les identités par bras.

Chemin et limites vérifiés : B1/B2, Top10 canonique, doublons préservés, 3 Gio de banques partagées, quatre lecteurs, aucun buffer d’experts supplémentaire ; mêmes fallbacks pour les autres formes et options. La construction W2 ne doit pas évaluer les banques avant join. Le test retient une lecture, vérifie la non-disponibilité du résultat et injecte un échec après construction. Le profil fenced reste après join. Les consommateurs Hot/W1 restent retenus et drainés sur erreur avant invalidation.

Les 14 sélections physiques U/D ont passé. La sélection D ne revendique pas le cas de préparation anticipée qu’elle désactive. MTP et DFlash vérifient aussi les états/logits ; DFlash couvre 36 préfixes, largeurs 1–8 et 64 000 valeurs capturées. La mutation historique désactivant W2 contre une politique de préparation active est détectée. Cela reste une validation dynamique, pas une preuve de sûreté générale MLX/Metal.

Risques de performance : déplacer la construction W2 peut déplacer des allocations et de la contention CPU ; supprimer async_eval peut déplacer du travail GPU hors d’une fenêtre utile. Les estimations antérieures de 0,395 et 0,105 ms/transition ne sont pas des gains réalisables garantis et ne s’additionnent pas. Le crible AR P/U/D/D/U/P doit décider, sans mélanger les chronos de campagnes différentes.
