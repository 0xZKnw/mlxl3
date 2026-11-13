# Données de test

- `image-input/` : images réelles et cas d’erreur utilisés par les contrôles d’import Desktop.
- `native-converter-host/` : données de référence pour les transformations côté hôte.
- `native-feedback/` : corpus de produits et recherches accompagné de son manifeste et de ses empreintes.
- `native-quantizer/` : tuiles et oracles indépendants de quantification et d’empaquetage.
- Les fichiers à cette racine servent notamment aux contrats de cache, à Kani et à l’affichage des cartes de modèles.

Ces données sont des références de test : ne pas modifier les contenus ou empreintes sans mettre à jour et requalifier les oracles concernés. Les tests correspondants sont dans [tests/](../README.md).
