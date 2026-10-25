# Rendu LaTeX des réponses — 16 octobre 2026

La capture utilisateur contient des blocs Markdown mathématiques reconnus, mais
le moteur SwiftMath embarqué rejette deux commandes : `\dots` et `\boxed`.
Le diagnostic contre l’objet compilé précédent reproduit `Invalid command \dots`
puis `Invalid command \boxed`. Les fractions abrégées et `\ldots` sont acceptées.

Le correctif ajoute l’alias points de suspension et un cadre au conteneur
mathématique existant, avec copie, sérialisation, scripts et marges. Le fallback
conserve le texte lorsqu’une autre commande n’est pas prise en charge. Les petites
images sont centrées sans couper un cadre vide. SwiftMath reste un sous-ensemble
de TeX, pas un compilateur LaTeX complet.

Portée : parseur réel → Markdown → cache d’image → rendu AppKit. Régressions de neuf
formules dans deux styles, pixels non vides, quatre côtés d’un cadre vide, centre
transparent, copies/roundtrip, erreurs et borne 4096 caractères. Le worker et le
parcours réponse existants restent exercés.

État : code local validé par les contrôles CPU/bitmap et le parcours de réponse.
Ancien objet compilé : test rouge (exit−5), puis neuf formules/deux styles,
huit combinaisons de délimiteurs, copies/roundtrip, pixels et erreurs passent.
La première suite Desktop s’est arrêtée sur le constructeur interne utilisé dans
le nouveau test ; le harnais a été corrigé par copie publique des atomes, puis
le contrôle affecté et tous les contrôles restants ont passé avec warnings-as-errors.
Les composants précédents, inchangés, avaient passé ; les deux logs sont conservés.
La préparation compte63cas, dont les deux blocs complets de la capture.
Les PNG ont été inspectés visuellement. [Détails et SHA](latex-verification.json),
[archive brute](latex-evidence.tar.gz). Aucun changement de poids
ni préférence. Pas de vérificateur formel Swift/AppKit compatible dans l’outillage
du dépôt ; les contrôles de types et bitmaps sont des tests, pas une preuve.
