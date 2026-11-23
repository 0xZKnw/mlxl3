# Styles visuels de Desktop : Graphite par défaut, Aurora en option (2026-11-23)

## Objectif

L'interface 2.0 (aurora ASCII violette, titres sérif, pilules de verre) était jugée belle mais « cheap » et proche
des signatures d'interfaces générées. Plutôt que de la remplacer, la refonte ajoute des **styles** sélectionnables dans
Réglages › Apparence › Style. Le style est indépendant du choix clair/sombre/système.

| Style | Contenu | Accent apporté |
|---|---|---|
| **Graphite** (défaut) | neutres chauds, titres SF Pro, une seule surface surélevée, trame ASCII monochrome sur l'écran d'accueil | Neutre |
| **Épuré** | Graphite sans aucun fond | Neutre |
| **Aurora ASCII** | le design 2.0 à l'identique (lumière violette imprimée en caractères derrière chaque vue) | Iris |
| **Aurora** | la première aurora, sans caractères (dégradé maillé + cinq lueurs, repris de `7becaaba3`) | Iris |

Choisir un style applique son accent, sauf si une couleur personnalisée est active. Un accent choisi ensuite reste.
Un utilisateur qui n'a jamais choisi d'accent ouvre sur l'accent du style. Une valeur de style inconnue retombe sur Graphite.

## Sources consultées

- Skill `frontend-design` d'Anthropic (révision du 2026-09-03) : les signatures « fond quasi noir + un seul accent vif »
  et « titres sérif contrastés », les capitales espacées, le verre partout.
- Refonte de Linear (mars 2026) : gris plus chauds, sidebar « quelques crans » plus sombre, moins d'icônes, bordures adoucies.
- HIG Apple (mode sombre, matériaux, typographie) ; plafond APCA ≈ Lc 90 pour le texte clair sur fond sombre.
- Tonsky (2026) sur les icônes : une icône par item ne différencie plus rien.

Ces sources sont des avis de designers et des guides, pas des études ; les valeurs ont été réglées sur captures réelles.

## Graphite : jetons

| Jeton | Clair | Sombre |
|---|---|---|
| canvas | `#F9F8F6` | `#1B1A19` |
| sidebar (plus sombre que le canevas) | `#F1EFEC` | `#151413` |
| surface surélevée (composer) | `#FFFFFF` | `#252321` |
| contraste (base des voiles et filets) | `#1A1816` | `#F4EFE7` |
| encre | `#1C1B19` | `#EAE6DF` (pas de blanc pur) |
| secondaire / discret | contraste à 74 % / 62 % | 72 % / 56 % |
| signal « en cours » | `#B4661C` | `#E3A25F` |

Autres règles appliquées en Graphite seulement : titres SF Pro semibold (≈ 0,68 × la taille sérif), étiquettes en
casse normale, pensée en police proportionnelle, pas de logo ni de nom au-dessus de chaque réponse, pas de « Vous »,
anneau de focus neutre sur le composer, contrôles de barre d'outils posés sur le canevas (pas de verre sans contenu
dessous), sélection de conversation par luminance (sans barre d'accent), métriques sans icône par puce.

La trame d'accueil réutilise les cellules 5 × 8, l'alphabet `.:-=+xX` et la dérive du shader aurora, sans la lumière
colorée : `AuroraScene.monochromeShaderSource`, à 30 i/s (60 pour l'aurora), masquée dans la moitié haute et
seulement visible sur l'accueil vide. Le shader de l'aurora ASCII (`shaderSource`) n'est pas modifié.

## Vérifications

Voir le récapitulatif de la session : captures `--ui-preview` des quatre styles (clair et sombre), mesure des couleurs
rendues au pixel, contrôle WCAG ajouté à `tests/desktop-preferences-check.swift` (encre ≥ 12:1, secondaire ≥ 6:1,
discret ≥ 4,5:1 sur canevas et sidebar, pour chaque style et chaque mode), et suite `scripts/check-desktop.sh`.

Non vérifié : coût GPU de la trame pendant une inférence (non mesuré), rendu sous « Réduire la transparence » et
« Augmenter le contraste » hors des tests existants, app Windows (non portée).
