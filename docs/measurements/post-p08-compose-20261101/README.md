# GDN, F02 et combinaison : comparaison du vrai moteur

1 novembre 2026 — POSTP08-COMPOSE-01. **Arrêt demandé par l'utilisateur pour
passer à la proposition Cold prête : AR terminé, MTP interrompu, DFlash non
lancé.** Aucun défaut activé et aucune app remplacée.

## Protocole

Même CLI Rust release `mlx,chat`, SHA256
`2b0ab6951ef850baa2bc728bcd19060edfc8ac69bbf598cd230d5bb5452ff838`,
Apple M5 24 Go, MLX 0.32.2, Qwen3.8 Flash-Next EXL3 2,05 bpw checkpoint
65c89531. Cache experts 3 Gio / 53 slots × 48 couches, 4 lecteurs. Draft DFlash
Q4 a22a1744, largeur 1 ; MTP profondeur 1. [Manifeste](manifest.json),
[identité de 692 sources et 3 binaires](identity.json).

A = référence ; G = coefficients GDN préparés ; F = épilogue dense F02 ;
C = les deux. Ordre fixé **A G F C C F G A** dans chaque mode. Deux processus
par configuration ; chaque processus chauffe MAIN et HOLD, puis mesure MAIN
et HOLD une fois, 87/82 tokens d'entrée et 128 sorties / 127 transitions.
Débit agrégé = somme des tokens / somme des durées, pas moyenne des tok/s.
Les chauffes sont conservées et exclues du chrono comparatif. Pas de gain
attribué au chargement froid, ni de somme de gains entre modes/candidats.

Avant vitesse, neuf sélections physiques G/C ont passé : logits/états AR,
paires/commits, sessions MTP, DFlash et 36 préfixes 1..8, plus composants des
48 couches sous C. [Qualité](quality-summary.json), [preuves réutilisées](reused-evidence.md).
Les bridges17 contrôlent chaque configuration/mode avant son pilote128.
Les sorties des chauffes et mesures sont contrôlées par les goldens externes.

## Résultats AR complets

| Configuration | Décode tok/s | Écart | Préfill tok/s | Écart |
| --- | ---: | ---: | ---: | ---: |
| A | 10,48155 | référence | 10,27312 | référence |
| G | 10,41334 | −0,65 % | 10,43240 | +1,55 % |
| F | 10,45675 | −0,24 % | 10,21204 | −0,59 % |
| C | 10,57961 | +0,94 % | 10,46604 | +1,88 % |

[Résumé](AR-summary.json), [contre-analyse indépendante](AR-independent-analysis.md).
16 mesures, aucune exclusion : le second HOLD G à 9,77736 tok/s reste inclus.
Le témoin A agrégé presque stable masque MAIN −2,66 % et HOLD +2,56 % entre
passages ; le préfill MAIN varie davantage. C est positif en agrégé et sur les
deux prompts, avec un petit effet observé. Cela ne démontre ni interaction
causale entre G/F ni absence de régression sur tous les contextes.

## Adaptation, gardes et limites

[Audit du vrai chemin CLI](adaptation-audit.md) : aucun flag perdu ni fallback
global détecté ; F02 s'applique au domaine prévu, GDN conserve son arrondi F16.
Surcoût actif GDN observé cohérent avec 5,625 Mio de payload. Pas de nouvelle
copie/synchronisation dans le forward ; effets de donation/allocation/overlap
non attribués. Les lectures SSD réelles ne sont pas comptées par ces rapports.

Ancre F02 originale inchangée, plafond utilisateur total de 32 Mio de nouveaux
Swapout, Swapin ≤256 Mio, RSS ≤10 Gio, empreinte/pic natif <12 Go. Secteur,
processus GPU sérialisés, aucun build/proveur concurrent. Admission initiale
10,723/10,722/10,706 Go sur 20 s, puis contrôle avant chaque processus.
Les faibles événements swap autorisés sont conservés dans les reçus.

[Inspecteur](inspection-verification.json) : 72 cas synthétiques composition
et 89 historiques passent, mutant du contrôle GDN détecté ; format/lint/compile
et YAML passent. CI PR/main câblée, non exécutée à distance pour ce code local.
CrossHair absent ; Kani GDN bloqué par erreur interne, preuve F02 géométrique
historique limitée à son domaine. Aucun de ces contrôles CPU ne prouve Metal.

Les plans contiennent toutes les commandes ; runner S32 existant, nouveau CLI,
sorties et reçus séparés de F02 historique. Le CLI a été construit avec
`cargo build --locked --release --features mlx,chat --bin mlxl3-rs --message-format=json`.
Pas de modification native supplémentaire dans cette campagne. Les candidats
restent OFF ; aucune activation générale décidée sur cette campagne incomplète.

MTP : quatre bridges et six pilotes terminés/inspectés ; septième pilote G
interrompu à la demande utilisateur après 8,94 s, huitième témoin non lancé.
Contrôleur arrêté par SIGINT, groupe de test arrêté/joint ; résultat partiel
conservé et non validé. Les 12 processus AR et 10 processus MTP complets restent
archivés. Aucun gain MTP conclu avec la matrice tronquée. Les tests DF numériques
préparatoires ont passé, mais aucun bridge/pilote de vitesse DF n'a été lancé.
