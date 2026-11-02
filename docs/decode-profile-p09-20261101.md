# P09 — génération Flash après intégration

État : **six séries CLI et trois captures corrélées terminées** sous S32.
Contextes longs terminés ; C1 et TAIL2 terminés ; TAIL2 prometteur conservé local, aucune garantie de non-régression decode.
Référence restaurable `d5a776e53425bb4d0ba61c0774c5399ae7987c06`.
Branche locale `codex/decode-profile-p09-20261101`, code non poussé et application installée inchangée. Rapport publié dans [l’issue #65](https://github.com/0xZKnw/mlxl3/issues/65). M5 24 Go, Qwen3.8 Flash-Next EXL3 2.05 bpw
`65c89531`, experts 3 Gio, quatre lecteurs statiques, Cold U actif.

## Révision S32 : résultats actuels

Les tableaux ci-dessous remplacent les conclusions provisoires des quatre
premières séries, conservées plus bas et dans leurs JSON. Les écritures restent
comptées depuis l'ancre d'origine ; après les trois captures : 786 432 octets
écrits (0,75 Mio), 65 421 312 octets lus. Aucun processus de capture restant.

| Mode | Prompt | Decode médian, tok/s | Étendue | MAD | Prefill médian, s | Génération médiane, s |
|---|---|---:|---:|---:|---:|---:|
| AR | MAIN | 11,487 | 11,471–11,544 | 0,012 | 7,824 | 18,885 |
| AR | HOLD | 11,423 | 11,383–11,608 | 0,034 | 7,289 | 18,413 |
| MTP | MAIN | 12,137 | 11,911–12,233 | 0,087 | 8,159 | 18,624 |
| MTP | HOLD | 11,172 | 11,066–11,348 | 0,082 | 7,641 | 19,002 |
| DFlash B2 | MAIN | 12,463 | 12,318–12,518 | 0,030 | 8,033 | 18,219 |
| DFlash B2 | HOLD | 11,919 | 11,737–12,107 | 0,143 | 7,553 | 18,243 |

Quatre mesures par case, deux processus, une chauffe par prompt/processus.
Entre premiers et derniers processus, les médianes de decode varient de
0,03 à 1,99 % selon la case. La pause avant MTP5/AR6 reste documentée.
Sorties et compteurs vérifiés ; aucune comparaison candidat/référence ici.
[Statistiques complètes](measurements/decode-profile-p09-20261101/baseline-complete-summary.json).

### Chronologie CPU/GPU sur les mêmes tokens

La fenêtre commune est **87–113, soit 27 tokens**, intersection des fenêtres
couvertes des trois captures, choisie selon la couverture et non la vitesse.
Les horloges sont recalées à partir des lectures POSIX et du nombre/octets des
lots natifs, avec une incertitude de l'ordre de 1–2 µs dans les analyses initiales.
Les contrôles de chaque capture comparent exactement les logits et états
cible/head/draft sur six phases, ainsi que les tokens au golden externe.

| Temps instrumenté, ms/token émis | AR | MTP | DFlash B2 |
|---|---:|---:|---:|
| Fenêtre complète | 97,861 | 102,622 | 90,544 |
| GPU du modèle actif | 59,175 | 66,386 | 58,665 |
| CPU inférence actif pendant GPU inactif | 19,545 | 17,106 | 14,834 |
| CPU inférence bloqué pendant GPU inactif | 17,910 | 17,185 | 16,060 |
| Attente des routes, inclusive | 33,863 | 31,328 | 26,570 |
| Dont GPU modèle actif | 33,767 | 30,732 | 26,464 |
| Soumissions MLX, inclusives | 24,897 | 26,915 | 23,336 |
| Dont CPU actif pendant GPU inactif | 12,433 | 9,323 | 7,940 |
| Lectures POSIX ∩ attente hôte ∩ CPU bloqué ∩ GPU inactif | 16,870 | 15,312 | 13,427 |
| Lectures POSIX ∩ GPU modèle actif | 5,844 | 9,644 | 8,257 |

Ces lignes se chevauchent : **ne pas les additionner**. « GPU inactif » signifie
aucune activité GPU observée, y compris celle d'autres processus. Les états
CPU, les activités GPU, les scopes inclusifs/exclusifs et leurs intersections
sont tous conservés dans les JSON ; le résidu de partition CPU est nul.
GPU actif ne mesure pas l'utilisation des unités arithmétiques. Les appels
POSIX peuvent être servis depuis le cache fichier ; une attente exposée ne
prouve ni une lecture NAND ni un gain récupérable équivalent.

Preuves : [AR](measurements/decode-profile-p09-20261101/AR-common/attribution.json),
[MTP](measurements/decode-profile-p09-20261101/MTP-common/attribution.json),
[DFlash](measurements/decode-profile-p09-20261101/DF-common/attribution.json).
Les premières fenêtres de 40 tokens sont conservées séparément. Leurs valeurs
diffèrent : notamment les 20,35 ms POSIX exposées AR annoncées initialement
portaient sur 78–117, pas sur cette fenêtre commune.

### Fonctions CPU et coût des observateurs

Sur AR commun : 977 échantillons du thread d'inférence, 527 rattachés à du CPU
actif pendant GPU inactif. Le contexte est celui juste avant l'interruption
d'échantillonnage, explicitement distingué d'un point directement Running.
Symboles MLX les plus proches : `CommandEncoder::get_command_encoder` (45),
`gpu::eval` (43), `dispatch_threads` (42), `maybeInsertBarrier` (40), `eval_impl`
(25), `commit` (21). 111 points n'ont pas de frame MLX. Les piles complètes,
allocations/libérations, frames inconnues et scopes sont conservés.
**Ces comptes ne sont pas des millisecondes par fonction.** Les appels de
traçage du driver sont aussi visibles : l'observateur coûte du temps.

| Decode de la requête complète, s | Contrôle avant | Capture Xcode | Trace 2 | Trace 3 | Contrôle après |
|---|---:|---:|---:|---:|---:|
| AR | 11,455 | 12,789 | 13,505 | 10,988 | 10,532 |
| MTP | 10,489 | 12,178 | 15,698 | 10,394 | 9,963 |
| DFlash | 10,054 | 11,625 | 12,124 | 10,089 | 9,735 |

La capture Xcode ajoute environ 12–22 % au temps selon le témoin retenu.
Les contrôles eux-mêmes dérivent ; aucun coefficient unique de correction
n'est justifié. Trace2 peut chevaucher la sauvegarde de Xcode. Trace3 est
environ 3,6–4,3 % plus lente que son contrôle suivant, mais cela reste une
comparaison séquentielle avec dérive. Les chronos CLI non instrumentés servent
aux décisions de performance. `pmset -g therm` n'a signalé aucun avertissement
au relevé AR ; température physique et fréquence GPU restent non mesurées.

### Ce qui mérite un essai, et les faux raccourcis

1. **Réduire le travail de graphe/encodage**, en préservant l'événement précoce
   des routes. Les 12,43 ms AR de soumission avec GPU idle forment un plafond
   observé dans cette trace, pas une économie réalisable garantie. L'encodage
   des calculs utiles reste nécessaire, et une partie du coût vient du traçage.
   Supprimer simplement le premier `async_eval` a déjà échoué dans POSTPROFILE03 :
   les routes attendaient alors la fin du calcul Hot. Aucun rerun de cette idée.
2. **Mieux masquer l'I/O Cold**, ciblé sur les 16,87 ms AR exposées dans la
   fenêtre commune. Le plafond suppose un masquage parfait sans travail ni
   mémoire supplémentaires ; il n'est pas un objectif réaliste à lui seul.
   W1 dépend déjà d'une partie des lectures, donc l'ancien raisonnement
   « tout GPU est indépendant de l'I/O » serait faux. Les lectures directes
   Metal et le chemin mono-lecteur sur le thread appelant existent déjà.
3. **Éviter les logits intermédiaires du prefill MTP**, changement bien plus
   petit : trace3 contient 87 évaluations du head cible pour MTP, contre 11 en
   AR et une en DFlash. Leur durée inclusive MTP est de 452,87 ms pour un
   prefill tracé de 10,739 s ; 86 extensions du head MTP prennent séparément
   120,91 ms. Les logits intermédiaires sont jetés, les états cachés et KV sont
   nécessaires. Les 452,87 ms incluent du travail amont : plafond optimiste
   d’environ 4,4 % sur ce prefill instrumenté, gain réel encore à tester.
   Risques à couvrir : dernier token, annulation, erreurs, états du head,
   continuation et compteurs d'acceptation. Le decode doit rester inchangé.

Changer seulement l'attente des routes ne peut pas récupérer ses 33,86 ms :
le GPU travaille presque tout ce temps. Les plafonds ci-dessus ne s'additionnent
pas et ne sont pas transposés en gain release. Les essais historiques de cache,
fusions et soumission restent conservés ; aucune idée n'est rejetée à partir
d'un seul chrono sans audit de son adaptation.

## Différence avec P08

P08 ne qualifiait pas une campagne complète et précédait Cold U et la fusion
des PR61–63. Ses durées restent historiques. P09 réutilise les outils, leurs
gardes et leurs tests, mais reprend les mesures sur les sources courantes.

L'audit initial trouve un défaut d'observation : `SsdMoe::fill_ready` récupère
les timings des lecteurs dans la FFI à la jointure, sans les exporter dans
`cold_ready_batch`. L'analyseur ne comptait que les lectures du fallback
synchrone. OBS1 exporte le mapping et les timings existants sous `cfg(test)`
et valide les deux chemins avec les mêmes contrats. Le cas de régression
montre 0 lecture comptée avant correction, 10 après, pour le même lot.
Ce correctif est diagnostique, sans gain d'inférence revendiqué.

## Matrice préenregistrée

1. Qualité Cold (préparation, clôture, fences et erreurs) puis oracle indépendant
   du modèle : sorties/états exacts avant les mesures.
2. CLI réel : AR, MTP, DFlash2 puis ordre inverse ; MAIN/HOLD (87/82 tokens
   d'entrée attendus), 128 sorties, une chauffe par prompt puis deux répétitions
   par processus. Tous les échantillons conservés, statistiques par charge.
   Cache experts neuf au lancement, chaud après les requêtes précédentes ;
   cache fichier macOS non purgé, contribution NAND non directement mesurée.
3. Diagnostic naturel : six phases par mode, chauffe, contrôle avant, trois
   phases instrumentées, contrôle après. Une capture corrélée CPU/GPU sur une
   phase ; les autres phases mesurent le coût propre des observateurs. Contrôles
   byte-exacts complets hors chronométrage. Jamais comparer le débit tracé au
   CLI normal pour conclure à une optimisation.
4. Contextes supplémentaires et éventuelles corrections : protocole détaillé
   à figer après le premier diagnostic, avant tout essai correspondant.

Admission : trois relevés espacés d'au moins dix secondes, estimation libre+
file-backed ≥10,5 Go, secteur, aucun build/proveur/modèle concurrent. Nouvelle
campagne explicitement demandée, ancre unique après préparation, aucune nouvelle
écriture swap et au plus 256 Mio de lectures cumulées ; jamais réancrer un
incident. RSS ≤10 Gio et footprint ≤12 Go. Estimation récupérable et RAM
physique ne sont pas synonymes ; ne pas présenter ces marges comme garanties.

Les [plans, sources figées et résultats](measurements/decode-profile-p09-20261101/)
conservent commandes, hash des binaires, refus et limites.

## Mesures CLI inspectées

Valeurs après chauffe, 128 tokens de sortie dont 127 chronométrés en decode.
MAIN contient 87 tokens d'entrée, HOLD 82. Une mesure désigne une requête ;
AR/MTP ont deux mesures par prompt dans un seul processus, DFlash en a quatre
dans deux processus. Le témoin AR final et le second processus MTP manquent :
la dérive temporelle de la matrice complète n'est pas qualifiée.

| Mode | Prompt | n | Decode médian, tok/s | Étendue, tok/s | MAD, tok/s | Prefill médian, s | TTFT médian, s | Génération médiane, s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR | MAIN | 2 | 11,475 | 11,471–11,479 | 0,004 | 7,824 | 7,824 | 18,891 |
| AR | HOLD | 2 | 11,389 | 11,383–11,395 | 0,006 | 7,292 | 7,293 | 18,444 |
| MTP | MAIN | 2 | 11,985 | 11,911–12,059 | 0,074 | 8,373 | 8,373 | 18,970 |
| MTP | HOLD | 2 | 11,090 | 11,066–11,115 | 0,025 | 7,698 | 7,699 | 19,150 |
| DFlash2 B2 | MAIN | 4 | 12,463 | 12,318–12,518 | 0,030 | 8,033 | 8,034 | 18,219 |
| DFlash2 B2 | HOLD | 4 | 11,919 | 11,737–12,107 | 0,143 | 7,553 | 7,554 | 18,243 |

DFlash utilise une seule proposition, largeur fixe, adaptation désactivée ;
ces chiffres ne décrivent pas toutes les configurations DFlash. Les colonnes
de durée proviennent du moteur ; elles n'incluent pas toutes les attentes du
client avant traitement. Les médianes séparées ne s'additionnent pas.
Ces données sont des références de l'état intégré, **pas un gain avant/après**.
La totalité des requêtes complètes a les mêmes 128 tokens et le même texte
entre modes : MAIN `f6b5f974c6db2e7a`, HOLD `9d02c1a97e015196`.
L'égalité du texte n'est pas à elle seule une preuve des états internes.

| Processus | Chargement annoncé, s | Spawn → ready, s | Pic footprint annoncé, Go décimaux |
|---|---:|---:|---:|
| AR1 | 0,589 | 1,630 | 7,137 |
| MTP2 | 1,021 | 1,763 | 7,886 |
| DF3 | 1,004 | 1,825 | 7,822 |
| DF4 | 1,008 | 1,824 | 7,833 |

L'écart spawn/chargement inclut notamment démarrage, checkpoint et transport ;
sa ventilation n'a pas été mesurée. Le footprint n'est ni le RSS ni la mémoire
allouée MLX. Les contrôles RSS du superviseur ne constituent pas une série
continue. Aucune température physique ni fréquence GPU n'est quantifiée ici.
Secteur et conditions d'admission sont consignés dans les reçus.

## Budget déjà observable et frontières réelles

Les compteurs existants de `Session::advance` dans
[`flash_draft_runtime.rs`](../native/src/flash_draft_runtime.rs) encadrent
successivement proposition, vérification et commit/cache. Ce sont des durées
murales côté hôte, **pas du temps CPU actif ni du temps GPU mesuré**.
Normalisation : sommes des durées divisées par les tokens de decode émis.

| Phase DFlash, ms/token émis | MAIN | HOLD |
|---|---:|---:|
| Proposition du brouillon | 3,864 | 3,932 |
| Vérification du modèle cible | 74,927 | 77,671 |
| Commit et cache | 1,582 | 1,604 |
| Hors de ces trois compteurs | 0,012 | 0,695 |
| Decode complet moyen | 80,385 | 83,903 |

HOLD termine par un token AR dans la branche `count == 0`, hors des trois
compteurs ; son résidu n'est donc pas intégralement un overhead. MAIN n'a pas
cette queue. La vérification représente environ 93 % du temps mesuré, mais
sa part CPU, GPU, SSD critique et SSD masquée reste **non mesurée dans P09**.
La durée inclusive du head ou des routes ne donnera pas à elle seule le coût
supprimable de ces opérations.

MTP accepte 54 propositions sur 72 pour MAIN (75 %) et 48 sur 78 pour HOLD
(61,54 %), avec un token de queue AR dans les deux cas. DFlash accepte 58 sur
69 (84,06 %) et 56 sur 70 (80 %). Ses compteurs rapportent 138/140 lignes de
vérification et respectivement zéro/un token AR de queue. Les lectures
d'experts demandées en DFlash sont 23,889/25,809 Go par requête : **demande
logique, pas volume physique NAND**, le cache fichier macOS n'étant pas purgé.

Source des statistiques complètes et de leurs limites :
[`baseline-partial-summary.json`](measurements/decode-profile-p09-20261101/baseline-partial-summary.json).
La mesure isolée de MTP5 reste dans son rapport incomplet, hors de cet agrégat.

## Priorités issues de ce diagnostic partiel

1. **Décomposer la vérification cible**, notamment `FlashNext::verify_draft`,
   le routage et `SsdMoe::fill_ready`. C'est le principal budget observé
   (75–78 ms/token DFlash), pas un plafond récupérable : supprimer ce calcul
   supprimerait aussi la vérification requise. Corréler les lectures POSIX,
   les états des threads et les intervalles GPU avant de choisir une correction.
   Risques : déplacer une attente sans réduire le chemin critique, conserver
   des buffers trop longtemps, ajouter un fence ou dégrader les fallbacks.
2. **Vérifier les heads inutiles du prefill MTP**.
   `prefill_flash_mtp` dans [`main.rs`](../native/src/main.rs) appelle
   `target.forward` pour chaque token, alors que le prefill AR et DFlash peuvent
   éviter les logits intermédiaires. L'essai historique HEADONLY laissait MTP
   inchangé. Le mécanisme est identifié dans le code ; coût GPU propre et gain
   global restent non mesurés. Ne pas attribuer les 0,4–0,55 s d'écart observé
   entre modes à ce seul calcul. Vérifier last_hidden, cache du head MTP,
   annulation, reset, erreurs et continuation avant tout chrono candidat.
3. **Brouillon et commit/cache** : budgets nettement plus petits. Une suppression
   fictive et complète du brouillon, à acceptation inchangée, plafonnerait le
   gain de débit à environ 4,9–5,0 % ; celle du commit/cache à environ 2,0 %.
   Ce sont des bornes arithmétiques irréalistes, pas des gains proposés.
   Les accélérer ne doit pas réduire l'acceptation ni augmenter le travail cible.

Aucune correction de performance retenue ou rejetée à ce stade. Seul le défaut
d'export des lectures Cold a été corrigé localement ; il ne modifie pas le
chemin release. Le coût de cet observateur et sa corrélation physique restent
à mesurer. Aucun ancien budget P08 n'est présenté comme nouveau résultat.

## Vérification et interruption

- `cargo fmt --check`, build release du libtest `mlx,chat` et Clippy strict
  tous targets : passent. Pas de changement du binaire CLI release.
- Suite pertinente complète : `python -m pytest -q tests/test_flash_*.py
  tests/test_decode_capture.py tests/test_decode_profile_p09.py` : **1 286 passés**
  en 96,35 s. Cas d'export Cold, erreurs de mapping/compteurs, prompts invalides,
  appel réel du CLI du benchmark avec fichier de prompts et gardes d'ancre inclus.
- Qualité physique Cold : une sélection passée en 2,310 s ; oracle indépendant :
  une sélection passée en 4,802 s, cinq tokens × 48 couches, 248 320 logits et
  reset exacts, 177 vagues naturelles. Ces contrôles ne qualifient pas encore
  la corrélation temporelle des nouveaux champs Cold.
- Pas de nouvelle preuve source : export Rust uniquement sous `cfg(test)`
  après jointure, chemin Kani MLX déjà non qualifié ; CrossHair absent. Pas de
  nouvelle suite physique exhaustive, ni de run CI du diff P09 local.

Les admissions initiales étaient 13,856/13,861/13,868 Go estimés disponibles.
Avant MTP5, l'estimation atteignait 16,670 Go ; pourtant la garde a observé
**196 608 octets (192 Kio) de nouvelles écritures swap globales** et
1 949 696 octets de lectures cumulées. Ces compteurs ne permettent pas
d'identifier le processus à l'origine de l'écriture. La série 5 a été arrêtée
et jointe, la 6 n'a pas démarré ; aucune application utilisateur n'a été fermée.
Le [contrôle après arrêt](measurements/decode-profile-p09-20261101/interruption-inspection.json)
confirme l'absence de processus de campagne restant. Tous les échecs sont gardés.

**Blocage concret :** la garde zéro nouvelle écriture de l'ancre P09 est
dépassée. La demande de plafond de 32 Mio depuis cette même ancre, incident
compris, attendait une réponse ; aucune réinitialisation ni relance automatique.
Les anciens plafonds autorisés concernaient une autre ancre. Il reste à terminer
les témoins inversés, mesurer l'overhead, capturer CPU/GPU, qualifier les contextes
longs et tester les corrections qui seraient alors justifiées.

**Amendement S32 autorisé ensuite par l'utilisateur :** plafond de 32 Mio
cumulés d'écritures depuis la même ancre, incident de 192 Kio compris. Limite
de lectures inchangée à 256 Mio, autres gardes conservées. Le compteur actuel
avant reprise était 192 Kio écrits et 11 812 864 octets lus. Les sources du
contrôleur précédent, l'identité et les échecs restent conservés ; nouvelle
identité `identity-s32.json`. Le wrapper P09 applique ce plafond aux deux
contrôleurs, celui de capture gardant zéro par défaut pour les autres campagnes.

## Référence longue complète

512 tokens réels, 64 sorties, une chauffe puis deux mesures par mode.

| Mode | Prefill médian, s | Decode médian, tok/s | Temps complet médian, s |
|---|---:|---:|---:|
| AR | 41.463 | 10.935 | 47.225 |
| MTP | 44.714 | 11.678 | 50.110 |
| DF | 44.058 | 12.193 | 49.226 |

Sorties identiques entre modes ; le contrôle long porte sur texte/tokens, pas sur tous les états numériques. Corpus différent des prompts courts, aucune causalité de longueur isolée. Séries séquentielles AR/MTP/DF, pas un A/B candidat. Cumul final : 786432B écrits et68108288B lus depuis l’ancre P09.

## P09-MTPHEAD : prototype prêt, qualification GPU bloquée

`FlashNext::prefill_mtp` centralise la boucle déjà utilisée par le bridge et
le profileur. Sous `MLXL3_FLASH_PREFILL_HEADS=1`, seuls les logits finaux sont
évalués ; chaque état caché et extension KV MTP reste calculé dans le même ordre.
Le flag 0 conserve toutes les évaluations. Le decode est inchangé dans le diff.
Ce prototype local n’est ni publié ni installé, et aucun gain n’est annoncé.

Le nouveau test physique compare l’ancien parcours explicite pour 1/2/7/8/9/17
tokens et les deux réglages : 248320 logits finis, états cibles et KV head exacts,
trois pas MTP avec compteurs, nombre de heads, vide/token invalide/annulation/
contexte épuisé et récupération. Son nom exact et les témoins AR/DFlash/oracle
sont présents dans le binaire compilé. Ces tests restent **non exécutés**.

Contrôles réellement terminés :

- `cargo fmt --check` et `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings` : succès.
- `cargo build --locked --release --features mlx,chat` et libtest release `--no-run` : succès.
- `cargo test --locked --no-default-features --features chat -- --test-threads=1` : 190 +35 +16 +35 tests réussis selon la cible, 9 ignorés ; aucune validation MLX par cette suite CPU.
- `python -m pytest -q tests/test_flash_*.py tests/test_decode_capture.py tests/test_decode_profile_p09.py tests/test_decode_p09_attribution.py` : 1302 réussis, 95,77 s.
- Ruff, contrôle du diff, 3375 unions +3375 intersections finies et cas XML : succès.
- `cargo kani --lib --tests --features mlx,chat --harness actual_mtp_prefill_contract --jobs 1` : interrompu à la borne RSS groupe 512 Mio après 5,43 s (pic 541622272 octets). Token u32 et booléens symboliques, séquence de deux tokens, aucune abstraction de MLX ; aucune obligation ni couverture vérifiée. Ce n’est pas une preuve.
- CI PR/main inspectée et étendue pour l’analyseur/gardes ; aucun run GitHub du diff local non publié.

[Identité et empreintes](measurements/decode-profile-p09-20261101/mtp-head/identity.json),
[contrôles CPU](measurements/decode-profile-p09-20261101/mtp-head/cpu-inspection.json),
[tentative Kani](measurements/decode-profile-p09-20261101/mtp-head/kani-result.json).
Les sources antérieures sont conservées dans `mtp-head/before/` et les binaires
P09 originaux restent disponibles.

Le cumul global atteignait déjà **348,109375 Mio de Swapouts et607,453125 Mio
de Swapins avant Kani**, contre les plafonds P09 32/256 Mio. La tentative n’a
modifié aucun de ces compteurs. La hausse entre la fin de LONG DF et la reprise
n’est pas attribuée à un processus précis. Aucun réancrage automatique ni
lancement GPU du candidat ; autorisation nouvelle nécessaire. Le dernier
contrôle disponible était de9,48Go estimés free+file-backed contre10,5Go requis.
Pas d’application utilisateur fermée ; YouTube reste ouvert. Les plans de qualité
puis A-C-C-A courts/longs sont prêts, mais aucune mesure candidate n’a commencé.

## Reprise MTPHEAD-S32 autorisée

L’utilisateur a fermé Brave et autorisé la nouvelle campagne32/256Mio avec
nouvelle ancre (`anchor-mtp-head.json`). L’ancienne ancre et son dépassement
restent conservés. Admission sur secteur,13,145Go estimés disponibles ; aucun
processus de compilation ou autre modèle concurrent. Contrôleur réutilisé par
une entrée dédiée,18tests de garde réussis ; CrossHair absent. Une erreur de
nom du parseur dans la préparation a interrompu la création avant toute écriture
d’ancre ; un lancement suivant sans identité a échoué avant tout modèle. Ces
échecs de préparation sont conservés, corrigés avant la campagne.

Les **trois tests GPU physiques passent** :

- Nouveau parcours MTP :6longueurs×2réglages,logits/états exacts à l’ancien
  parcours,continuation et compteurs,5cas d’erreur et reprise,headN→1 (19,26s).
- Oracle modèle indépendant :logits/états,177waves Cold (2,74s).
- Contrôle AR/DFlash :6longueurs×2modes,états/captures/KV/continuation/erreurs.

Le bridge CLI MTP passe aussi :2chauffes+2requêtes MAIN/HOLD de128sorties,
texte/hash/tokens et compteurs identiques à l’ancien binaire P09. Processus
et lecteurs joints ; zéro nouvelle écriture/lecture swap à cette étape.
Ces contrôles qualifient la correction observée, pas une preuve générale.

La comparaison de vitesse utilise les mêmes sources et le même binaire,
`MLXL3_FLASH_PREFILL_HEADS=0/1`,ordre A-C-C-A,une chauffe parprompt puis deux
répétitions ; corpus court MAIN/HOLD puis contexte512. Résultats à venir,
aucun boost livré/annoncé à ce stade. Pas de publication ni d’app remplacée.

## Résultats C1 complets et variante TAIL2

C1 signifie « seulement le dernier head MTP ». Même binaire, A désactivé/C1
activé, A-C-C-A,4mesures parvariante/prompt après chauffe. Tous les textes,
hashes,tokens et compteurs MTP restent identiques à la référence initiale.

| Charge | Prefill A→C1, s | Réduction prefill | Decode A→C1, tok/s | Variation decode | Réduction temps complet |
|---|---:|---:|---:|---:|---:|
| MAIN87,128sorties |8,168→7,897|3,32%|12,113→11,954|−1,31%|0,59%|
| HOLD82,128sorties |7,671→7,512|2,07%|11,149→11,136|−0,12%|0,74%|
| LONG512,64sorties |44,432→42,954|3,33%|11,906→11,718|−1,58%|2,93%|

Médianes, pas une addition de gains. Les valeurs brutes, médianes parprocessus,
conditions et contrôles sont conservés dans `mtp-head/short-inspection.json`
et `mtp-head/long-inspection.json`.36requêtes au total dont12chauffes.
L’absence de régression decode **n’est pas validée** ; C1 n’est pas retenu
comme réglage livré. Le gain prefill reste observé et l’idée n’est pas supprimée.

Audit : bon chemin CLI,formes/dtypes inchangés,aucun fallback inattendu,copies
ou coût mémoire important ajoutés ; les synchronisations/états et extensions
KV sont conservés. Un point à isoler : le loop initial garde le logits précédent
vivant pendant le dernier forward, contrairement à C1. TAIL2 calcule donc les
deux derniers heads pour rapprocher les allocations terminales de la référence.
Ce mécanisme est une hypothèse,aucune causalité allocator prouvée. Modification
limitée à la condition et au contrôle du nombre de heads ; anciens binaires et
preuves C1 conservés. Qualification numérique puis pilote C2-A préenregistrés
avant modification,confirmation A-C2 seulement si le pilote est favorable.

Le pic `RUSAGE_CHILDREN` est cumulatif au sein du superviseur multiprocessus ;
il ne permet pas d’attribuer un pic distinct à chaque variante. Les snapshots
`stats.memory` restent attribuables aux requêtes. Cumul à la fin de LONG C1 :
0,5Mio de Swapouts et0,75Mio de Swapins depuis l’ancre MTPHEAD-S32.

## Bilan final TAIL2/C2

Les tests de la variante à deux derniers heads sont terminés. Qualité physique
et bridge réel : logits/états/continuation et compteurs exacts dans les domaines
exercés. Même binaire pour A et C2 :
`0ebebb3fcd0315b336497305e73bec223b99352826d17ba5b347afc3f45527ba`.

| Charge | Prefill A→C2, s | Réduction prefill | Decode A→C2, tok/s | Variation decode | Réduction temps complet |
|---|---:|---:|---:|---:|---:|
| MAIN87/128sorties |8,258→7,853|4,91%|11,997→12,124|+1,05%|3,13%|
| HOLD82/128sorties |7,763→7,390|4,81%|11,090→11,135|+0,41%|1,78%|
| LONG512/64sorties |45,000→43,734|2,81%|11,748→11,661|−0,74%|2,43%|

Courts : C2-A-A-C2, quatre mesures par variante/prompt après chauffe. Long :
pilote C2-A, deux mesures par variante après chauffe. Toutes les requêtes
produisent le texte, hash et compteurs de la référence initiale. Le prefill
HOLD8,479s reste dans les données ; aucun échantillon supprimé. Les médianes
et mesures brutes sont dans `mtp-head-tail2/short-inspection.json` et
`mtp-head-tail2/long-inspection.json`. Ne pas comparer les pourcentages C1/C2
comme une causalité allocator démontrée : les campagnes sont distinctes.

**Décision : prometteur conservé dans le code local pour composition.** Le
prefill et le temps complet sont favorables dans ces mesures, avec coût mémoire
borné et mécanisme simple. **L’absence totale de régression decode n’est pas
validée**, notamment avec−0,74%sur le petit pilote long. Aucun gain decode
n’est attribué à la variante. Pas de push, de release ni d’app remplacée.
La commande supplémentaire `long-confirm-plan.json`, préparée pour une éventuelle
extension, n’a pas été lancée ; elle ne fait pas partie des résultats annoncés.

C2 : fmt/Clippy strict/builds passent ; test physique affecté et bridge passent.
Kani réel réessayé sur le domaine de deux tokens : arrêt limite mémoire en3,54s,
aucune obligation/couverture vérifiée. Les contrôles AR/DFlash et l’oracle cible
de C1 restent applicables à leurs chemins inchangés ; ils n’ont pas été rejoués
après la seule modification de condition MTP. Les hashes source/binaires finaux
correspondent à l’identité figée. Aucun processus de campagne restant.

Le rapport consolidé et son bilan sont publiés dans
[l’issue65](https://github.com/0xZKnw/mlxl3/issues/65). Les captures et prototypes
restent locaux ; leur publication n’est pas implicite dans celle de l’issue.
