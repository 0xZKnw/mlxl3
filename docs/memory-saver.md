# Économie de mémoire — validation du 8 octobre 2026

Le mode **Économie de mémoire** est implémenté dans les réglages Desktop, sur
`feat/mtp-toolbar`. Il est désactivé par défaut. Sur les deux passages du cas
Qwen3.8 réutilisant son contexte, il rend **475 à 779 Mo supplémentaires** après
la réponse. Pour Qwen3.6, la baisse de RAM du processus reste **non concluante**.
Les sorties greedy sont identiques sur les cas vérifiés.

## Comportement

La branche réunit le bouton Tune MTP de `88c2abd` et le correctif RAM déjà mesuré
sur `optimize/ram-idle-context`, commit `8f59264`. Ce correctif libère les états
finaux et les gros buffers inutilisés après une réponse. Ses anciens résultats
et diagnostics sont préservés dans [le rapport RAM](ram-idle-context.md).

Le nouveau mode renonce aussi au checkpoint de contexte et libère les petits
buffers inutilisés. Le modèle, la tête MTP, les poids, la quantification et les
conversations restent disponibles. **Le prochain message recalcule son contexte
et peut démarrer plus lentement.** La préférence est persistée et s'applique aux
prochaines requêtes ; elle ne libère pas rétroactivement un cache déjà au repos.
La modification du réglage est bloquée pendant génération ou Tune.

Le trajet de production est `AppSettingsView` → `StudioModel` → payload
`memory_saver` de génération/Tune → `BridgeRequest`. Le moteur supprime le
checkpoint précédent, désactive sa capture/réutilisation et appelle le cleanup
après le tour complet, y compris après erreur, annulation et Tune. Le cleanup
réinitialise l'état de travail, synchronise le GPU et libère les allocations
inutilisées : seuil historique >512 Mio en mode normal, >0 en mode économie.
Les poids et tableaux encore vivants conservent leur stockage.

Une capacité `memory_saver_supported` annoncée par le moteur évite une activation
silencieuse avec un ancien moteur. Dans ce cas, les réglages demandent sa mise
à jour ; la préférence ne modifie pas la génération ancienne.

## Comparaison de RAM

La référence contient **déjà le correctif RAM précédent**. La nouvelle matrice
n'est pas le scénario long de la capture utilisateur : contexte synthétique de
687 tokens pour Qwen3.8 et 645 pour Qwen3.6, limite 4096, 32 tokens de réponse,
greedy, tête MTP locale. Le tableau porte sur le cas `cache-warm`, MTP3 : la
référence réutilise512 tokens, le mode économie0, avec les mêmes tokens de sortie,
texte, contexte et compteurs MTP. Deux ordres indépendants A→B et B→A ; chaque
processus effectue aussi ordinary/MTP2, cache froid, erreur, annulation et reprise.

Footprint macOS du **moteur**, lu via `proc_pid_rusage` après un ping accusant la
fin du traitement. Go=10⁹octets, Mo=10⁶octets ; une valeur négative signifie une
hausse de footprint. L'interface n'est pas incluse dans ces mesures.

| Modèle / ordre | Normal (Go) | Économie (Go) | RAM rendue (Mo) |
|---|---:|---:|---:|
| Qwen3.8-27B / AB | 10.361 | 9.582 | 778.6 |
| Qwen3.8-27B / BA | 10.059 | 9.584 | 474.7 |
| Qwen3.6-35B-A3B / AB | 13.584 | 13.641 | -56.7 |
| Qwen3.6-35B-A3B / BA | 13.692 | 13.459 | 232.2 |

L'allocation **active MLX** diminue exactement de444596224 octets (424 Mio) sur
Qwen3.8 et203309056 octets (~194 Mio) sur Qwen3.6 dans chacun des deux ordres.
Ce compteur ne mesure pas la RAM du processus. Les snapshots peuvent retenir
un stockage partagé plus grand que leur taille logique ; leur borne cible de
256 Mio n'est pas un plafond d'empreinte physique.

Décision : économie de footprint **validée sur les deux cas denses mesurés** ;
footprint MoE **non concluant**, malgré la réduction d'allocations MLX vérifiée.
Aucun gain universel, baisse de pic ou accélération n'est revendiqué. Ne pas
additionner ces chiffres courts aux gains de 8,44 Go/5,5 Go du scénario long RAM-02.
Les gains sur les contextes longs déjà nettoyés par RAM-02 sont **non mesurés**.

## Sources et protocole

Mac Apple M5, Mac17,3, 24 Gio, macOS27.2/build26B5101f ; MLX 0.32.2,
Rust/Cargo 1.99.0, Swift 6.4 avec SDK 26.5, Kani 0.68.0/CBMC 6.11.0.
Qwen3.8 EXL3 config bits 2/head 3/MUL1 et Qwen3.6 EXL3 config bits 2,49/head 6/MCG ;
têtes managed locales 4 bit. Checkpoints identiques entre les bras et préservés.
Conditions alimentation/swap/diagnostics thermiques relevées par campagne,
température et fréquences non mesurées. Timings sur batterie diagnostiques.

Identités sources/binaires/modèles/dépendances et paramètres dans
[identity.json](measurements/memory-saver/identity.json). La référence est
HEAD 88c2abd plus les fichiers RAM-02 importés ; le candidat ajoute le mode.
Les hashes de binaire sont distincts et conservés. Le rebuild final du candidat
est byte exact au candidat initial, les dernières modifications Rust concernent
uniquement ses tests. Aucune compilation/proveur/autre modèle pendant les mesures.

Commandes exactes, échéances et conditions dans `*-command.json`. Échanges180 s,
campagne900 s, contrôleur930 s et nettoyage du groupe de processus sur erreur.
Les données synthétiques n'utilisent aucun PDF ni conversation personnelle.

## Vérification

- **338 tests Python passés / 4 sautés** : trois adaptateurs PonyExl3 absents et
  modèle Ling absent. Sous-ensemble driver/transport 69 tests, dont réponses
  invalides, grilles vides/dupliquées, divergence, silence, timeout et vrai enfant
  fermé/rejoint ; rapports d'échec `failed/parity:false` et handlers restaurés.
- **70 Rust passés / 60 ignorés** dans la variante livrée `mlx,chat`. Deux tests
  physiques ignorés sélectionnés séparément : snapshot et cache allocateur,
  **2/2 passés**. Snapshot : formes/nombres/finitude des 80 états et logits vérifiés,
  comparaison bit à bit après purge/restauration, préfixes 128/256 également testés.
  Variante CPU chat 57 passés/2 ignorés, recoupant la suite livrée.
- Buildrelease, format et Clippy strict `chat`/`mlx,chat` réussis. Desktop E2E
  complet passé, dont réglage/persistance/payloads génération/Tune et114 assertions
  idle/MTP ; recontrôle Swift strict passé après mutation.
- **41/41 harnais Kani CPU réussis**, 5470 obligations SUCCESS,106 couvertures SATISFIED,
  72 UNREACHABLE exclusivement stdlib/modèles Kani. Nouvelle règle : compteur u64
  et booléen entièrement symboliques, sans assume/stub/boucle ; 3 obligations et
  4 couvertures réussies. Vérification CPU bornée, **pas une preuve de MLX/Metal,
  allocateur, Swift, concurrence ou application complète**. CrossHair absent,
  tentatives conservées : Python non vérifié formellement ; aucun vérificateur
  source Swift applicable disponible.
- Huit moteurs de comparaison terminés/rejoints : **64 cas**,48 completions,
  8 annulations et8 erreurs attendues. Comparaison indépendante entre les deux
  ordres : tokens/texte/contexte/MTP exacts, caches économie nuls au repos.
- Un neuvième moteur vérifie OFF/OFF/ON/ON/OFF/OFF : réutilisation
  0/512/0/0/0/512 tokens, six completions exactes. Vrai Tune ON complet :4 profondeurs,
  190 tokens de decode chacune,2 hashes non vides identiques, cache nul après Tune.
- Mutations détectées sur copies isolées : seuil Rust erroné, mauvaise clé Swift,
  validation Python du cache supprimée. Production intacte ; contrôles concernés
  relancés. Le crash volontaire Swift a laissé un dossier/préférence jetable,
  identifiés précisément et nettoyés, audit conservé.

Les premiers problèmes de sandbox/SDK, erreurs des fixtures, timeout dense
historique et mutations rouges restent conservés. Pas de timeout Kani dans cette
campagne, ni contre-exemple de parité après correction des tests. Les logs sont
archivés sans perte `.log.gz`, chaque décompression comparée aux octets originaux
([index](measurements/memory-saver/log-archives.json)) ; originaux dans
`build/memory-saver/raw-logs/`, hors Git. Aucun log manquant.

La CI du **SHA déjà publié 88c2abd** est réussie,4/4 jobs inspectés ; le premier
lookup par SHA abrégé retournait une liste vide, le lookup complet a corrigé ce
constat. Cette CI **ne valide pas le nouveau diff local**. Les workflows PR/push
couvrent les nouvelles régressions et leur lint. Aucune nouvelle CI distante
exécutée, puisque le diff n'est ni commité ni poussé.

Parcours non qualifiés physiquement par cette campagne : autres architectures,
DFlash, MCP avec modèle réel, grandes conversations/imports et pic de préfill.
Le mode ne réduit pas les poids ; l'essentiel de la RAM restante leur correspond.
Il reste une option explicite, avec compromis de réemploi du contexte.

## État final

Code et preuves locaux, bouton Tune préservé. Tous les essais de cette campagne
sont terminés, résultats MoE qualifiés non concluants. **Aucune PR, aucun push,
aucune publication et aucune app installée remplacée.** Synthèse détaillée :
[verification-summary.json](measurements/memory-saver/verification-summary.json),
[mesures](measurements/memory-saver/memory-summary.json),
[transitions/Tune](measurements/memory-saver/transitions-physical.json).
