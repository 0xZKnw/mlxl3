# Profil Flash après correction RAM et lecteurs SSD parallèles

Essai FLASHSSD-16, source native `2e83f157`, HEAD documentaire
`f4d97231617fc3eb9db6f84b59a944e556a6d84a`. Apple M5/24 Gio,
MLX 0.32.2, Qwen3.8 Flash Next EXL3 2,05 bpw, checkpoint `65c89531`.
Cache experts fixé à 5 Gio (89 slots/couche), cible allocator 512 Mio,
quatre lecteurs, overlap actif. Applications utilisateur conservées, secteur.

Le profil historique09 utilisait 8 Gio et des lectures sérielles. Ce nouveau
diagnostic sert à choisir les prochaines expériences, pas à comparer les
débits des deux versions.

| Diagnostic, par pas | Français MAIN | Code HOLD |
|---|---:|---:|
| Parcours natif instrumenté | 147,96 ms | 138,59 ms |
| Remplissage des experts | 28,09 ms | 27,68 ms |
| Frontière de récupération des routes | 96,76 ms | 90,01 ms |
| Données lues logiquement | 125,34 Mio | 123,15 Mio |
| Routes trouvées dans les slots RAM | 78,06 % | 78,44 % |
| Part de la vérification cible dans le temps MTP | 95,67 % | 95,58 % |

La frontière des routes attend aussi le calcul GPU amont : ce n'est pas le
coût du routeur seul. Les octets lus incluent le cache de fichiers macOS et ne
mesurent pas les lectures physiques du SSD.

Avec des barrières explicites après chaque composant, la préparation
HC/attention représente 98,60/92,13 ms par pas, le MoE 80,36/74,57 ms et
l'injection HC 11,45/10,86 ms. Cette instrumentation modifie l'ordonnancement :
ces coûts ne s'additionnent pas au parcours natif et ne prédisent pas un gain.
Ils motivent les essais17 (cohorte unique contre séparation hot/cold) et18
(injection HC fusionnée en conservant les arrondis).

Une exécution de 177,19 s, deux prompts ×32 pas avec chauffe T1/MTP. Les
64 logits complets de la passe intrusive et tous les états cible sont
identiques ; tokens et états cible MTP sont également identiques. La tête MTP
reste résidente dans toutes les phases. Swapouts système supplémentaires :
0 octet ; RAM processus non mesurée par ce harnais, sans garantie sur les
autres conversations. Le swap déjà présent avant le lancement n'est pas
attribué au modèle.

Commande exacte, environnement, empreintes et conditions :
[execution.json](measurements/flash-next-ssd-impl/opt-16-profile/execution.json).
Données intégrales : [summary.json](measurements/flash-next-ssd-impl/opt-16-profile/summary.json)
et [run.log](measurements/flash-next-ssd-impl/opt-16-profile/run.log).
Le binaire de test `b564c6dd` est inchangé avant/après. Aucun compilateur,
proveur ou autre modèle de cette campagne ne tournait pendant le diagnostic.
Ce contrôle n'est ni un benchmark du bridge ni une preuve formelle du GPU.

## Essai17 : conserver le recouvrement hot/cold

Hypothèse : les quatre lecteurs ont-ils rendu inutile la séparation du calcul
des experts déjà en RAM et de ceux lus sur SSD ? Le chemin existant avec
`MLXL3_SSD_OVERLAP=0` a été comparé au même moteur `bd89a7d6` avec overlap1.
Deux wrappers identifiés changent uniquement cette variable, après le pilote.
Un prompt/mode par processus, une chauffe puis64 tokens/63 décodés, contexte4096,
cache5 Gio/allocator512 Mio, greedy, même ordre H-C pour ce premier crible.

| Cas | Overlap1, tok/s | Overlap0, tok/s | Écart |
|---|---:|---:|---:|
| Français OFF | 7,267 | 7,248 | −0,26 % |
| Code OFF | 7,347 | 7,200 | −2,00 % |
| Français MTP | 7,328 | 7,148 | −2,46 % |
| Code MTP | 7,175 | 6,793 | −5,33 % |

**Rejeté.** Le protocole exigeait au moins1 % sur chaque paire du premier
tour pour lancer les deux tours de confirmation. Huit bras terminés,
seize non lancés ; aucune conclusion statistique au-delà de ce crible.
Les16 réponses terminées gardent leurs textes/hashes exacts, tous les
swapouts sont nuls. Pics maximaux sur les deux variantes :8,685 Gio OFF et
9,454 Gio MTP ; plus grand supplément du candidat :4,88 Mio.
Le réglage overlap1 est conservé et aucun gain17 n'est publié.

Avant vitesse :11 tests Qwen physiques,11 positions indépendantes tenues
à part,3 tests SSD (dont90 cohortes, frontières et récupération après
erreur) et bridge1/2/3/17 avec configuration, refus et annulation/reprise
ont passé. Sources natives inchangées, aucun nouveau code de production.

Preuves : [identités](measurements/flash-next-ssd-impl/opt-17-cohort/identity.json),
[qualité](measurements/flash-next-ssd-impl/opt-17-cohort/quality.json),
[matrice](measurements/flash-next-ssd-impl/opt-17-cohort/matrix.json),
[exécution](measurements/flash-next-ssd-impl/opt-17-cohort/execution.json),
[résultat](measurements/flash-next-ssd-impl/opt-17-cohort/result.json).
Les durées des validations fonctionnelles ne font pas partie du benchmark.
