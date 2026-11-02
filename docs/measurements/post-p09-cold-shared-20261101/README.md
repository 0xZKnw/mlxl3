# Cold shared — filtre court, 1 novembre 2026

**Clos, expérimental OFF ; signal de vitesse indécis.** À la demande de Justin,
pas de campagne longue : passage direct au prototype DEC-ONE. Dynamic+ColdU
reste ON. `MLXL3_SSD_COLD_SHARED=1` construit le shared après lancement SSD
et le soumet avec W1. OPT27/73 le préparaient avant les lectures.

Même binaire, M5/24 Go, Flash EXL3 2.05 bpw checkpoint65c89531, 3 Gio,
quatre lecteurs dynamiques, Cold U, prepared/F02/GDN OFF. AR MAIN/HOLD,
32 sorties/31 transitions, un warmup et une mesure par prompt, ordre A puis C.

| Prompt | A tok/s | C tok/s | Decode | Temps prefill |
|---|---:|---:|---:|---:|
| MAIN |11,253891|11,237026|−0,150%|−0,206%|
| HOLD |11,229366|11,266913|+0,334%|+0,312%|

Écran trop court et sans ordre inverse pour établir un petit gain ou une
régression. Huit réponses warm/mesure concordent avec le préfixe128 externe ;
texte et hashes A/C égaux. `screen-summary.json`, `screen-*/report.json`
et les plans conservent tous les résultats, options et commandes.

Trois tests physiques ciblés passent : 20 cas mixtes + 20 Hot et frontières/
doublons, erreur/reprise, puis oracle indépendant cible logits/états. Ils
assertent le chemin de soumission shared quand activé. Build mlx/chat, fmt
et Clippy passent. Kani sans `--tests` a d'abord sélectionné zéro harnais ;
la commande corrigée avec le vrai nom qualifié échoue dans intrinsics.rs:243
avant obligations. Aucune preuve formelle MLX/Metal. Rustfmt a ensuite réuni
une ligne d'assertion sans changer ses tokens ; l'identité mentionne cette
différence entre source de test et compilation.

Audit : shared calculé une seule fois, poids et mathématiques inchangés,
consommateurs retenus/joints sur erreur. MLX0.32.2 parcourt une bande BFS
à rebours ; placer W1 en premier dans la liste de sorties ne prouve pas son
exécution prioritaire. Pas de capture supplémentaire dans ce filtre rapide.
Il reste donc une incertitude de placement, pas un rejet général du shared.

Cinq processus GPU joints, zéro nouvelle écriture swap pendant ces processus,
même ancre MTPHEAD (32 Mio écritures/256 Mio lectures), secteur. Pas de MTP/DF
modèle, de suite exhaustive, d'activation, de push ni d'app installée pour
ce prototype. Les modifications locales préexistantes restent conservées.
