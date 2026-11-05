# Moteur 1.4.8 : validation du défaut livré

Source propre du binaire signé : `26987a2c1aa9cc10a1908cdb6ffbec0eca55ccef`.

Le défaut sans variable active le chemin M5/EXL3 qualifié Q27-19. Les deux appelants et la clé Tune interprètent absence/1 comme ON et 0/vide/invalide/non Unicode comme OFF. Aucun shader, poids ou garde numérique ne change.

## Résultats inspectés

- `cargo fmt --all --check`, Clippy strict, build et `cargo test --locked --release --features mlx,chat` : 179 passés, 85 ignorés. Python : 442 tests passés, puis 29 tests du collecteur NDJSON.
- `cargo kani --lib --no-default-features --harness ...` : quatre harnais ciblés, 276 obligations satisfaites, 17 couvertures, sept branches internes std inatteignables. Parser symbolique présent/absent et 0..4 octets, unwind10 sans suppression des assertions ; garde et permutation ont leurs domaines complets déjà documentés. Les quatre passent aussi en CI Linux.
- Défaut/1/0 : six sorties FR130/code256 exactes aux références Q27-03 en IDs, texte, historique, nombres et compteurs MTP. Valeurs vides, malformées et non Unicode : clés stock0 distinctes. Six processus terminés/rejoints.
- Vrai bridge du paquet signé, sans variable MLXL3 : six budgets × quatre parcours, deux préfixes avec cache≥256, deux annulations/reprises, trois refus et fallback de pénalité. 17 records complets, enfant0/rejoint. Installateur Desktop1.4.6 : installation réelle dans un store jetable, sélection1.4.8, exécution relocalisée, rejet et fallback ; store utilisateur intact.
- Build arm64/macOS26.2/MLX0.32.2, trois signatures ad-hoc strictes, cinq membres TAR, tailles et hashes, manifeste et deux archives byte-identiques. Desktop1.4.6/build28 ne change pas.

## Vitesse et conditions

Paquet signé actuel, même binaire A=0 / B=variable absente, MTP1/tête d'origine/MLP automatique, allocateur par défaut, pipeline/adaptive/lookup0, greedy, contexte4096/cachepromptOFF, warm64/FR130/code256, ABBA. Quatre processus0/rejoints ; huit mesures et huit warmups non vides, exacts aux références historiques. Autre chat coordonné : aucune compilation/preuve/inférence concurrente.

Français A9,1697/B11,2051tok/s, dérives A8,818%/B0,668% ; code A8,6222/B11,2218, dérives4,772%/3,085%. **Non concluant en vitesse**, seuil3% dépassé ; BAAB conditionnelle non lancée. Batterie34→29%, swap global5,56–5,80Gio, pas d'alertepmset ; températures/fréquences et cause du débit absolu inférieur non établies. Les13,65/13,38 restent les chiffres qualifiés de Q27-19 ;15nonatteints, aucun gain de sessions différentes additionné.

## Diagnostics et limites

Régression du défaut reproduite sur l'ancienne fonction (test101 attendu), puis corrigée. `py_compile` a d'abord été refusé par le sandbox en écriture du cache, puis a passé ; collecte de conditions initiale sans chemin sysctl a échoué avant GPU, reprise avec chemins absolus. Le parent du checker attendait un objetJSON unique et a échoué sur le NDJSON ; ses étatsfailed sont conservés. Le nouveau validateur, testé sur29cas (vide,17retraits, divergence, nonfini, clés/cache/comptes/recovery invalides), inspecte les mêmes17records et le vrai exit0 sans répéter le GPU. Les sorties incomplètes ne deviennent pas des succès.

CI source : Linux/macOS natifs, Desktop/protocole et conversion/packing réussis. SuiteKani complète arrêtée143 pendant ancien `dflash_copy_len_17`, après65harnais réussis, sans échec d'assertion rapporté. Code/workflow de ce harnais inchangés depuis1.4.7 ;143 est compatible avecSIGTERM, cause non établie. Relance du seul job auSHAexact demandée, résultat global non vérifié. Aucun OOM ou timeout affirmé.

Limites : model checkingCPU borné ; pas de preuve MLX/Metal/allocateur/concurrence/génération complète. VraieAPIArray : ICEKani connu ; CrossHair retenté sur le vrai collecteur et absent. Longcontexte/autresMac/corpusqualité non requalifiés. Les contrôles physiques détaillésQ27-19 restent attachés à leurs sources inchangées ; ils n'ont pas été intégralement répétés pour un changement de défaut.

Les commandes complètes, options, identités, états et diagnostics sont dans l'archive de vérification jointe. La release publie séparément les noms, tailles etSHA-256 des quatre artefacts. Moteur uniquement, latest=false ; app installée et modèles non remplacés par cette publication.

## Publication vérifiée

[Release1.4.8](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.8), tag/source26987a2c. Quatre artefacts distants : tailles etSHA-256 identiques aux locaux ; Desktopv1.4.6 reste latest, aucun remplacement de l’app installée. Voirpublication.json pour les empreintes.

Le commit documentaire final ne modifie aucun exécutable et déclenche les workflows habituels sur main. La concurrence du workflow peut annuler la relance précédente ; ses états et l’interruption143 restent conservés, sans annonce de CI globale réussie. La proposition de sauter la CI a été refusée par la revue automatique et abandonnée avant toute exécution.
