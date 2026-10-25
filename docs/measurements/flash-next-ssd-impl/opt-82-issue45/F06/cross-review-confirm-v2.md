# 82F06 — confirmation C-H v2 et clôture

Les huit réponses C-H v2 ont texte et hash gold60 exacts, 128/127 tokens, modes et compteurs conformes. Avec H-C, seize réponses complètes sont vérifiées. Aucun résultat de la confirmation v1 interrompue n’est réutilisé.

| Prompt | C-H H tok/s | C-H C tok/s | Écart brut C/H | Géomoyenne brute HCCH |
|---|---:|---:|---:|---:|
| MAIN | 11.486666 | 11.441572 | -0.3926% | -0.3556% |
| HOLD | 10.762146 | 10.223068 | -5.0090% | -1.9597% |

**Aucun gain qualifié.** MAIN est légèrement négatif dans les deux ordres (géomoyenne −0,3556 %), sans preuve de régression causale universelle. HOLD-C présente **27 197 440 octets de swap-out système**, soit 25,9375 Mio ; son temps C-H n’est pas qualifié au critère zéro swap-out global. La garde stock 256 Mio est respectée ; l’origine des pages est inconnue. La moyenne brute HOLD et son écart −5,0090 % restent descriptifs et ne démontrent pas une régression du candidat. Le monitor en direct a vu zéro, mais le compteur de fin de phase a bien capturé ce delta : il doit être conservé.

Acceptations, histogrammes à huit cases, largeurs et octets SSD logiques identiques entre H et C dans chaque phase. Compteurs MTP inactifs JSON null ; index 1 des histogrammes = ancre + une proposition. Les durées de profil sont inclusives côté hôte et ne sont pas additionnées aux totaux block/decode.

Les six phases, enfants, lecteurs et sampler sont clos et joints, secteur AC ; 120 observations sans erreur. Aucun swap-in système sur C-H v2. Les autres phases C-H ont zéro swap-out.
Pic physique de durée de vie sur douze snapshots : **7.851 Go**, compression logique courante maximale **3.158 Go**, ledger swap-in propre zéro ; caps 77B inchangés et non qualifiés.

Les 85 sources du snapshot archivé correspondent à l’identité gelée ; le CLI cbc03630 et le contrôleur 8935 sont inchangés. Le worktree actif est ensuite retiré/repris pour F05 : ces mesures restent celles du prototype gelé, pas du nouveau code. Le retrait effectif est cohérent ; aucun contrôle OFF/MTP de débit n’a été exécuté. Aucun modèle, test, build, preuveur ou GPU lancé par cette revue. Détails dans `cross-review-confirm-v2.json`.
