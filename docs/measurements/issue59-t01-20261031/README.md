# Issue 59 — T01 : coût hôte après hit de cache

31 octobre 2026. **Filtre terminé ; aucun boost modèle établi ni appliqué.**
Le portage T01A qui préparerait seulement le source est différé après audit du
faible budget observé. T01B, qui toucherait hash/pipeline/encodeur pendant
l'évaluation GPU, reste non mesuré et non réfuté.

La vraie entrée `linear::expert_slots` construit Hot GU K2/MUL1,20×2560→640,
53 slots résidents masqués. Fixture synthétique, graphes détruits sans
évaluation : **aucune valeur/logit/état calculé, aucune parité numérique**.
64 appels de chauffe, puis8 lots alternés×3 modes×256 appels :6144
constructions,24 échantillons. Les2048 appels instrumentés C++ sont des hits,
sans éviction ; sortie différée et zéro readback contrôlés à chaque appel.

| Poste | Temps moyen par appel | Limite |
| --- | ---: | --- |
| Construction complète, instrumentation désactivée |5,529µs| Inclut destruction et contrôles de métadonnées ; libtest, pas CLI modèle |
| Enveloppe Array Rust, instrumentation fine |2,854µs| Exclut collecte et destruction des CString/Vec locaux |
| CString Rust |0,692µs| Inclus dans Array |
| FFI Rust |2,028µs| Inclut les postes C++ suivants |
| Lookup C++ |0,650µs| Clé complète, cache à2 entrées dans cette fixture |
| Invocation de la closure MLX |0,698µs| Marshalling, signature/source et nouveau graphe ; source seule non isolée |

Les modes externes coarse/fine donnent5,370/5,883µs : l'effet observateur
est non monotone. Ces moyennes de lots ne sont ni des gains ni une distribution
sous charge modèle. Ne pas additionner les enveloppes inclusives ou corriger
les résultats par soustraction arbitraire.

Une [trace AR historique](historical-call-count.json) possède exactement la
même signature :48 appels par chacun de127 pas. Multiplier les coûts hôte du
filtre par ce compte donne **0,265ms pour la boucle entière**, **0,0335ms
pour la closure incluse**. Ce sont des projections conditionnelles, pas des
plafonds de gain modèle prouvés : autre charge/date, travail obligatoire dans
les enveloppes, fixture F16 non évaluée et concurrence CPU/GPU non reproduite.
Ces chiffres ne décomposent pas les12,142ms historiques de CPU pendant GPU idle.

Deux audits avant décision n'ont trouvé aucun défaut bloquant :
[adaptation et portée](audit-before-decision.md), [données](host-inspection.md).
Le profilage et ses tests sont archivés dans [instrumented-source/](instrumented-source/)
avec le binaire20a0006f… ; instrumentation retirée sélectivement,5 sources
restaurées et2 modules nouveaux retirés, sans toucher aux modifications
antérieures. [Reçu de retrait](retirement-result.json), [contrôles et limites
de preuve](verification.md). Aucun modèle benchmarké, app remplacée ou push.

À la demande de Justin, la procédure de vérification de [PR61](https://github.com/0xZKnw/mlxl3/pull/61)
est reprise dans [opti.md](../../../opti.md) : smoke ciblé, qualification des
chemins affectés avant vitesse, confirmation finale pertinente avant activation.
Les [tests repris et limites](pr61-staged-verification-audit.md) distinguent
T17 conservé OFF, T04 qualifié puis retiré et T18a encore à porter.
