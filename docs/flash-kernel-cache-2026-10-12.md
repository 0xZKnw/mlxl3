# Diagnostic du cache de kernels Flash — essai19

Le cache de factories Metal n'évince aucun kernel sur le parcours mesuré.
Cette piste n'explique donc pas la régression réelle du
[prototype HC18](flash-hc-injection-2026-10-12.md), malgré son microbenchmark
favorable. Aucune modification du cache n'est retenue.

L'ancien diagnostic HENK0O-QWEN27-10 avait relevé59 clés sans éviction sur
Qwen27B dense. La nouvelle raison de contrôle est le modèle Flash à512
experts, ses cohortes variables et sa vérification MTP, avec le kernel HC18
supplémentaire. Le comparateur de clés sans copies déjà intégré reste intact.

| Compteur du modèle complet | Résultat |
|---|---:|
| Appels de kernels personnalisés | 819 270 |
| Créations de factories | 58 |
| Cardinalité maximale | 58 / 128 |
| Évictions | 0 |
| Appels HC | 115 968 |
| Créations HC | 1 |
| Recherche des clés, cumulée | 715,40 ms |
| Construction des factories, cumulée | 4,23 ms |
| Construction de la factory HC | 4,04 µs |
| Durée du pilote instrumenté | 177,95 s |

La recherche et la construction représentent ensemble environ0,40 % de
cette durée totale. Cela n'inclut ni les CString côté Rust, ni la compilation
Metal différée, ni un profil GPU exclusif. Les débits contenus dans les
fichiers du pilote instrumenté ne sont utilisés pour annoncer aucun gain.

Même M5/24 Gio, MLX0.32.2, checkpoint65c89531/EXL3 2,05 bpw ; cache5 Gio,
allocator512 Mio, lecteurs4, overlap1, contexte4096, secteur et applications
utilisateur conservées. Un processus, MAIN/HOLD ×OFF/MTP, quatre chauffes
et quatre réponses de64 tokens. Les huit textes/hashes sont exacts. Aucun
compilateur/proveur ni autre modèle de la campagne pendant ce diagnostic.

Swapouts système supplémentaires :69 926 912 octets (66,69 Mio). Pic du
processus :10 697 253 000 octets (9,96 Gio). Ce protocole continu avec
changements de mode diffère des bras isolés17/18 : aucune attribution de ce
swap au seul modèle, aucune baisseRAM revendiquée. Les essais de vitesse
17/18 ont chacun relevé zéro swapout supplémentaire.

Avant le modèle, le contrôle existant de collisions/évictions passe :330
requêtes,327 créations,199 évictions,128 entrées maximales et1 320 valeurs
finies exactes. Les compteurs détectent donc effectivement une éviction.
Clippy strict, build livré et107 tests release/98 ignorés passent avec
l'instrumentation. CBMC6.11 échoue au parsing de libc++ Apple sur le vrai
bridge avec ses includes MLX ; aucune preuve C++ n'est revendiquée.

L'instrumentation temporaire et le prototype HC sont archivés et retirés.
Le moteur conservé est toujours le lecteur4 validé14B, source native2e83.
Aucun gain runtime nouveau, aucune installation ou release d'application.

Preuves : [commandes et conditions du modèle](measurements/flash-next-ssd-impl/opt-19-factories/model-execution.json),
[contrôle des compteurs](measurements/flash-next-ssd-impl/opt-19-factories/cache-quality.json),
[checks](measurements/flash-next-ssd-impl/opt-19-factories/checks.json),
[identités](measurements/flash-next-ssd-impl/opt-19-factories/identity.json),
[patch temporaire](measurements/flash-next-ssd-impl/opt-19-factories/diagnostic.patch),
[vérification du retrait](measurements/flash-next-ssd-impl/opt-19-factories/restore-identity.json).
