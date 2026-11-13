# PR161 R4 — sources, vérification et deux pilotes courts

[Rapport complet](../../issues/reviews/pr161-upstream-r4-2026-11-08.md). Le prototype réduit les copies du GDN groupé, conserve la parité numérique et passe la validation API/GPU de Metal ; aucun gain de vitesse démontré. Il reste local, sans promotion. La référence R3 figée est conservée.

- [Patch incrémental R3→R4](r4-views-increment.patch), [patch complet natif+inventaire](r4-final-delivery.patch), [identités et contrôles](r4-final-verification.json).
- [Matrices physiques](r4-views-physical.json), [erreur tardive](r4-views-late-error.json), [mutant détecté](r4-views-mutant.log), [preuve CPU ciblée](r4-views-kani.json), [validation Metal](r4-metal-validation.json).
- [Premier pilote](r4-views-speed-summary.json) : R3 8,337 / R4 8,058 / R3 8,291 tok/s ; [audit de l'adaptation](r4-views-adaptation-audit.json).
- [Fermeture normale des applications](r4-app-closure.json), [plan après fermeture](r4-quiet-speed-plan.json), [mesures](r4-quiet-speed-summary.json), [analyse](r4-quiet-speed-analysis.json) : R3 8,234 / R4 8,116 / R3 8,133 tok/s, mêmes 64 tokens, Brave/Deezer absents avant et après chaque bras.
- [Sources TurboEXL3](turbo-upstream-r4/manifest.json), [audits Regolith](regolith-r4-sources.json) : révisions vérifiées, SHA Git et SHA-256 enregistrés. Les audits Regolith restent dans le clone de recherche épinglé, leurs liens figurent dans le rapport.

Les binaires, dont le mutant volontairement incorrect, restent sous `build/pr161-speed-20261108` dans le worktree identifié par les manifestes ; ils ne sont pas copiés ici. Leurs SHA identifient précisément les exécutables testés. Le rapport sépare les résultats, les hypothèses de coût et les futures adaptations non implémentées. Les mesures instrumentées ne servent pas de mesures de vitesse.
