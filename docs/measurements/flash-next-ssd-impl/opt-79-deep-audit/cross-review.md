# Revue croisée des constats prioritaires — 79R

23 octobre 2026. Lecture seule par un second agent de chaque domaine ; aucune exécution GPU ni nouvelle preuve formelle.

## A01 / A02 : sélection DFlash2

La dépendance est confirmée : en mode fixe, les valeurs de confiance ne pilotent ni le choix de largeur ni l’observation adaptative. La longueur et la validité restent des contrats. Le `to_f32` peut être candidat à suppression si un indicateur GPU équivalent est renvoyé avec les IDs. Supprimer seulement softmax et sa validation accepterait une projection ou un codebook NaN auparavant rejeté, même si les logits top16 sont finis. Valider tous les scores finis serait aussi trop strict : des valeurs −Inf ou des prédécesseurs inaccessibles peuvent ne pas invalider la probabilité effectivement sélectionnée.

La première ligne de prédécesseurs contient seize fois l’ancre. En singleton, seul le rang zéro est accessible ; la transition Markov est cependant indispensable. Le passage d’une matrice16×256 à1×256 peut changer le kernel GEMM/GEMV et l’ordre des sommes. Une identité mathématique ne valide donc pas une transformation bit exacte. Un bloc adaptatif raccourci peut également avoir rows=1 : il doit continuer à retourner les confiances.

## B01 / B02 : entrée routée et restitution Hot/Cold

La chaîne réelle `verify_draft`→`forward_draft_lookahead`→`forward_rows_prepared` fournit les activations HC FP32[B,2560]. `routed_input` les duplique en[10B,2560] avant la conversion FP16. Déplacer cette conversion élémentaire avant la duplication est une piste exacte à tester ; la baisse d’intermédiaire calculée reste nominale.

La concaténation puis le gather Hot/Cold sont présents aussi sur B2 mixte. Un copieur direct doit choisir selon le snapshot Hot initial, jamais selon un hit du cache devenu vrai plus tard : une prélecture peut terminer entre ces deux instants. Il doit supporter Hot plein/masqué et Hot compact, conserver l’ordre Cold et l’ordre des dix contributions. Une somme masquée peut propager un NaN inutilisé ; elle n’est pas un choix équivalent.

Les garanties actuelles restent nécessaires : snapshot excluant pending, réservation protégeant les IDs routés, éviction seulement après complétion des consommateurs, durée de vie de `self.last` et fences. Aucune fuite ni course n’a été démontrée pendant cette revue.

## Portée

Les constats n’ont pas été invalidés par ces contre-exemples, mais les correctifs proposés ne sont pas implémentés ou validés. Les détails des domaines, sources et tests futurs sont dans verifier.md/json et ssd-memory.md/json.
