# Portée des preuves reprises pour COMPOSE-01

Le code natif est celui de GDN R2, sans nouvelle édition pour cette campagne.
Le libtest SHA605f18cd est celui dont les oracles GDN et les 202 tests MLX
viennent de passer ; le CLI est reconstruit en release mlx,chat sur les mêmes
sources. Les anciens chronos ne sont pas repris dans la nouvelle matrice.

- GDN R2 : 36 coefficients transposés/castés face à CPU F16 indépendant,
  trajectoire externe 5 positions, cohortes 1..8 et états/préfixes/reset,
  mutant temporel détecté. La branche stock None construit toujours la même
  transpose/cast et l'ordre de calcul est inchangé. F seul utilise cette branche.
- F02 : module, shader, géométrie, appelants Linear et tests inchangés depuis
  les qualités 192 projections ON/OFF et les six oracles modèle historiques.
  Réutilisation pour F seul, complétée par son vrai bridge sur nouveau CLI.
  Ses dispatchs 8→5 restent une preuve du mécanisme, pas du gain actuel.
- Tête MTP externe : QSA, sans convolution GDN. Sa préparation et ses opérateurs
  ne sont pas changés par PREPARED_CONV ; fixture held-02 historique réutilisée.
  En revanche les sessions MTP sont retestées sous G/C car leur hidden vient
  réellement du target GDN.
- CPU sans MLX et parseur des flags : inchangés depuis les vérifications locales.
  Kani F02 géométrie : 201 obligations/9 covers, limites de modèle arithmétique
  conservées. Kani GDN : échec interne avant obligations ; aucune preuve GPU.

Nouvelles sélections sous G puis C : oracle indépendant logits/états AR,
paires/transactions multi-token, session MTP, DFlash et 36 préfixes1..8.
`flash_components_match_reference_on_identical_inputs` en plus sous C :
interaction du GDN préparé et de la projection F02 sur les 48 couches.

Les tests composant établissent les branches des loaders ; inspection des
sources strictes et empreintes garantit le code exécuté. Les oracles modèle
ne comptent pas eux-mêmes les dispatchs des flags : pas de nouvelle mesure
ni revendication de leur nombre. L'inspection des bridges vérifie les deux
flags de chaque processus, le mode réel, les propositions/acceptations,
les sorties golden, la complétion, les jointures et les budgets mémoire.
