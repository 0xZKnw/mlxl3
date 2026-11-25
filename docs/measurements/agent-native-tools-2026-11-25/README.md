# AGENT-2026-11-25-NATIVE-TOOLS — preuves

- `mutants.py` : 22 mutants temporaires des gardes ajoutées (voir la liste dans le script).
- `mutants-run1-interrupted.log` : passage dans l'arbre de travail, M01–M08, 8 détectés sur 8,
  interrompu pour libérer l'arbre (sources comparées à leur copie après l'arrêt : identiques).
- `mutants-run2-interrupted.log` : passage sur une copie des sources finales, M09–M12 :
  M09, M11 et M12 détectés, M10 ne compile pas (ne prouve rien) ; M13–M22 **non lancés**
  (arrêt demandé par l'utilisateur, une autre session mesurait sur le GPU). Aucun rapport JSON
  n'a été écrit : le script ne l'écrit qu'à la fin.
- `definitions.json` : définitions d'outils rendues par `tools/list` (entrée `--agent-tools`,
  `long-calls`), pour les harnais `native` et `javascript`, avec et sans Internet, mises au
  format des fonctions que le moteur passe au gabarit. Tokens comptés sur ce JSON (pas sur le
  prompt rendu par le gabarit) : Qwen3.8-27B 1 165 / 975 (natif, avec / sans Internet) contre
  846 / 732 (JavaScript) ; Inkling 2 Mini 1 587 / 1 331 contre 945 / 824.
