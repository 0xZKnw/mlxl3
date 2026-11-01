# Contrôleur P08 : revue source avant modèle

31 octobre 2026. Lecture seule des exécutables ; aucun test, build, proveur,
GPU ou modèle lancé par cette revue. Préenregistrement :
`DECODE-PROFILE-08-CONTROLLER-01`, `opti.md`.

Sources relues (elles peuvent évoluer après cette revue) :

- `capture.py` : SHA-256
  `99b804cf8e935e8b740cee7682f46e9a3b54c77a1833aa05928b7ce6f767dab5`.
- `tests/test_decode_capture.py` : SHA-256
  `3c7e1b9e188dbf616a577bde80c6283e2a76b713ab3cb9fc9b70d7a9571d7e98`.

## P1 ouvert : SIGTERM peut abandonner les enfants

`capture.py:233–235` et `277–284` démarrent xctrace et le moteur dans des
sessions distinctes. Le contrôleur n'installe aucun handler SIGTERM. Le
SIGTERM par défaut termine Python sans passer dans le `finally:372–392` ;
un arrêt externe ou timeout qui termine le contrôleur peut donc laisser le
moteur et le traceur actifs. Le test d'interruption courant injecte
`KeyboardInterrupt` dans `validate_phase`, ce qui exerce bien le `finally`,
mais ne reproduit pas cette interruption de processus réelle.

Correction minimale proposée à l'orchestrateur : convertir SIGTERM en
exception contrôlée, nettoyer et joindre les deux groupes, puis restaurer
le handler. Ajouter un test jetable qui lance réellement le contrôleur,
attend deux enfants actifs, lui envoie SIGTERM et vérifie l'état d'échec
persisté et l'absence des deux groupes. SIGKILL reste hors des garanties
d'un nettoyage Python.

## Autres contrôles relus

- Les six phases correspondent au harnais natif actuel. L'attach suit le
  marqueur validé de `control_before`, l'arrêt suit `trace_1`, puis
  éventuellement une seconde capture couvre la fin de `trace_2`.
- Le sélecteur exact, le footer strict « 1 passed; 0 failed; 0 ignored »,
  les six résumés fermés, les témoins started/final/complete, les 128 tokens,
  les flags de baseline, les empreintes du binaire/sources/golden/options
  et la parité interne sont contrôlés. Les logs historiques natifs montrent
  les marqueurs au début de ligne ; le refus d'un préfixe n'introduit pas
  de défaut identifié pour ce harnais.
- Les arrêts silencieux, sorties prématurées et délais sont bornés dans la
  boucle. Les erreurs Python passent par le nettoyage des deux enfants.
- Une trace non vide n'est pas présentée comme couverture GPU validée : le
  statut reste `native_validated_captures_pending_inspection` et le champ
  de couverture exige l'inspection TOC/time-info externe. Le nombre de
  transitions réellement retenues par la fenêtre terminale de 8 s reste à
  mesurer ; ni 40 transitions, ni attribution GPU exclusive ne sont acquis.
- Les tests positifs utilisent des enfants CPU synthétiques et le schéma 1
  historique. Le binaire natif courant écrit le schéma 2, dont les exigences
  readbacks/signatures/requêtes sont appliquées par `profile_flash.analyze`.
  Ces tests ne qualifient donc pas une capture physique xctrace ni son arrêt
  SIGINT réel ; ils ne doivent pas être présentés ainsi.

Aucun autre défaut P1/P2 concret relevé dans la portée relue. La revue ne
remplace pas les tests affectés après le correctif du cycle de vie, ni
l'inspection des captures physiques avant toute attribution de temps.
