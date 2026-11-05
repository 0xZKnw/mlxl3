# Échec Kani GitHub — lecture du 25 octobre 2026

Le [job PR42c Kani121301090365](https://github.com/0xZKnw/mlxl3/actions/runs/40236783641/job/121301090365), SHA42c0958ea25db16d2dbba93f8bedad114ac94b18, est `completed/failure`. API usage Kani a réussi ; safety contracts a échoué. L’annotation GitHub dit exactement : « The hosted runner lost communication with the server. Anything in your workflow that terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error. »

Cette annotation établit une perte de communication du runner. Elle ne distingue pas OOM, CPU saturé, processus terminé ou réseau. Aucun contre-exemple, timeout de preuve ou nombre de harnais terminés ne peut être identifié : endpoint `/actions/jobs/121301090365/logs` renvoie404, et `gh run view … --log-failed` renvoie `log not found`.

Le [job main121301938901](https://github.com/0xZKnw/mlxl3/actions/runs/40237453435/job/121301938901), SHA9c8474ca1292e29bae3c7d4d6383bd1820549349, expose la même annotation. Au snapshot, le job est encore `in_progress` mais son étape safety est `cancelled` (fin21:58:45Z). API usage est réussi. Ce statut transitoire ne doit pas être présenté comme une CI verte.

Les métadonnées PR indiquent fin du job22:38:59Z et fin de l’étape safety21:54:57Z ; les deux horodatages sont conservés sans leur inventer une cause. Fichiers bruts : `job.json`, `annotations.json`, `main-job.json`, `main-annotations.json`, `job-log-error.txt`. `summary.json` conserve les commandes et limites. Aucun relancement, workflow, build ou test local n’a été effectué.
