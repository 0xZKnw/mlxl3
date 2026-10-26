# PR47 — premier run Native, diagnostic final

SHA `174f4935029e7bafaf66e7b5cdb8a2c8a81e00a2`, [run 40241842056](https://github.com/0xZKnw/mlxl3/actions/runs/40241842056), tentative 1 : **completed/failure**.

Linux, macOS et `kani-flash-contracts` sont réussis. Le job global `kani` 121315116334 a réussi API usage, puis son étape safety a été annulée ; le job est en échec.

Le journal réel est lisible avec `gh run view … --job 121315116334 --log` et conservé dans `kani-gh.log` (211350 octets). L’endpoint direct n’était pas404 : gh refusait des séquences d’échappement terminal. À 22:54:41.3887355Z, le runner écrit : « The runner has received a shutdown signal. This can happen when the runner service is stopped, or a manually started runner is canceled. » À 22:54:42.0838940Z suit « The operation was canceled. » L’annotation API est « The operation was canceled. »

Dans l’étape safety, 45 harnais ont commencé et 44 ont un résultat `SUCCESSFUL`. Le dernier succès est `dflash::dflash_layouts_have_bounded_published_geometry`. Le 45e, `flash_cache_profile::verification::cache_allocation_is_optimal_and_exact_on_two_layers_two_levels`, commence à 22:52:41.8453279Z et n’a pas de résultat avant le signal d’arrêt. Aucun contre-exemple ou propriété échouée n’est visible. Cela ne prouve ni OOM, ni dépassement d’une borne de preuve, ni cause utilisateur du signal. La suite globale n’est pas entièrement vérifiée et la CI ne doit pas être annoncée verte.

Les deux occurrences réussies d’API usage sont séparées de ces 44 succès safety. Les métadonnées et les noms exacts sont dans `summary.json`, `run.json`, `jobs.json`, `annotations.json`. Aucun build, GPU, relancement ou changement de workflow effectué pendant cette lecture.
