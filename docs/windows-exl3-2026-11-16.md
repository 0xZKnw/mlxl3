# MLXL3 Desktop pour Windows — moteur ExLlamaV3 (CUDA / ROCm)

Date : 2026-11-16. État : **code local, non publié, jamais exécuté sous Windows
ni sur GPU NVIDIA/AMD**. Ce document n'est pas un essai d'optimisation (rien
n'est ajouté à `opti.md`).

## Objectif

Version Windows de MLXL3 Desktop dont le moteur est **ExLlamaV3 amont**
(turboderp-org, MIT) intégré directement :

- **NVIDIA** : roue officielle `exllamav3 1.6.7+cu128.torch2.10.0` (win_amd64),
  native Windows, avec `torch 2.10.0+cu128` et `triton-windows 3.6.0.post26`.
- **AMD** : backend ROCm **d'ExLlamaV3 amont** (roue officielle
  `1.6.7+rocm10.0.torch2.13.0`, Linux), exécuté dans **WSL2**. Expérimental.

## Choix du moteur AMD

Recherche du 2026-11-16 (sources ci-dessous) :

| Candidat | Activité | Verdict |
| --- | --- | --- |
| ExLlamaV3 amont, backend ROCm (depuis v1.6.0, roues ROCm 10 publiées à chaque release, v1.6.6 le 2026-10-31, v1.6.7 le 2026-11-13) | Très active, même API que le chemin CUDA | **Retenu** : seul candidat avec des roues officielles, versionnées et maintenues par l'auteur d'EXL3 |
| rti/exllamav3-rocm | Active, RDNA3 (DFlash2/MTP, vision, KV 8 bits) | Plus rapide sur gfx1100 selon turboderp-org/exllamav3#543 (décodage amont 10-19 % plus lent), mais fork sans roues officielles. Candidat d'un backend « RDNA3 rapide » ultérieur |
| SelfRef/exllamav3-rdna | Active : amont + kernels RDNA3 de rti + réglages gfx1100 mesurés | Même remarque ; le plus proche de l'amont |
| phoenixhaxor/exllamav3-rocm | Port historique, inactif, fautes mémoire GPU intermittentes au chargement (issue #1) | Rejeté |
| CarouselAether/rocm_exl3, vcruz305/exllamav3-amd, loxdude/exllamav3-rocm-moe, autres | Niche / spécialisés | Rejetés |

Pourquoi WSL2 et pas HIP natif Windows : PyTorch publie des roues ROCm
Windows, mais ExLlamaV3 sous Windows lie cuBLAS et ne publie aucune roue HIP
Windows. La seule voie supportée amont pour AMD est Linux, donc WSL2 sous
Windows. Le support ROCm dans WSL2 dépend du pilote AMD et du GPU (RDNA3/RDNA4
principalement) : **non vérifié ici**.

Sources : https://github.com/turboderp-org/exllamav3 (releases v1.6.x),
https://github.com/turboderp-org/exllamav3/issues/543,
https://github.com/rti/exllamav3-rocm, https://github.com/SelfRef/exllamav3-rdna,
https://github.com/phoenixhaxor/exllamav3-rocm/issues/1,
https://github.com/woct0rdho/triton-windows, https://github.com/ROCm/ROCm.

## Architecture

```
Tauri 2 (apps/MLXL3Windows/src-tauri)  ──  UI HTML/JS sans bundler (src/)
        │ commandes / événements Tauri
mlxl3-windows-core (apps/MLXL3Windows/core) : framing, protocole, WSL, GPU,
        runtime (plans uv), bibliothèque de modèles, supervision de processus
        │ stdin/stdout JSON-lines, bridge protocol 1 (le même que mlxl3-rs bridge)
python -m mlxl3_exl3 bridge MODEL   (engines/exl3/mlxl3_exl3)
        │
ExLlamaV3 (Generator / Job / Cache paginé 256)  →  torch CUDA ou HIP
```

- **Runtimes privés** : `uv 0.13.8` (embarqué, sha256 vérifié en CI) crée un
  CPython 3.12 géré et un venv, puis installe les exigences épinglées
  (`engines/exl3/runtime/cuda-windows.txt` avec hashes, `rocm-wsl.txt`).
  L'identifiant de runtime est `cuda-`/`rocm-` + 12 hex du sha256 des
  exigences : changer une version crée un nouveau runtime au lieu d'en muter un.
  CUDA : `%LOCALAPPDATA%\MLXL3\runtimes`. ROCm : `$HOME/.local/share/mlxl3`
  dans la distribution WSL2 choisie, appelée par `wsl.exe -d <distro> --exec
  /usr/bin/env …` (argv seulement, pas de shell). Un marqueur JSON n'est écrit
  qu'après une sonde `probe` réussie (torch voit un GPU, ExLlamaV3 s'importe).
- **Worker** : tout le stdout natif est redirigé vers stderr (dup2) pour que
  seules les lignes du protocole sortent sur le descripteur dupliqué ;
  annulation via `cancel` sur stdin (pas de signal, compatible Windows) ;
  le gabarit de chat est rendu par Jinja2 sandboxé sans transformers.
- **Événements** : loading, ready (bridge_protocol 1), generation_status,
  delta, done (statistiques, tok/s), cancelled, error (fatal ou non). L'app
  marque chaque moteur par une époque et ignore les événements d'un moteur
  déchargé.
- **Téléchargements** : `huggingface_hub.snapshot_download`, dépôt et
  sous-dossier validés (aucune traversée de chemin) côté Rust et Python.

## Correctif AMD « torch sees no GPU » (2026-11-16)

Cause : dans WSL2, ROCm ne voit le GPU Radeon qu'à travers le pont DXG d'AMD,
**librocdxg** (paquet `rocdxg-roct`), qui n'est pas une roue pip. Le runtime
ne l'installait pas, donc `torch.cuda.device_count()` renvoyait 0 même avec le
backend ROCm choisi. Pour les versions de ROCm antérieures à 7.13, il faut aussi
`HSA_ENABLE_DXG_DETECTION=1`, ainsi qu'un pilote Adrenalin récent et un GPU pris
en charge (RX 9000/7000, PRO W7000/R9700, Ryzen AI Max ; pas de RX 6000).

Changements :

- La CI télécharge `rocdxg-roct_1.2.2_amd64.deb` (sha256
  `28ded125…702d541`, sans dépendance) dans `bin/` de l'installeur.
- Le plan ROCm commence par `wsl.exe -d <distro> -u root --exec /usr/bin/dpkg -i
  <deb>`. Cette commande ne demande pas de mot de passe et vise les
  distributions Debian/Ubuntu.
- Tous les processus ROCm (sonde, bridge, téléchargement) reçoivent
  `HSA_ENABLE_DXG_DETECTION=1` et `LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib`.
- La sonde vérifie `/dev/dxg`, `libdxcore.so`, `librocdxg` et la variable
  d'environnement. Elle renvoie des `hints` affichés dans l'erreur
  d'installation.

Pour l'utilisateur : mettre à jour l'app, lancer `wsl --update` puis
`wsl --shutdown`, installer le dernier pilote Adrenalin, puis cliquer
**Réinstaller le runtime**. **Non vérifié sur GPU AMD réel.**

Sources : https://github.com/ROCm/librocdxg,
https://github.com/ROCm/ROCm/issues/3470,
https://github.com/emilywilder/rocm-wsl-ai.

## CI : `.github/workflows/windows-desktop.yml`

- `release: published` sur un tag Desktop `vX.Y.Z[-pre]` (les `engine-v*`
  sont ignorés) : la porte attend jusqu'à 20 min l'asset
  `MLXL3-Desktop-v<ver>-b<N>-Apple-Silicon.dmg`, puis construit
  `MLXL3-Desktop-v<ver>-Windows-x64-setup.exe` (NSIS, installation par
  utilisateur) et son `.sha256`, et les attache à **la même release** avec
  `gh release upload --clobber`. Le statut « latest » n'est pas modifié, donc
  l'updater Mac (regex DMG exacte) n'est pas affecté.
- `workflow_dispatch` avec un tag : reconstruit et rattache pour une release
  existante.
- push sur `main` / PR touchant `apps/MLXL3Windows/**`, `engines/exl3/**` :
  contrôles + build, installeur en artefact de workflow seulement.
- Contrôles : tests Python du worker, `cargo fmt/clippy -D warnings/test` du
  core sous Ubuntu **et** Windows (supervision de vrais processus Python),
  tests Node de l'UI, puis `tauri build -- --locked`.

Limites CI : le workflow s'exécute depuis le commit du tag ; il ne
fonctionnera donc qu'à partir de la première release dont le tag contient ce
code (les tags existants jusqu'à v2.0.3 ne l'ont pas). Une release créée avec
le `GITHUB_TOKEN` d'un autre workflow ne déclenche pas d'événement ; une
release publiée avec `gh` par l'utilisateur, si. L'installeur n'est **pas
signé** (alerte SmartScreen attendue) et il n'y a **pas d'auto-updater
Windows**.

## Parité avec MLXL3 Desktop macOS (2026-11-16)

Référence : `origin/main` v2.0.4 (inventaire complet des fonctions Mac, fichier
par fichier). « Fait » = implémenté dans `apps/MLXL3Windows` et testé dans le
harnais stub (UI) ou par des tests unitaires ; rien n'a été exécuté sous Windows.

| Fonction Mac | Windows | Notes |
| --- | --- | --- |
| Interface 2.0.3 (barre latérale, conversations épinglées/récentes, recherche, aurora, accueil) | Fait | |
| Chat en streaming, réflexion repliable, Markdown, blocs de code, stats | Fait | |
| Copier / modifier / régénérer | Fait | Modifier et régénérer réécrivent la conversation courante |
| Créer une branche ici (et « modifier dans une branche » depuis un message utilisateur) | Fait | Titre « · branch », texte remis dans le brouillon |
| Lire à voix haute | Fait | `speechSynthesis` de WebView2 (voix Windows) |
| Export Markdown de la conversation (menu +) | Fait | Même format que `MarkdownExport` ; enregistrement par boîte de dialogue native |
| Contrôle d'affichage (largeur, taille du texte 12–24) | Fait | Popover de l'en-tête et Réglages → Apparence |
| Apparence : thème, accents Iris/Rose/Ambre/Neutre + personnalisé, espacement, police, bulles, largeur de barre latérale, réinitialisation | Fait | Valeurs reprises des enums Swift ; valeurs héritées sage/mint/coral converties |
| Notifications de fin de génération | Fait | Plugin Tauri ; seulement en arrière-plan, sans contenu de conversation |
| Modèle au démarrage (dernier sélectionné ou choisi) | Fait | Ignoré si le runtime n'est pas installé |
| Déchargement automatique après inactivité (jamais … 2 h, défaut 15 min) | Fait | Échéance non prolongée par les rendus ; jamais pendant une génération |
| Export / import de l'historique | Fait | JSON `mlxl3-windows-conversations` v1 ; l'import n'ajoute que les ids inconnus |
| Compte Hugging Face (token) | Fait | `%USERPROFILE%\.cache\huggingface\token`, validé par `whoami-v2` ; HF_TOKEN prioritaire |
| Page Confidentialité | Fait | Flux réseau propres à Windows (PyPI, download.pytorch.org, GitHub) |
| Mises à jour auto avec canaux Stable / Bêta | Fait | Tags `vX.Y.Z-beta.N` marqués prerelease ; installeur NSIS vérifié par SHA-256 |
| Bibliothèque, catalogue recommandé, recherche/téléchargement HF avec reprise | Fait | |
| Pièces jointes texte/PDF, images (modèles VL) | Fait (révision « tout ») | Bouton trombone, glisser-déposer Tauri, miniatures ; images envoyées seulement à un modèle VL, avec un budget limité. Les octets d'image ne sont pas conservés : une conversation rouverte garde les noms, et le modèle ne voit plus les pixels. Pas de brouillon par conversation |
| Outils MCP, mode Agent, Agent Studio | Fait, partiellement (révision « tout ») | Client MCP (stdio et HTTP), permissions par outil (refus par défaut dans un projet), lignes d'activité des outils. Agent : import d'un dossier, permissions lecture/écriture/commandes, revue des changements puis application. Les **commandes ne sont pas sandboxées** (désactivées par défaut ; pas d'AppContainer/Job Objects, alors que le Mac utilise Seatbelt). Pas de checkpoints git, d'export ZIP ni d'aperçu HTML depuis Agent Studio |
| Projets, profils, bibliothèque de documents et citations | Fait (révision « tout ») | Projets persistants avec instructions et profils ; documents indexés et recherchés, avec extraits cités et une fiche des sources. Les messages d'erreur d'extraction Rust sont en anglais seulement |
| Réponses JSON validées (`response_schema`) | Fait (révision « tout ») | Un schéma invalide affiche un toast et garde le prompt ; une réponse conforme affiche le badge « JSON validé » |
| Mode Serveur (API OpenAI), mode Lab | Fait (révision « tout ») | Serveur local ou réseau, avec une clé API régénérée à chaque démarrage et un journal. Il partage le modèle chargé : Chat et Agent sont bloqués tant qu'il tourne, et il s'arrête quand on quitte le mode. Lab : statistiques des tours (préremplissage, décodage, cache), graphique et matériel |
| MTP, DFlash2, auto-tuning | Sans objet | Spécifiques au moteur MLX/Metal |
| Quantization Studio | Sans objet | Le convertisseur Mac utilise MLX |
| Streaming SSD Flash Next, menu de barre de menus, dictée sur l'appareil | Sans objet | macOS uniquement (dictée : pas de reconnaissance locale fiable dans WebView2) |

## Vérification (2026-11-16, macOS arm64)

| Contrôle | Résultat |
| --- | --- |
| Worker Python (`python -m unittest discover -s tests -t .`, backend factice) | 30/30 ; mutations détectées : redirection dup2, remise à zéro de l'annulation, annulation ciblée |
| Core Rust `cargo test` | 27 tests unitaires + 6 tests d'intégration avec le vrai worker Python |
| Core `cargo clippy --all-targets -D warnings` | OK sur `aarch64-apple-darwin` et `x86_64-pc-windows-msvc` (check croisé, sans lien) |
| Shell Tauri `cargo clippy -D warnings` | OK sur macOS ; check croisé Windows impossible localement (`llvm-rc` absent pour tauri-winres) |
| UI `npm run check:ui` | 15/15 (Markdown sûr contre HTML hostile, machine d'état chat/moteur, i18n) |
| Révision parité Mac (même jour) : UI `npm run check:ui` | 47/47 (apparence, sauvegarde/import, branche, export Markdown, parité des clés en/fr, CSP, icônes) |
| Révision parité Mac : core `cargo test` / shell `cargo clippy --all-targets -D warnings` | 45 unitaires + 6 intégration ; clippy OK (macOS) |
| Révision parité Mac : worker Python | 33/33 |
| Révision parité Mac : harnais stub dans un navigateur | Popover d'affichage, Réglages (8 panneaux, en/fr, clair/sombre), chargement au démarrage, déchargement auto (minuterie raccourcie), notification seulement hors focus, export/import, branche, lecture à voix haute, connexion HF |
| Révision « tout » (même jour) : UI `npm run check:ui` | 64/64 |
| Révision « tout » : core `cargo test` / `cargo clippy --all-targets -D warnings` (core et shell Tauri) | 99 unitaires + 6 intégration ; clippy OK (macOS) |
| Révision « tout » : worker Python | 38/38 |
| Révision « tout » : harnais stub dans un navigateur | Création et persistance d'un projet ; ajout de documents (toast des fichiers ignorés) et recherche ; profils ; pièces jointes texte et miniature d'image ; envoi avec MCP, instructions du projet, extraits cités et images ; ligne d'outil et fiche des sources (Échap la ferme) ; mode Agent (import d'un dossier, popover des permissions avec l'avertissement sur les commandes, revue puis application) ; Serveur (démarrage réseau, journal, URL, arrêt quand on quitte le mode) ; Lab ; page MCP (transport local, permissions d'outil) ; schéma JSON invalide puis valide ; interface en français |
| Workflow | `actionlint` OK ; porte de release simulée sous bash (tags valides, prerelease, `engine-v*`, tags invalides, regex DMG) |
| uv 0.13.8 | sha256 Windows/Linux vérifiés contre les `.sha256` publiés ; arborescence des archives contrôlée |

**Non vérifié** : build NSIS réel, exécution sous Windows, génération sur GPU
NVIDIA (CUDA) ou AMD (ROCm/WSL2), compatibilité WSL2 ROCm du pilote/GPU,
installation effective des runtimes (plusieurs Go), parité numérique avec
ExLlamaV3 hors app, notifications Windows, boîtes de dialogue natives
d'export/import, connexion HF réelle, voix Windows, harnais Kani `buffer_and_lines_stay_bounded` (présent,
**non exécuté** à la demande de l'utilisateur), signature de code. Révision
« tout » : vrais serveurs MCP, exécution réelle des outils Agent, vrai serveur
OpenAI face à un client, extraction de vrais PDF, vision avec un vrai modèle VL
et glisser-déposer sous Windows ne sont pas vérifiés. Ces parcours n'ont été
exercés que dans le harnais stub et par les tests unitaires. Kani n'a pas été
exécuté.
