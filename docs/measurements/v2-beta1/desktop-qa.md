# Qualification Desktop complète — V2 bêta1

**PASS**, sourcee04c6558959609443dda362f2e8509b998adaa6d. `scripts/check-desktop.sh` exécuté intégralement, code0 en366,27s, limite1200s non atteinte. Les87fichiers de source/fixtures sélectionnés restent identiques ; leur manifeste SHA détaillé figure dans result.json. Aucune source M n’a été modifiée par cette qualification.

Les17programmes de test ont chacun émis leur marqueur final attendu, puis le script a terminé par « Desktop E2E checks passed ». Aucun programme n’a été sauté par le lanceur ; aucune ligne skipped/ignored n’est rapportée. Les compteurs décrivent des domaines différents et ne sont pas additionnés.

- Contexte240oracles ; hardening complet avec8scénarios Flash DFlash local,49états composer, MTP/DFlash legacy, cancellation/erreurs/persistance ; idle114assertions et sampling965assertions. Le message sampling historique annonce180états mais son domaine réel est216.
- Mise à jour : installation de **la vraie archive moteur bêta** dans un store jetable, relocalisation, exécution et fallback ; préflight de **la vraie image DMG bêta** avec signature, versions internes2.0.0/build31, compatibilité et hashes runtime/list, puis démontage. Les tests signés de remplacement/backup utilisent une app factice jetable ; l’app installée de l’utilisateur n’a pas été remplacée.
- Appupdater15cas ; fichiers150frontières/256encodages ; workspace219contrôles ; timeline/MCP/bridge ; rendu1194fragments code/4845Markdown/786appends/63dessins natifs ;9formulesLaTeX×2styles ;63cas response parser/chunk ; CLI ;300cas progress ;64grilles tuning ; migration/configMCP.

Les deux variables des paquets réels étaient renseignées avec leurs versions attendues2.0.0. Leurs tailles/SHA ont été comparés avant départ au manifeste ROOT/build/v2-beta1/artifacts.json, sans changement de contenu :

- MLXL3-Engine-v2.0.0-beta.1-arm64.tar.gz :67229225B, SHAca2e650f8cb944486592b60fc02f773e28388dd46a4ab38707971c0a11057412.
- MLXL3-Desktop-v2.0.0-beta.1-b31-Apple-Silicon.dmg :73990428B, SHA2042d20a2927723f4d386c8edb6b70f4e1f90eb257c24991d0021065ae3567b3.

## Commande et environnement

```sh
MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
MLXL3_TEST_PYTHON=/Users/justin/Documents/mix-stq1_0/.venv/bin/python3.12 \
MLXL3_TEST_ENGINE_ARCHIVE=/Users/justin/Documents/mix-stq1_0/build/v2-beta1/release/MLXL3-Engine-v2.0.0-beta.1-arm64.tar.gz \
MLXL3_TEST_ENGINE_VERSION=2.0.0 \
MLXL3_TEST_APP_DMG=/Users/justin/Documents/mix-stq1_0/build/v2-beta1/release/MLXL3-Desktop-v2.0.0-beta.1-b31-Apple-Silicon.dmg \
MLXL3_TEST_APP_VERSION=2.0.0 \
/bin/zsh scripts/check-desktop.sh
```

Depuis /Users/justin/.codex/worktrees/v2-merge/mix-stq1_0 ; PATHpréfixé ROOT/.venv/bin, /Users/justin/.cargo/bin et /opt/homebrew/bin. Swift6.4, mode de langage6, SDK26.5. La commande exacte et l’environnement pertinent sont conservés dans result.json.

## Limites et nettoyage

Le groupe59193 est vide après nettoyage, parent joint ; le contrôle global à09:43:32UTC ne trouve aucun compilateur Swift actif. GO GPU transmis au parent après cette vérification. Le test mémoire MCP a exercé son buffer Metal64MiB prévu (+99041280B d’empreinte observée), aucun modèle d’inférence GPU n’a été chargé ici.

Deux avertissements de chemins de recherche CLT au link et un message CoreGraphicsPDF sont conservés dans le log ; tous les programmes associés passent. Tests impératifs et contrôles de paquet, pas une preuve formelle Swift ni une mesure nouvelle de débit/mémoire d’inférence. Les générations physiques Flash, leurs modes, annulation/reprise et contrôles d’exactitude du paquet restent à la campagne séparée du parent. Aucun commit, push ou publication par cet agent.
