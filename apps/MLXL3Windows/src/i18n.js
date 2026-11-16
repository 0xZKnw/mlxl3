const STRINGS = {
  en: {
    engine: "Engine", detecting: "Detecting GPUs…", backend: "Backend", distro: "WSL2 distribution",
    install: "Install engine runtime", reinstall: "Reinstall engine runtime", models: "Models",
    addFolder: "Add folder…", unload: "Unload", download: "Download from Hugging Face", downloadBtn: "Download",
    context: "Context (tokens, 0 = auto)", settings: "Generation settings", thinking: "Thinking",
    newChat: "New conversation", placeholder: "Message", send: "Send", stop: "Stop", noModel: "No model loaded",
    runtimeReady: "Runtime ready", runtimeMissing: "Runtime not installed (downloads ~3–5 GB)",
    loading: "Loading", ready: "Ready", load: "Load", remove: "Remove", noGpu: "No supported GPU detected",
    engineStopped: "The engine stopped", reasoning: "Reasoning", noDistro: "No WSL2 distribution found",
    confirmRemove: "Remove this model from the library? Files stay on disk.", prefill: "Reading the prompt…",
  },
  fr: {
    engine: "Moteur", detecting: "Détection des GPU…", backend: "Backend", distro: "Distribution WSL2",
    install: "Installer le runtime", reinstall: "Réinstaller le runtime", models: "Modèles",
    addFolder: "Ajouter un dossier…", unload: "Décharger", download: "Télécharger depuis Hugging Face",
    downloadBtn: "Télécharger", context: "Contexte (tokens, 0 = auto)", settings: "Réglages de génération",
    thinking: "Réflexion", newChat: "Nouvelle conversation", placeholder: "Message", send: "Envoyer",
    stop: "Arrêter", noModel: "Aucun modèle chargé", runtimeReady: "Runtime prêt",
    runtimeMissing: "Runtime non installé (télécharge ~3–5 Go)", loading: "Chargement", ready: "Prêt",
    load: "Charger", remove: "Retirer", noGpu: "Aucun GPU compatible détecté", engineStopped: "Le moteur s'est arrêté",
    reasoning: "Réflexion", noDistro: "Aucune distribution WSL2 trouvée",
    confirmRemove: "Retirer ce modèle de la bibliothèque ? Les fichiers restent sur le disque.",
    prefill: "Lecture du prompt…",
  },
};

export function pickLanguage(languages) {
  for (const tag of languages || []) {
    const base = String(tag).toLowerCase().split("-")[0];
    if (STRINGS[base]) return base;
  }
  return "en";
}

export function translator(language) {
  const table = STRINGS[language] || STRINGS.en;
  return (key) => table[key] ?? STRINGS.en[key] ?? key;
}

export const LANGUAGES = Object.keys(STRINGS);
export const KEYS = Object.keys(STRINGS.en);
export const table = (language) => STRINGS[language];
