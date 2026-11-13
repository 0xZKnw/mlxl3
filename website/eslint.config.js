export default [
  {
    files: ["*.js", "*.mjs"],
    languageOptions: {
      globals: {
        document: "readonly",
        window: "readonly",
        navigator: "readonly",
        HTMLButtonElement: "readonly",
        HTMLDetailsElement: "readonly",
        HTMLDialogElement: "readonly",
        HTMLAnchorElement: "readonly",
        process: "readonly",
        URL: "readonly",
        innerWidth: "readonly",
        getComputedStyle: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "no-undef": "error",
      "no-unreachable": "error",
      "no-constant-condition": "error",
      eqeqeq: "error",
    },
  },
];
