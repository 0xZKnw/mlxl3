# Tests natifs

`contracts.rs` exerce des contrats CPU ; `runtime_gpu.rs` contient des contrôles nécessitant le runtime et le matériel appropriés. Les fichiers C++ couvrent notamment la lecture disque, les files, les capacités Metal et les helpers de mémoire ; certains sont des harnais de preuve. `fixtures/` conserve une fixture Qwen de contre-exemple.

Les cibles Rust sont déclarées dans le [Cargo.toml racine](../../Cargo.toml). Les commandes et options réellement utilisées en CI sont dans [rust.yml](../../.github/workflows/rust.yml) ; les tests physiques ignorés par défaut doivent être sélectionnés explicitement et ne deviennent pas validés par la CI hébergée.
