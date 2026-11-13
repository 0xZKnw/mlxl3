# Test fixtures

- `image-input/`: real images and error cases for Desktop import checks.
- `native-converter-host/`: reference data for host-side transformations.
- `native-feedback/`: product and search corpus with its manifest and hashes.
- `native-quantizer/`: tiles and independent quantization and packing oracles.
- Files at this level support cache contracts, Kani, and model-card display checks.

These fixtures are test references: do not change their contents or hashes without updating and requalifying the affected oracles. Related tests are in [tests/](../README.md).
