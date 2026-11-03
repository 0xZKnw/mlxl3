//! Captured independent converter words also exercise the real CPU format reader/writer.
use mlxl3_native::codec::{self, Codebook};
use std::path::Path;

#[test]
fn captured_conversion_words_match_the_real_codec_without_mlx() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native-quantizer");
    let cases: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("reference.json"))?)?;
    assert_eq!(cases["status"], "complete");
    assert_eq!(cases["tile_shape"], serde_json::json!([4, 256]));
    let cases = cases["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 21);
    for case in cases {
        let k = case["k"].as_u64().unwrap() as usize;
        let cb = Codebook::try_from(case["codebook"].as_u64().unwrap() as u32)?;
        let read =
            |field: &str| std::fs::read(root.join(format!("oracle-{k}-{}-{field}", cb as u32)));
        let state_bytes = read("states.u16")?;
        let states: Vec<_> = state_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect();
        let packed_bytes = read("packed.u16")?;
        let packed: Vec<_> = packed_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect();
        let decoded_bytes = read("decoded.f32")?;
        let decoded: Vec<_> = decoded_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect();
        assert_eq!(
            (
                state_bytes.len(),
                states.len(),
                decoded.len(),
                decoded_bytes.len()
            ),
            (2048, 1024, 1024, 4096)
        );
        assert_eq!(packed_bytes.len(), 4 * 16 * k * 2);
        assert_eq!(codec::pack(&states, k)?, packed);
        assert_eq!(codec::unpack(&packed, k)?, states);
        for (&state, value) in states.iter().zip(decoded) {
            assert!(value.is_finite());
            assert_eq!(
                codec::decode_codeword(state, cb).to_f32().to_bits(),
                value.to_bits()
            );
        }
    }
    Ok(())
}
