import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from test_smallm_benchmark import BENCH, ROOT, load_screen

SCREEN = load_screen("benchmark_smallm_reuse")


def test_m2_and_m4_matrix_contains_the_fixed_reference_and_only_preregistered_tiles():
    assert SCREEN.geometries(2) == [
        ("stock", 2, 1),
        ("stock", 2, 2),
        ("stock", 2, 4),
        ("shared", 2, 1),
        ("shared", 2, 2),
        ("shared", 2, 4),
    ]
    assert SCREEN.geometries(4) == [
        ("stock", 2, 2),
        ("stock", 4, 1),
        ("stock", 4, 2),
        ("shared", 4, 1),
        ("shared", 4, 2),
    ]


@pytest.mark.parametrize("rows", [True, 2.0, -1, 0, 1, 3, 5, 8, 23, 24, 2**31 - 1])
def test_unregistered_widths_are_rejected(rows):
    with pytest.raises(ValueError):
        SCREEN.geometries(rows)


def test_shared_decode_moves_each_conversion_outside_the_row_loop():
    source = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    candidate = SCREEN.shared_decode(source)
    marker = "for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {"
    body = candidate[candidate.index("float weight0") :]
    for i in range(4):
        declaration = f"float weight{i} = float(mlxl3_decode_codeword(cw{i}, CB));"
        assert candidate.count(declaration) == 1
        assert body.index(declaration) < body.index(marker)
        assert f"#else\n                                weight{i},\n#endif" in candidate
    assert candidate.count("fma(") == source.count("fma(")
    for broken in ["", source + source, candidate]:
        with pytest.raises(ValueError):
            SCREEN.shared_decode(broken)


@pytest.mark.parametrize("transform", [SCREEN.shared_decode, SCREEN.lookup_decode])
def test_decode_transform_preserves_grouped_k3_and_all_outside_loop_bytes(transform):
    source = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    grouped = source[
        source.index("#if MLXL3_GROUPED_ROUTES\n                        float decoded[4]") :
    ]
    grouped = grouped[: grouped.index("#endif") + len("#endif")]
    start = source.index("                    for (uint group = 0u; group < 2u; ++group)")
    prefix, suffix = (
        source[:start],
        source[source.index("\n            for (uint batch_row", start) :],
    )
    candidate = transform(source)
    assert grouped in candidate
    assert candidate.startswith(prefix) and candidate.endswith(suffix)
    for i in range(4):
        assert (
            candidate.count(f"float(mlxl3_decode_codeword(cw{i}, CB)),")
            == source.count(f"float(mlxl3_decode_codeword(cw{i}, CB)),") - 1
        )
        assert f"                                decoded[{i}],\n#else" in candidate
    assert candidate.count("fma(") == source.count("fma(")
    assert "#if !MLXL3_GROUPED_ROUTES\n                        float weight0" in candidate


@pytest.mark.parametrize("transform", [SCREEN.shared_decode, SCREEN.lookup_decode])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "closing", "nested", "extra_fma"])
def test_decode_transform_rejects_changed_loop_instead_of_matching_other_branches(
    transform, corruption
):
    source = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    old = "#else\n                                float(mlxl3_decode_codeword(cw0, CB)),\n#endif"
    assert source.count(old) == 1
    if corruption == "missing":
        source = source.replace(old, old.replace("cw0", "cw1"))
    elif corruption == "duplicate":
        source = source.replace(
            old, old.replace(",\n#endif", ", float(mlxl3_decode_codeword(cw0, CB)),\n#endif")
        )
    elif corruption == "closing":
        source = source.replace("                        }\n                    }\n#endif\n", "", 1)
    elif corruption == "nested":
        source = source.replace(old, "{" + old + "}")
    else:
        source = source.replace(old, old.replace("cw0, CB)),", "cw0, CB)), fma(0.0f, 0.0f, 0.0f),"))
    with pytest.raises(ValueError):
        transform(source)


def test_complete_mul1_table_matches_every_scalar_codeword_with_one_half_rounding():
    def half(bits):
        return struct.unpack("<e", struct.pack("<H", bits))[0]

    expected = b"".join(
        struct.pack(
            "<e",
            half(0x6400 + sum(((word * 0x83DCD12D) >> (8 * i)) & 255 for i in range(4)))
            * half(0x1EEE)
            + half(0xC931),
        )
        for word in range(65536)
    )
    table = SCREEN.mul1_table_cpu()
    assert table.shape == (65536,) and table.dtype == np.float16
    assert table.nbytes == 131072 and np.isfinite(table).all()
    assert table.tobytes() == expected
    assert not np.array_equal(table[:8], table[8:16])
    with pytest.raises(AssertionError):
        BENCH.assert_exact(table, np.roll(table, 1))


def test_lookup_source_preserves_all_fma_and_masks_full_codewords():
    stock = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    candidate = SCREEN.lookup_decode(stock)
    for i in range(4):
        assert candidate.count(f"float weight{i} = float(codebook_table[cw{i} & 0xffffu]);") == 1
        assert f"float(mlxl3_decode_codeword(cw{i}, CB));" not in candidate
    assert candidate.count("fma(") == stock.count("fma(")
    for broken in ("", stock + stock, candidate):
        with pytest.raises(ValueError):
            SCREEN.lookup_decode(broken)


def test_table_qmv_signature_and_actual_inputs_share_the_complete_buffer():
    specifications, invocations = [], []

    def kernel(**specification):
        specifications.append(specification)

        def call(**arguments):
            invocations.append(arguments)
            return ["result"]

        return call

    mx = SimpleNamespace(
        fast=SimpleNamespace(metal_kernel=kernel),
        float16=np.float16,
        float32=np.float32,
        uint32=np.uint32,
        array=lambda *args: np.array(*args),
    )
    shape = {"input": 5120, "widths": [17408, 17408], "k": 2, "cb": 2}
    table = SCREEN.mul1_table_cpu()
    BENCH.qmv(mx, shape, 2, source="body")
    call = BENCH.qmv(mx, shape, 2, source="body", codebook_table=table)
    assert specifications[0]["name"] != specifications[1]["name"]
    assert specifications[1]["input_names"] == [
        "xhat",
        "trellis",
        "tile_map",
        "tile_sub",
        "codebook_table",
    ]
    assert call("input", "weights", "sub") == "result"
    assert len(invocations[0]["inputs"]) == 5 and invocations[0]["inputs"][-1] is table
    for bad in (table[:8], table[:-1], table.reshape(256, 256), table.astype(np.float32)):
        with pytest.raises(ValueError, match="complete FP16"):
            BENCH.qmv(mx, shape, 2, codebook_table=bad)
    with pytest.raises(ValueError, match="Mul1"):
        BENCH.qmv(mx, {**shape, "cb": 1}, 2, codebook_table=table)


def test_qmv_kernel_identity_includes_actual_source_and_rejects_empty_body():
    calls = []
    mx = SimpleNamespace(fast=SimpleNamespace(metal_kernel=lambda **kw: calls.append(kw)))
    shape = {"input": 5120, "widths": [17408, 17408], "k": 2, "cb": 2}
    stock = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    BENCH.qmv(mx, shape, 2)
    BENCH.qmv(mx, shape, 2, source=stock)
    BENCH.qmv(mx, shape, 2, source=SCREEN.shared_decode(stock))
    assert calls[0]["name"] == calls[1]["name"] != calls[2]["name"]
    assert calls[0]["source"] == stock and calls[2]["source"] != stock
    with pytest.raises(ValueError, match="empty"):
        BENCH.qmv(mx, shape, 2, source=" \n")


def test_speed_gate_uses_paired_orders_wins_and_drift():
    result = SCREEN.speed_gate([[1.0] * 40, [0.8] * 40])
    assert result["passed"] and result["wins"] == 40
    assert result["gain_percent"] == pytest.approx(25)
    assert result["AB_BA_gain_percent"] == pytest.approx([25, 25])
    drift = [[1.0] * 20 + [1.04] * 20, [0.8] * 20 + [0.832] * 20]
    assert not SCREEN.speed_gate(drift)["passed"]
    bad_order = [[1.0] * 40, [1 / (1.02 if r % 2 == 0 else 1.08) for r in range(40)]]
    result = SCREEN.speed_gate(bad_order)
    assert result["wins"] == 40 and result["gain_percent"] > 3 and not result["passed"]
    few_wins = [[1.0] * 40, [0.5] * 20 + [1.01] * 20]
    assert not SCREEN.speed_gate(few_wins)["passed"]


@pytest.mark.parametrize(
    "bad", [[], [1.0] * 7, [1.0] * 9, [0.0] * 8, [True] * 8, [float("nan")] * 8]
)
def test_invalid_timing_pairs_are_rejected(bad):
    with pytest.raises(ValueError):
        SCREEN.speed_gate([bad, bad])


@pytest.mark.parametrize("iterations", ["0", "7", "9"])
def test_cli_rejects_invalid_protocol(iterations, tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as error:
        SCREEN.main(
            ["--checkpoint", str(tmp_path), "--output", str(output), "--iterations", iterations]
        )
    assert error.value.code == 2 and not output.exists()


def test_cli_preserves_previous_results(tmp_path):
    output = tmp_path / "result.json"
    output.write_text("previous failure")
    with pytest.raises(SystemExit):
        SCREEN.main(["--checkpoint", str(tmp_path), "--output", str(output)])
    assert output.read_text() == "previous failure"


def test_codebook_screen_rejects_unregistered_m4(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit) as error:
        SCREEN.main(
            [
                "--checkpoint",
                str(tmp_path),
                "--output",
                str(output),
                "--rows",
                "4",
                "--codebook-table",
            ]
        )
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("failure", [None, AssertionError("table divergence"), KeyboardInterrupt()])
def test_codebook_cli_keeps_failed_partial_evidence(tmp_path, monkeypatch, failure):
    def run(_, __, report, save):
        report["table"] = {"count": 65536, "bit_exact_cpu_gpu": False}
        save()
        if failure:
            raise failure

    monkeypatch.setattr(SCREEN, "run", run)
    output = tmp_path / "result.json"
    with pytest.raises(type(failure) if failure else AssertionError):
        SCREEN.main(["--checkpoint", str(tmp_path), "--output", str(output), "--codebook-table"])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["checks"] == [] and report["timings"] == []
    assert report["table"]["bit_exact_cpu_gpu"] is False


@pytest.mark.parametrize("bad", ["table_parity", "table_count", "mutation"])
def test_complete_counts_cannot_hide_bad_table_or_undetected_mutation(tmp_path, monkeypatch, bad):
    def run(_, __, report, save):
        report["checks"] = [{"bit_exact": True}] * 8
        report["timings"] = [{"status": "complete"}] * 4
        report["table"] = {
            "count": 65535 if bad == "table_count" else 65536,
            "bit_exact_cpu_gpu": bad != "table_parity",
        }
        report["mutation"] = {
            "rejected": True,
            "different_fp32_words": 0 if bad == "mutation" else 1,
        }
        save()

    monkeypatch.setattr(SCREEN, "run", run)
    output = tmp_path / "result.json"
    with pytest.raises(AssertionError):
        SCREEN.main(["--checkpoint", str(tmp_path), "--output", str(output), "--codebook-table"])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False


@pytest.mark.parametrize("failure", [None, AssertionError("divergence"), KeyboardInterrupt()])
@pytest.mark.parametrize("rows", [2, 4])
def test_cli_never_promotes_empty_or_interrupted_reports(tmp_path, monkeypatch, failure, rows):
    def run(_, __, report, save):
        report["checks"].append({"visited": True})
        save()
        if failure:
            raise failure

    monkeypatch.setattr(SCREEN, "run", run)
    output = tmp_path / "result.json"
    with pytest.raises(type(failure) if failure else AssertionError):
        SCREEN.main(["--checkpoint", str(tmp_path), "--output", str(output), "--rows", str(rows)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["checks"] == [{"visited": True}]
