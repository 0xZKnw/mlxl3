"""Exercise the real Kani orchestration without installing a verifier or using a GPU."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import kani_ci as ci

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("prlimit") is None, reason="Linux proof runner only")


def inventory(names):
    return {
        "kani-version": "0.68.0",
        "file-version": "0.1",
        "standard-harnesses": {"real.rs": names},
        "contract-harnesses": {},
        "contracts": [],
        "totals": {
            "standard-harnesses": len(names),
            "contract-harnesses": 0,
            "functions-under-contract": 0,
        },
    }


def success(name, count=1):
    return (
        f"Checking harness {name}...\n ** 1 of 1 cover properties satisfied\n"
        "VERIFICATION:- SUCCESSFUL\n"
    ) * count + f"Complete - {count} successfully verified harnesses, 0 failures, {count} total.\n"


def test_real_baseline_inventory_has_complete_disjoint_shards():
    payload = json.loads(
        (ROOT / "docs/measurements/kani-ci-repair/baseline-library-inventory.json").read_text()
    )
    names = ci.inventory_names(payload)
    assert len(names) == 140
    selected = [n for shard in ci.SHARDS for n in ci.select_names(names, shard)]
    assert len(selected) == len(set(selected)) == len(names)
    assert set(selected) == set(names)
    assert ci.select_names(names, "all") == names
    assert ci.shard_for("new_module::new_proof") == "core"
    assert ci.shard_for("flash_cache_profile::proof") == "allocator"
    assert ci.shard_for("flash_support::grouped_route_verification::future_proof") == "routes"
    with pytest.raises(ValueError, match="zero"):
        ci.select_names(["only::core"], "flash")


def test_required_obligations_preserve_baseline_and_complete_partitions():
    baseline = ci.inventory_names(
        json.loads(
            (ROOT / "docs/measurements/kani-ci-repair/baseline-library-inventory.json").read_text()
        )
    )
    replaced = {
        "flash_cache_profile::verification::cache_allocation_is_optimal_and_exact_on_two_layers_two_levels",
        "speculative::verification::lookup_proposes_only_latest_in_bounds_history",
        "flash_support::draft_contract_proofs::unique_expert_prefix_is_bounded_for_eighty_routes",
        "ssd_prefetch::verification::prediction_order_preserves_all_values_within_two_rows",
        *(f"speculative::verification::dflash_copy_len_{n}" for n in range(21)),
    }
    required = set(ci.required_names())
    assert set(baseline) - required == replaced
    frozen = set(
        json.loads((ROOT / "tests/fixtures/kani-obligations-d5a776e5.json").read_text())[
            "required_harnesses"
        ]
    )
    added = {
        "resident_scratch_policy::completed_slot_admission_and_reservation_are_bounded",
        "ar30_router_policy::paired_router_actual_geometry_preserves_every_reference_output_bound",
        "gpu_greedy_policy::device_greedy_admission_and_readback_preserve_actual_bounds",
        "gpu_wait_policy::gpu_spin_budget_never_exceeds_either_production_limit",
        "flash_support::ar20_resident_bundle_preserves_exact_formats_offsets_and_complete_output_grids",
        "flash_support::ar20_qsa_chain_selection_preserves_dense_private_and_both_pipeline_capabilities",
        "flash_support::ar20_head_gates_require_all_actual_runtime_and_capability_guards",
        "ar20_kernel_policy::native_kernel_mask_preserves_every_actual_compiler_switch",
        "expert_ar_policy::finish_mix_preserves_actual_capability_and_record_view_bounds",
        "flash_support::slru::slru_actual_promotions_preserve_every_slot_and_protected_capacity",
        "ssd_reads::dedicated_readahead_never_accepts_unqualified_source_or_replaced_inode",
        "ar10_defaults::qualified_defaults_preserve_strict_overrides_and_legacy_io",
        "flash_support::hc_write_norm_preserves_the_actual_dtype_and_workspace_guards",
        "flash_qmv::routed_nt2_requires_exact_records_and_covers_the_complete_output",
        "expert_ar_policy::expert_axis_admits_only_nonempty_bounded_ready_tables",
        "expert_ar_policy::expert_ar_flags_preserve_qualified_defaults_and_overrides",
        "expert_ar_policy::expert_ar_prefixes_are_bounded_disjoint_and_never_overflow",
        "expert_ar_policy::expert_ar_selection_requires_exact_actual_record_extents",
        "ssd_reads::split_record_tests::native_isolated_record_ranges_partition_exactly_one_valid_destination",
        "ssd_reads::split_record_tests::native_split_record_claims_never_overlap_or_repeat",
        "ssd_reads::record_reuse_tests::native_record_reuse_requires_completed_producers_and_private_foreground_slots",
        "flash_support::deferred_moe_cannot_escape_the_existing_cross_layer_guard",
        "flash_support::ar10_whole_gdn_requires_every_real_private_capability_guard",
        "flash_support::ar10_qsa_gate_requires_every_actual_short_dense_guard",
        "flash_support::gdn_private_epilogue_requires_every_actual_consumer_guard",
        "ssd_reads::common_queue_tests::native_common_queue_flag_is_strict_and_off_without_override",
        "flash_support::hc_kernel_flags_are_off_by_default_and_require_exact_m5_literals",
        "contracts::hc_fused_dispatches_require_exact_shapes_and_preserve_index_bounds",
        "fixed_residency_policy::fixed_residency_enrollment_requires_capability_and_admitted_nonempty_bytes",
        "hc_workspace_policy::tests::hc_workspace_exclusivity_and_error_clear_follow_actual_state_transitions",
        "hc_workspace_policy::tests::hc_workspace_reserve_is_exact_or_rejects_overflow",
        "hc_workspace_policy::tests::hc_workspace_flag_preserves_off_default_and_bounded_literals",
        "hc_workspace_policy::tests::hc_workspace_actual_admission_sum_is_checked_and_includes_every_buffer",
        "flash_support::cross_layer_selection_requires_all_transaction_guards",
        "ssd_reads::wave_tests::native_extra_wave_is_bounded_and_never_splits_two_or_fewer_remaining_reads",
        "ssd_reads::wave_tests::native_extra_wave_flag_is_strict_default_off_and_qualified_only",
        "ssd_reads::wave_tests::native_ready_minimum_rejects_impossible_thresholds_before_waiting",
        "hc_gdn_policy_tests::hc_gdn_joint_selection_requires_every_guard_for_symbolic_shapes",
        "hc_gdn_policy_tests::hc_gdn_flag_preserves_qualified_defaults_and_exact_literals",
        "ssd_reads::direct_tests::native_direct_destination_requires_exact_shared_extent",
        "ssd_reads::direct_tests::native_direct_reads_require_depth_zero_without_prefetch",
        "ssd_reads::direct_tests::native_direct_flag_preserves_qualified_default_and_exact_override",
        "flash_support::prefill_heads_never_skip_a_valid_final_row_or_accept_an_invalid_row",
        "flash_support::hc_chain_flag_uses_qualified_hardware_default_and_exact_literals",
        "flash_support::hc_gate_chain_selection_requires_every_production_guard",
        "tensor_layout::accepted_matrix_views_bound_every_sampled_coordinate",
        "tensor_layout::matrix_row_preserves_backing_and_address",
        "contracts::draft_blas_extents_reject_overflow_and_truncation",
        "contracts::flash_expert_submissions_share_intermediate_extents",
        "contracts::projection_offsets_fit_submitted_extents",
        "contracts::prepare128_validates_block_and_scale_indices",
        "flash_support::flash_record_projection_bindings_fit_the_pinned_read",
        "contracts::flash_gate_dispatch_bounds_output_and_weight_addresses",
        "contracts::flash_hc_groups_cover_exactly_the_checked_weight_matrix",
        "contracts::packed_flash_gdn_groups_partition_state_entries",
        "flash_support::ple_codewords_select_the_wrapped_nibbles",
        "flash_support::dense_flash_prompt_cursor_never_overflows_or_partially_accepts",
        "flash_support::qsa_append_preserves_all_published_rows",
        "flash_support::qsa_growth_fits_all_copied_and_appended_rows",
        "flash_support::qsa_stable_ranking_preserves_best_blocks_and_time_order",
        "tensor_layout::trace_extent_checks_bytes_and_slice_bound",
        "quantize::native_tile_search_launch_stays_inside_input_and_scratch",
        "convert_matmul::conversion_matmul_geometry_covers_all_shader_addresses",
        "runtime_stats::native_memory_stats_separate_metal_and_process_validity",
        "flash_support::native_demand_generation_rejects_stale_and_duplicate_publication",
        "flash_support::native_prefetch_room_bounds_owned_and_running_records",
        "contracts::native_qwen_gdn_geometry_bounds_every_addressed_row_and_state",
        "contracts::mapped_expert_dispatch_bounds_rows_splits_and_fixed_simd_width",
        "contracts::mapped_expert_row_offsets_are_aligned_and_fit_the_complete_bank",
        "contracts::ordinary_bundle_width_matches_wide_sum_for_up_to_three_projections",
        "contracts::ordinary_group_lane_layout_selects_the_original_scalar_operands",
        "contracts::native_projection_thread_count_matches_its_fixed_kernel",
        "contracts::native_flash_resident_nt2_selector_preserves_every_fallback",
        "contracts::native_flash_reference_grid_covers_only_complete_output_tiles",
        "contracts::native_flash_nt2_default_and_supported_overrides_are_exact",
        "flash_expert_scratch::native_routed_scratch_selection_preserves_all_fallbacks",
        "flash_expert_scratch::native_routed_scratch_prefix_capacities_use_wide_arithmetic",
        "flash_qmv::native_static_qmv_key_matches_the_actual_pso_table_and_every_fallback",
        "flash_qmv::native_fixed_qmv_key_checks_complete_shape_split_and_format",
        "flash_qmv::native_static_qmv_flag_preserves_default_and_bounded_utf8_literals",
        "contracts::native_flash_nt2_flag_default_and_bounded_unicode_literals",
        "contracts::native_projection_plan_has_bounded_power_of_two_splits_and_extents",
        "contracts::model_gqa_widths_reject_invalid_groups_and_overflow",
        "speculative::normalized_greedy_retains_first_finite_maximum",
        "contracts::router_dispatch_sizes_bound_static_memory_and_output",
        "dec_one::dec_one_storage_contract",
        "cold_ready::cold_row_limit_has_only_qualified_and_opt_in_values",
        "cold_ready::two_job_priority_keeps_records_and_stable_order",
        "contracts::flash_reference_mb34_preserves_coverage_and_fallbacks",
        "contracts::flash_reference_mb34_shape_scope_rejects_other_dimensions",
        "contracts::flash_reference_mb34_role_parser_preserves_default_and_named_choices",
        "contracts::flash_reference_mb34_role_shapes_and_selection_are_exact",
        "flash_memory::admission_and_shrink_respect_memory_bounds",
        "flash_memory::native_admission_covers_the_complete_geometric_slab",
        "flash_memory::native_request_budget_keeps_every_configured_limit_ar_only",
        "contracts::native_prompt_reuse_requires_the_whole_prefix_and_matching_owners",
        "flash_draft_budget::request_budget_reset_discards_all_observations",
        "flash_support::native_prepare_grid_contains_complete_hadamard_groups",
        "flash_support::stale_prefetch_completion_preserves_new_reservation",
    }
    assert len(frozen) == 322
    assert frozen <= required
    assert required - frozen == added
    assert {
        f"flash_support::verification::slot_capacity_{size}_matches_std"
        for size in ("empty", "one_byte", "two_bytes")
    } <= required
    grid = {
        f"speculative::verification::dflash_copy_len_{n}::accepted_{a}"
        for n in range(21)
        for a in range(8)
    }
    grid |= {f"speculative::verification::lookup_len_{n}" for n in range(21)}
    grid |= {
        f"flash_support::draft_contract_proofs::unique_expert_prefix_rows_{n}" for n in range(1, 9)
    }
    assert grid <= required
    ci.check_obligations(required | {"future_module::new_proof"})
    for missing in grid | added:
        with pytest.raises(ValueError, match="missing proof obligations"):
            ci.check_obligations(required - {missing})


@pytest.mark.parametrize("change", ["version", "duplicate", "empty", "total", "contract", "name"])
def test_inventory_rejects_missing_or_unsupported_obligations(change):
    payload = inventory(["core::proof"])
    if change == "version":
        payload["kani-version"] = "next"
    elif change == "duplicate":
        payload["standard-harnesses"]["real.rs"] *= 2
    elif change == "empty":
        payload["standard-harnesses"] = {}
    elif change == "total":
        payload["totals"]["standard-harnesses"] = 2
    elif change == "contract":
        payload["contract-harnesses"] = {"new": ["new::proof"]}
    else:
        payload["standard-harnesses"]["real.rs"] = [None]
    with pytest.raises(ValueError):
        ci.inventory_names(payload)


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Complete - 0 successfully verified harnesses, 0 failures, 0 total.",
        success("other::proof"),
        success("core::proof") + success("core::proof"),
        success("core::proof").replace("1 of 1", "0 of 1"),
        success("core::proof").replace("SUCCESSFUL", "FAILED"),
        success("core::proof").replace("0 failures", "1 failures"),
    ],
)
def test_zero_partial_foreign_and_failed_verdicts_never_pass(output):
    assert not ci.check_verdict(output, "core::proof", 1)


def test_verdict_requires_both_api_instances_and_keeps_expected_unreachable_assertions():
    assert ci.check_verdict(success(ci.API_HARNESS, 2), ci.API_HARNESS, 2)
    assert not ci.check_verdict(success(ci.API_HARNESS), ci.API_HARNESS, 2)
    assert ci.check_verdict(success("p") + " ** 0 of 10 failed (2 unreachable)\n", "p", 1)


def test_model_conversion_warning_cannot_qualify_as_a_successful_proof():
    warning = (
        "file native/src/flash_support.rs line 183: call to Iterator::next: "
        "not enough arguments, inserting non-deterministic value\n"
    )
    assert not ci.check_verdict(warning + success("core::proof"), "core::proof", 1)
    assert ci.check_verdict("warning: unused import\n" + success("core::proof"), "core::proof", 1)


def parser_loops():
    return f"Checking harness {ci.PARSER_HARNESS}...\n" + "".join(
        f"Loop {symbol}.{number}:\n  file source.rs line 1 column 1 function {pretty}\n"
        for symbol, pretty, numbers in [
            ("rust_utf8", "core::str::validations::run_utf8_validation", range(3)),
            ("rust_parser", "flash_support::parse_layer_slots", range(2)),
        ]
        for number in numbers
    )


def test_parser_loop_bounds_resolve_pretty_names_instead_of_fixed_mangled_symbols():
    assert ci.parser_loop_bounds(parser_loops()) == (
        "rust_utf8.0:10,rust_utf8.1:17,rust_utf8.2:4,rust_parser.0:4"
    )


@pytest.mark.parametrize("change", ["empty", "duplicate", "foreign", "missing"])
def test_parser_loop_discovery_fails_closed_on_invalid_or_ambiguous_inventory(change):
    output = parser_loops()
    if change == "empty":
        output = ""
    elif change == "duplicate":
        output += output[output.index("Loop ") :]
    elif change == "foreign":
        output = output.replace(ci.PARSER_HARNESS, "foreign::proof")
    else:
        output = output.replace("run_utf8_validation", "different_function")
    with pytest.raises(ValueError):
        ci.parser_loop_bounds(output)


def test_parser_discovery_and_verification_share_the_harness_deadline(tmp_path, monkeypatch):
    now = [0.0]
    budgets = []

    def run(command, log, timeout):
        budgets.append(timeout)
        log.write_text(parser_loops() if "--show-loops" in command else success(ci.PARSER_HARNESS))
        now[0] += 2
        return {"status": "completed", "seconds": 2, "returncode": 0}

    monkeypatch.setattr(ci.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(ci, "run_process", run)
    record = {"name": ci.PARSER_HARNESS, "command": ["cargo", "--output-format", "regular"]}
    assert ci.execute_proof(record, tmp_path / "proof.log", 10, 7)["status"] == "completed"
    assert budgets == [7, 5]
    assert record["command"][-2] == "--unwindset"
    assert record["loop_inventory"]["status"] == "completed"


def test_address_space_limit_is_really_applied(tmp_path):
    log = tmp_path / "limit.log"
    result = ci.run_process(
        [sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"],
        log,
        5,
    )
    assert result["status"] == "completed"
    assert log.read_text().strip() == f"({ci.MEMORY_BYTES}, {ci.MEMORY_BYTES})"


@pytest.mark.parametrize("outcome", ["error", "signal", "timeout"])
def test_real_process_failures_keep_their_reason(tmp_path, outcome):
    source = {
        "error": "raise SystemExit(23)",
        "signal": "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        "timeout": "import time; time.sleep(30)",
    }[outcome]
    result = ci.run_process([sys.executable, "-c", source], tmp_path / "child.log", 0.3)
    assert (
        result["status"]
        == {"error": "process_error", "signal": "signal", "timeout": "timeout"}[outcome]
    )
    assert result["returncode"] is not None
    assert result["seconds"] < 4


def test_timeout_kills_descendant_even_if_parent_exits_and_child_ignores_term(tmp_path):
    child_pid = tmp_path / "pid"
    source = (
        "import os, signal, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f" open({str(child_pid)!r}, 'w').write(str(os.getpid()))\n"
        " time.sleep(30)\n"
        "else:\n"
        " time.sleep(30)\n"
    )
    result = ci.run_process([sys.executable, "-c", source], tmp_path / "tree.log", 0.5)
    assert result["status"] == "timeout"
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 1
    while status.exists() and status.read_text().split()[2] != "Z" and time.monotonic() < deadline:
        time.sleep(0.01)  # SIGKILL delivery to a descendant is asynchronous.
    assert not status.exists() or status.read_text().split()[2] == "Z"


@pytest.fixture
def fake_cargo(tmp_path, monkeypatch):
    names = ci.required_names() + ["core::a", "core::b"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_INVENTORY", json.dumps(inventory(names)))
    monkeypatch.setenv("FAKE_LOOPS", parser_loops())
    cargo = tmp_path / "cargo"
    cargo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "if 'list' in sys.argv:\n"
        " if '-Z' not in sys.argv[sys.argv.index('list') + 1:]: sys.exit(31)\n"
        " if not os.environ.get('FAKE_NO_INVENTORY'):\n"
        "  Path('kani-list.json').write_text(os.environ['FAKE_INVENTORY'])\n"
        " sys.exit(0)\n"
        "name = sys.argv[sys.argv.index('--harness') + 1]\n"
        "if '--show-loops' in sys.argv:\n"
        " print(os.environ['FAKE_LOOPS']); sys.exit(0)\n"
        "if name.endswith('tuning_never_selects_missing_or_slower_candidate') and '--verbose' in sys.argv: sys.exit(101)\n"
        "if os.environ.get('FAKE_SLEEP'):\n"
        " Path('started').write_text(str(os.getpid()))\n"
        " time.sleep(30)\n"
        "if name == os.environ.get('FAKE_EMPTY'): sys.exit(0)\n"
        f"if name == {ci.PARSER_HARNESS!r}: print('  - Stub: super :: parse_slot_capacity -> proof_two_byte_capacity')\n"
        "count = 2 if '--tests' in sys.argv else 1\n"
        "for _ in range(count):\n"
        " print(f'Checking harness {name}...')\n"
        " print('VERIFICATION:- SUCCESSFUL')\n"
        "print(f'Complete - {count} successfully verified harnesses, 0 failures, {count} total.')\n"
        "sys.exit(23 if name == os.environ.get('FAKE_FAIL') else 0)\n"
    )
    cargo.chmod(0o755)
    return tmp_path


@pytest.mark.parametrize("failure", [None, "FAKE_FAIL", "FAKE_EMPTY"])
def test_cli_runs_remaining_proofs_but_aggregate_rejects_any_failure(
    fake_cargo, monkeypatch, failure
):
    if failure:
        monkeypatch.setenv(failure, "core::a")
    result = ci.main(["--shard", "core", "--report-dir", "results"])
    report = json.loads(Path("results/report.json").read_text())
    assert (result == 0) == (failure is None)
    assert report["discovered"] == len(ci.required_names()) + 2
    assert len(report["proofs"]) == len(ci.select_names(ci.required_names(), "core")) + 3
    assert report["proofs"][-1]["status"] == "success"  # Both API instances really ran.
    if failure:
        assert report["status"] == "failure"
        assert next(r for r in report["proofs"] if r["name"] == "core::a")["status"] != "success"


def test_stale_inventory_cannot_turn_missing_listing_into_success(fake_cargo, monkeypatch):
    Path("kani-list.json").write_text(os.environ["FAKE_INVENTORY"])
    monkeypatch.setenv("FAKE_NO_INVENTORY", "1")
    assert ci.main(["--shard", "core", "--report-dir", "results"]) == 1
    report = json.loads(Path("results/report.json").read_text())
    assert report["status"] == "failure"
    assert report["proofs"] == []


@pytest.mark.parametrize("invalid_inventory", [False, True])
def test_parser_cli_checks_discovery_and_keeps_conversion_companions_required(
    fake_cargo, monkeypatch, invalid_inventory
):
    if invalid_inventory:
        monkeypatch.setenv("FAKE_LOOPS", "no compiled loops")
    assert (ci.main(["--shard", "flash", "--report-dir", "results"]) == 0) != invalid_inventory
    report = json.loads(Path("results/report.json").read_text())
    parser = next(r for r in report["proofs"] if r["name"] == ci.PARSER_HARNESS)
    assert parser["status"] == ("invalid_output" if invalid_inventory else "success")
    companions = [r for r in report["proofs"] if "::slot_capacity_" in r["name"]]
    assert len(companions) == 3 and all(r["status"] == "success" for r in companions)
    assert Path("results", parser["loop_inventory"]["log"]).exists()


def test_speculative_cli_avoids_kani_068_verbose_statistics_crash(fake_cargo):
    assert ci.main(["--shard", "speculative", "--report-dir", "results"]) == 0
    report = json.loads(Path("results/report.json").read_text())
    names = ci.inventory_names(json.loads(os.environ["FAKE_INVENTORY"]))
    expected = ci.select_names(names, "speculative")
    assert expected
    assert sorted(record["name"] for record in report["proofs"]) == sorted(expected)
    assert all(r["status"] == "success" for r in report["proofs"])


def test_whole_job_deadline_marks_pending_proofs_failed(tmp_path):
    records = [{"name": "p", "status": "pending"}]
    report = {"status": "incomplete", "proofs": records}
    assert not ci.run_proofs(records, tmp_path, report, tmp_path / "report.json", 1, 0)
    assert records[0]["status"] == "budget_exhausted"


def test_termination_saves_incomplete_verdict_and_reaps_child(fake_cargo, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1")
    runner = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/kani_ci.py"),
            "--shard",
            "core",
            "--report-dir",
            "results",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not Path("started").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Path("started").exists()
        pid = int(Path("started").read_text())
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=5) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        report = json.loads(Path("results/report.json").read_text())
        assert report["status"] == "failure"
        assert report["proofs"][0]["status"] == "interrupted"
        assert report["proofs"][1]["status"] == "pending"
    finally:
        if runner.poll() is None:
            runner.kill()
        runner.wait()


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "5401"])
def test_invalid_time_budgets_rejected_before_work(duration):
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        ci.positive_seconds(duration)


def test_only_the_documented_panicking_unreachable_stub_is_accepted():
    notice = "  - Stub: allocate_valid_levels -> reject_unreachable_dp\n"
    assert ci.check_verdict(
        success(ci.INVALID_BUDGET_HARNESS) + notice, ci.INVALID_BUDGET_HARNESS, 1
    )
    assert not ci.check_verdict(success(ci.INVALID_BUDGET_HARNESS), ci.INVALID_BUDGET_HARNESS, 1)
    assert not ci.check_verdict(success("other::proof") + notice, "other::proof", 1)


def test_parser_accepts_only_its_exact_conversion_stub():
    notice = "  - Stub: super :: parse_slot_capacity -> proof_two_byte_capacity\n"
    assert ci.check_verdict(success(ci.PARSER_HARNESS) + notice, ci.PARSER_HARNESS, 1)
    assert not ci.check_verdict(success(ci.PARSER_HARNESS), ci.PARSER_HARNESS, 1)
    assert not ci.check_verdict(success("other::proof") + notice, "other::proof", 1)
    assert not ci.check_verdict(success(ci.PARSER_HARNESS) + notice * 2, ci.PARSER_HARNESS, 1)
