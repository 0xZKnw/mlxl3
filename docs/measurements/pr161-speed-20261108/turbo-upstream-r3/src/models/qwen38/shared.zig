//! Qwen3.8-Flash-Next's gated shared expert and its softmax router inside the routed decode launches (`root.moeFusedShared`,
//! `moeFusedSharedRouted`): the shared heads on form 9's launches (gu3s / gu8s / fin7s / down3s / dfin7s, the fin fold
//! gu3sf), the heads standalone beside them, the v2 heads, and the router folded into form 9's routed prepare launch.
//! `SharedHeads` holds their kernels and configs beside a geometry's `decode.Kernels3` (the routed decode's launches,
//! shapes and configs, which every method takes): each launch the routed launch's statements with the shared head's
//! before them, so the bytes of the host's chain.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const Global = @import("../../core/global.zig").Global;
const config = @import("../../core/config.zig");
const format = @import("../../core/format.zig");
const lane = @import("../../core/lane.zig");
const sources = @import("../../core/sources.zig");
const decode = @import("../../core/decode.zig");
const Kernels3 = decode.Kernels3;
const Kernels = decode.Kernels;
const Geometry = decode.Geometry;
const Cfg = Kernels3.Cfg;
const header = decode.header;
const header5 = decode.header5;
const freeKernel = decode.freeKernel;
const one = Kernels3.one;

// the core's texts, pieces and operand names these compose
const dfin3_inputs = decode.dfin3_inputs;
const down3_1 = decode.down3_1;
const down3_2 = decode.down3_2;
const down3_3 = decode.down3_3;
const down3_inputs = decode.down3_inputs;
const down3_outputs = decode.down3_outputs;
const down3_w = decode.down3_w;
const down_outputs = decode.down_outputs;
const expert_line = decode.expert_line;
const fin3_inputs = decode.fin3_inputs;
const fin3_outputs = decode.fin3_outputs;
const fin7_2 = decode.fin7_2;
const fin7_3 = decode.fin7_3;
const fin7_4 = decode.fin7_4;
const fin7_5 = decode.fin7_5;
const fin7_planes = decode.fin7_planes;
const fin7_sums = decode.fin7_sums;
const fin7_swiglu = decode.fin7_swiglu;
const gu3_1 = decode.gu3_1;
const gu3_2 = decode.gu3_2;
const gu3_3 = decode.gu3_3;
const gu3_4 = decode.gu3_4;
const gu3_body = decode.gu3_body;
const gu3_inputs = decode.gu3_inputs;
const gu3_outputs = decode.gu3_outputs;
const gu3_stores = decode.gu3_stores;
const gu3_w = decode.gu3_w;
const gu8_inputs = decode.gu8_inputs;
const gu8_outputs = decode.gu8_outputs;
const prep2_body = decode.prep2_body;
const prep2_inputs = decode.prep2_inputs;
const prep2_outputs = decode.prep2_outputs;
const rk_line = decode.rk_line;
const text_down3 = decode.text_down3;
const text_fin7 = decode.text_fin7;
const text_gu3 = decode.text_gu3;
const text_gu8 = decode.text_gu8;
const text_prep2 = decode.text_prep2;

/// The shared expert inside the routed launches (`SharedHeads.moeShared`): dense.zig's decode-row arithmetic (prep_s, dgemv,
/// dfinish), so the shared expert's bytes are the dense linears'.
const metal_shared = @embedFile("kernels/shared/shared.metal");

pub const header_s: [:0]const u8 = header5 ++ metal_shared;

/// The shared expert in the routed launches (`Kernels3.moeShared`), launch 1's head: grid rows from RKC on are the shared
/// gate|up GEMV's threadgroups, j = (y - RKC) * SZ + z -> projection j / (SKSG / SSPT) and the SSPT consecutive K slices from
/// (j % (SKSG / SSPT)) SSPT (a slice = one 128-block of hidden: SKSG = hidden / 128); each slice is rotated in-kernel (dense
/// prep_s) and run as dense dgemv's slice for the grid's BT output tiles, exactly as a threadgroup of its own computes it, so
/// SSPT is launch shape only. Planes szg / szu f32 [SKSG, SRS, inter], block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu_head = sgh_1 ++ sgh_2 ++ sgh_3 ++ sgh_z ++ sgh_4 ++ sgh_sums ++ sgh_5 ++ sgh_end;
/// the head's pieces, cut where gu3sf differs: after the threadgroup arrays and the thread lines (its declarations), at
/// the planes' pointer, at the input sums' store and before the end
const sgh_1 = @embedFile("kernels/shared/shared_gu_head_1.metal");
const sgh_2 = @embedFile("kernels/shared/shared_gu_head_2.metal");
const sgh_3 = @embedFile("kernels/shared/shared_gu_head_3.metal");
const sgh_z = @embedFile("kernels/shared/shared_gu_head_z.metal");
const sgh_4 = @embedFile("kernels/shared/shared_gu_head_4.metal");
const sgh_sums = @embedFile("kernels/shared/shared_gu_head_sums.metal");
const sgh_5 = @embedFile("kernels/shared/shared_gu_head_5.metal");
const sgh_end = @embedFile("kernels/shared/shared_gu_head_end.metal");
pub const text_gu3s: [:0]const u8 = shared_gu_head ++ text_gu3;
pub const text_gu8s: [:0]const u8 = shared_gu_head ++ text_gu8;
const shared_gu_inputs = [_][*:0]const u8{ "xr", "sh_gate_t", "sh_up_t", "sh_gate_suh", "sh_up_suh" };
const shared_gu_outputs = [_][*:0]const u8{ "szg", "szu", "sxsg", "sxsu" };
const gu3s_inputs = gu3_inputs ++ shared_gu_inputs;
const gu3s_outputs = gu3_outputs ++ shared_gu_outputs;
const gu8s_inputs = gu8_inputs ++ shared_gu_inputs;
const gu8s_outputs = gu8_outputs ++ shared_gu_outputs;

/// Launch 3's head: grid rows RKC.. RKC + SRS are the shared expert's rows: per 128-block of the intermediate, dense dfinish of
/// the gate and up planes (to OUT_T), the host's table SwiGLU (`T act = g * sigtab[g]; act * up`, sigtab = MLX's sigmoid of
/// every 16-bit pattern), then dense prep_s of the product for the down: sxpd half [SRS, inter] lane order, sxsd [SRS, NBI].
const shared_fin_head = sfh_1 ++ sfh_2 ++ sfh_xg ++ sfh_xu ++ sfh_3 ++ sfh_g ++ sfh_u ++ sfh_4 ++ sfh_end;
/// the head's pieces: its constants and grid lines; its body from the input sums, cut at the two sums' reads and the two
/// finishes (the lines gu3sf maps); its end
const sfh_1 = @embedFile("kernels/shared/shared_fin_head_1.metal");
const sfh_2 = @embedFile("kernels/shared/shared_fin_head_2.metal");
const sfh_xg = @embedFile("kernels/shared/shared_fin_head_xg.metal");
const sfh_xu = @embedFile("kernels/shared/shared_fin_head_xu.metal");
const sfh_3 = @embedFile("kernels/shared/shared_fin_head_3.metal");
const sfh_g = @embedFile("kernels/shared/shared_fin_head_g.metal");
const sfh_u = @embedFile("kernels/shared/shared_fin_head_u.metal");
const sfh_4 = @embedFile("kernels/shared/shared_fin_head_4.metal");
const sfh_end = @embedFile("kernels/shared/shared_fin_head_end.metal");
pub const text_fin7s: [:0]const u8 = shared_fin_head ++ text_fin7;
const sfin_inputs = [_][*:0]const u8{ "szg", "szu", "sxsg", "sxsu", "sh_gate_svh", "sh_up_svh", "sh_down_suh", "sigtab" };
const sfin_outputs = [_][*:0]const u8{ "sxpd", "sxsd" };
const fin7s_inputs = fin3_inputs ++ sfin_inputs;
const fin7s_outputs = fin3_outputs ++ sfin_outputs;

// ── fin7s folded into gu3s at the fused verify widths ──
//
// gu3sf is gu3s (the shared gate|up head, then gu3) with fin7s's work done by the threadgroups that complete it, so the fused
// shared expert's rows past gu8_rows run gu3sf, down3s, dfin7s after a counter zeroing that depends on nothing.
// - Routed: every threadgroup stores its planes through coherent pointers and counts itself at each member's (row-slot,
//   128-block of the intermediate) counter; the one that brings it to ARR (the block's GC column groups x the slot's K
//   threadgroups) runs fin7's statements for that pair in simdgroup j for member j (the input sums are the routed
//   prepare's, read as fin7 reads them).
// - Shared: every head threadgroup stores its planes through coherent pointers, publishes its slices' input sums into its
//   block's own copy (sxsx [NBI, SRS, 2, NBH], written where s_blk % S_GC == 0: one writer per value, and every writer is
//   one of the block's arrivals) and counts itself at its 128-block's counter; the one that brings it to S_ARR (the block's
//   column groups x both projections' slice groups) runs the shared fin head's statements for the block, every row
//   (simdgroup s_sg: rows s_sg, s_sg + 4, ...), with `turboexl3_sfinish`'s statements inline over the coherent planes
//   (vf_sfin_g / vf_sfin_u: its body with its parameters bound by name).
// DEC-ONE's arrival idiom: device-scope seq_cst fences around one relaxed fetch_add per threadgroup, nothing waits. The
// counters (routed [RK, NBI], then shared [NBI]) come zeroed from text_vfold_zero. fin7s's bytes, one dependent launch fewer.
pub const text_gu3sf = sgh_1 ++ vf_head_tg ++ sgh_2 ++ vf_head_decls ++ sgh_3 ++ vf_head_zc ++ sgh_4 ++ vf_head_sumsc ++ sgh_5 ++
    vf_head_arrive ++ sfh_2 ++ vf_sfin_xgc ++ vf_sfin_xuc ++ sfh_3 ++ vf_sfin_g ++ vf_sfin_u ++ sfh_4 ++ vf_head_close ++ sgh_end ++
    gu3_1 ++ rk_line ++ gu3_2 ++ vf_gu3_decls ++ gu3_w ++ gu3_3 ++ vf_gu3_stores ++ gu3_4 ++ vf_arrive ++
    expert_line ++ fin7_2 ++ vf_fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5 ++ vf_fin7_close;
const vf_head_tg = @embedFile("kernels/shared/vf_head_tg.metal");
const vf_head_decls = @embedFile("kernels/shared/vf_head_decls.metal");
const vf_head_zc = @embedFile("kernels/shared/vf_head_zc.metal");
const vf_head_sumsc = @embedFile("kernels/shared/vf_head_sumsc.metal");
const vf_head_arrive = @embedFile("kernels/shared/vf_head_arrive.metal");
const vf_sfin_xgc = @embedFile("kernels/shared/vf_sfin_xgc.metal");
const vf_sfin_xuc = @embedFile("kernels/shared/vf_sfin_xuc.metal");
const vf_sfin_g = @embedFile("kernels/shared/vf_sfin_g.metal");
const vf_sfin_u = @embedFile("kernels/shared/vf_sfin_u.metal");
const vf_head_close = @embedFile("kernels/shared/vf_head_close.metal");
const vf_gu3_decls = @embedFile("kernels/shared/vf_gu3_decls.metal");
const vf_gu3_stores = @embedFile("kernels/shared/vf_gu3_stores.metal");
const vf_arrive = @embedFile("kernels/shared/vf_arrive.metal");
const vf_fin7_planes = @embedFile("kernels/shared/vf_fin7_planes.metal");
const vf_fin7_close = @embedFile("kernels/shared/vf_fin7_close.metal");
/// gu3s's inputs, then fin7s's (less the planes and the shared sums gu3sf keeps) and the counters
const gu3sf_inputs = gu3s_inputs ++ [_][*:0]const u8{ "gate_svh", "up_svh", "down_suh", "xsg", "xsu", "sh_gate_svh", "sh_up_svh", "sh_down_suh", "sigtab", "cnt" };
/// fin7s's outputs (hd, hs, sxpd, sxsd), then the routed and shared planes and the shared input sums' block copies (scratch)
const gu3sf_outputs = fin7s_outputs ++ [_][*:0]const u8{ "zg", "zu", "szg", "szu", "sxsx" };
/// gu3sf's arrival counters, zeroed by a launch nothing it reads depends on (gate_svh is a weight, taken only for
/// ordering), so MLX needs no barrier before it: dense's DEC-ONE zeroing text
pub const text_vfold_zero = @embedFile("../../core/kernels/dense/dec_zero.metal");
const vfold_zero_inputs = [_][*:0]const u8{"gate_svh"};
const vfold_zero_outputs = [_][*:0]const u8{"cnt"};

/// Launch 4's head: grid rows from RKC on are the shared down GEMV's threadgroups, each over the SSPTD consecutive slices from
/// ((y - RKC) * SZ + z) SSPTD of SKSD (dense dgemv over sxpd, every slice as a threadgroup of its own computes it), planes
/// szd f32 [SKSD, SRS, hidden].
const shared_down_head = @embedFile("kernels/shared/shared_down_head.metal");
pub const text_down3s: [:0]const u8 = shared_down_head ++ text_down3;
const sdown_inputs = [_][*:0]const u8{ "sxpd", "sh_down_t" };
const sdown_outputs = [_][*:0]const u8{"szd"};
const down3s_inputs = down3_inputs ++ sdown_inputs;
const down3s_outputs = [_][*:0]const u8{ "zd", "szd" };

/// Launch 5 with the shared expert: dfin7's slots in simdgroups 0..TOPK-1 and the shared down's dense dfinish in simdgroup TOPK,
/// then per output the routed sum in OUT_T plus the host's gate tail in OUT_T: sigmoid(logit) (MLX's, by table) * down, then
/// the sum (MLX's Multiply and Add), so y = moeSharedGateTail(routed, logit, down)'s bytes.
pub const text_dfin7s = @embedFile("kernels/shared/dfin7s.metal");
const dfin7s_inputs = dfin3_inputs ++ [_][*:0]const u8{ "szd", "sxsd", "sh_down_svh", "sigtab", "slogit" };

/// The v2 shared heads' GEMV helpers (`Kernels3.moeShared2` / `moeSharedSide2`), appended to `header_s` (`header_t`). One
/// simdgroup per (K slice, OTS consecutive output tiles) computes dense dgemv's plane values of its tiles by itself: the
/// slice's four parts (dgemv's simdgroups: PER input tiles each) in dgemv's order (tile order, weights m = 0..7), each part
/// from zero and folded over the quad lanes (dgemv's `turboexl3_fold`), the slice the left sum of its parts (dgemv's plane
/// store ((part 0 + part 1) + part 2) + part 3). The planes are dgemv's bit for bit with no threadgroup reduction and no
/// second barrier; the finishes keep dense dfinish's order (plane 0, then + plane k in slice order) and read the planes in
/// the natural layout or lane-major (LM: one float4 per plane and lane). The parts take every codeword from the window's
/// 32-bit words (`turboexl3_scw`, exl3_tile_cw's 96-bit rules) for both window readers: no 64-bit window (`Exl3BitsM<1>`),
/// which the Metal back-end compiler failed to build in the down head at 2+ rows. Same bits.
const metal_shared2 = @embedFile("kernels/shared/shared2.metal");

pub const header_t: [:0]const u8 = header_s ++ metal_shared2;

/// The v2 heads' shared-threadgroup test and id (`moeShared2`): with SYOFF 0 the grid rows from RKC on are the shared
/// expert's (the routed rows first, as v1); SYOFF > 0 puts the SYS shared rows first (the routed body's rk = y - SYOFF,
/// `text_*_y`). The shared threadgroups are numbered z-major (z * SYS + row) * SX + x, the dispatch order of the grid's slabs.
const shared2_prologue = @embedFile("kernels/shared/shared2_prologue.metal");

/// Launch 1's v2 head, the gate|up GEMV with the input rotated in-kernel (form 9's gu8 launch, gu3 without `sprep`, and the
/// standalone beside variant): shared threadgroup s_id = (K slice, group of 4 SOTS output tiles of the 2 inter / 16 tiles,
/// gate's then up's); simdgroup sg owns the group's SOTS tiles from (4 group + sg) SOTS. It walks the slice part by part
/// (dgemv's four simdgroup shares), its weights SPRE input tiles ahead of the decode in whole parts (at most 3 parts, at
/// most 16 tiles of raw words in flight); the first ones are issued before the threadgroup rotates the slice's 128-block
/// (dense prep_s: one simdgroup per (projection, row) the threadgroup reads, the block sums written by each projection's
/// first threadgroup) and its barrier. The plane values are stored straight from the quads. Planes szg / szu f32 [SKSG,
/// SRS, inter] (layout SLM), block sums sxsg / sxsu [SRS, hidden / 128].
const shared_gu2_head = @embedFile("kernels/shared/shared_gu2_head_1.metal") ++
    shared2_prologue ++
    @embedFile("kernels/shared/shared_gu2_head_2.metal");

/// Launch 1's v2 head with the input prepared once by the routed prepare launch (`shared_prep_head`, form 9 from two rows,
/// `sprep`): `shared_gu2_head` reading the lane-order rows sxg / sxu half [SRS, hidden] from device memory, so no rotation,
/// no threadgroup memory and no barrier. Planes szg / szu.
const shared_gu2p_head = shared2_prologue ++ @embedFile("kernels/shared/shared_gu2p_head.metal");

/// Launch 0's head for `shared_gu2p_head` (prep2 from two rows; standalone beside): grid rows from RKC on are the shared
/// expert's rows, one simdgroup per (128-block of hidden, row) running dense prep_s for gate and up: sxg / sxu half [SRS,
/// hidden] lane order, the block sums sxsg / sxsu [SRS, hidden / 128].
const shared_prep_head = @embedFile("kernels/shared/shared_prep_head.metal");

/// Launch 3's v2 head: `shared_fin_head` with the planes read in layout SLM (`turboexl3_sfinish2`).
const shared_fin2_head = @embedFile("kernels/shared/shared_fin2_head.metal");

/// Launch 4's v2 head: shared threadgroup s_id = (K slice of SKSD, group of 4 SOTS of the hidden / 16 output tiles), simdgroup
/// sg owning SOTS tiles from (4 group + sg) SOTS over the slice's four parts of sxpd (lane order, device memory): no
/// threadgroup memory, no barrier. Planes szd f32 [SKSD, SRS, hidden] (layout SLM).
const shared_down2_head = shared2_prologue ++ @embedFile("kernels/shared/shared_down2_head.metal");

/// Launch 5 with the v2 planes: `text_dfin7s` with the shared down's finish reading layout SLM.
pub const text_dfin7t = @embedFile("kernels/shared/dfin7t.metal");

const rk_line_y = @embedFile("kernels/shared/rk_line_y.metal");
const text_gu8_y: [:0]const u8 = @embedFile("../../core/kernels/decode/gu8_1.metal") ++ rk_line_y ++ @embedFile("../../core/kernels/decode/gu8_2.metal");
const text_gu3_y: [:0]const u8 = gu3_1 ++ rk_line_y ++ gu3_body;
const text_down3_y: [:0]const u8 = down3_1 ++ rk_line_y ++ down3_2 ++ down3_w ++ down3_3;
pub const text_gu8t: [:0]const u8 = shared_gu2_head ++ text_gu8_y;
pub const text_gu3t: [:0]const u8 = shared_gu2_head ++ text_gu3_y;
pub const text_gu3p: [:0]const u8 = shared_gu2p_head ++ text_gu3_y;
pub const text_prep2t: [:0]const u8 = shared_prep_head ++ text_prep2;
pub const text_fin7t: [:0]const u8 = shared_fin2_head ++ text_fin7;
pub const text_down3t: [:0]const u8 = shared_down2_head ++ text_down3_y;
const shared_gu2p_inputs = [_][*:0]const u8{ "sxg", "sxu", "sh_gate_t", "sh_up_t" };
const shared_gu2p_outputs = [_][*:0]const u8{ "szg", "szu" };
const shared_prep_inputs = [_][*:0]const u8{ "x", "sh_gate_suh", "sh_up_suh" };
const shared_prep_outputs = [_][*:0]const u8{ "sxg", "sxu", "sxsg", "sxsu" };
const gu3p_inputs = gu3_inputs ++ shared_gu2p_inputs;
const gu3p_outputs = gu3_outputs ++ shared_gu2p_outputs;
const prep2t_inputs = prep2_inputs ++ [_][*:0]const u8{ "sh_gate_suh", "sh_up_suh" };
const prep2t_outputs = prep2_outputs ++ shared_prep_outputs;

// ── The host's softmax router folded into form 9's routed prepare launch ──

/// The host's fused softmax router (transformer.zig `moeRouterSource(.softmax)`, the MLX custom kernel
/// "mlxserve_moe_router_softmax"), byte for byte: the GPU test runs it as the chain's router, the merged launch below takes
/// its statements from it (its pieces), and the host's test pins this copy to its own source
/// (`exl3_text_router_softmax`). Inputs logits; outputs inds, scores; template TOUT, NE, NK, TG, NORM (the host's
/// config: TG = 256 at 512 experts).
pub const text_router_softmax: [:0]const u8 = @embedFile("kernels/shared/router_softmax_1.metal") ++ router_softmax ++ router_sel ++
    @embedFile("kernels/shared/router_softmax_4.metal") ++ router_weights ++ @embedFile("kernels/shared/router_softmax_6.metal");
pub const router_inputs = [_][*:0]const u8{"logits"};
pub const router_outputs = [_][*:0]const u8{ "inds", "scores" };

/// `parts` joined and 0-terminated (a kernel source).
fn textJoin(comptime parts: []const []const u8) [:0]const u8 {
    comptime {
        var out: []const u8 = "";
        for (parts) |p| out = out ++ p;
        var buf: [out.len:0]u8 = undefined;
        @memcpy(buf[0..out.len], out);
        const final = buf;
        return &final;
    }
}

/// The router's selection (simdgroup 0's K rounds of max-then-mask), its softmax tree and its weights (renormalized, then
/// written with the slots), and prep2's body after its index lines: verbatim slices of their kernels' texts.
const router_sel = @embedFile("kernels/shared/router_softmax_3.metal");
const router_softmax = @embedFile("kernels/shared/router_softmax_2.metal");
const router_weights = @embedFile("kernels/shared/router_softmax_5.metal");

/// The shared gate's logit for one row exactly as MLX computes it at the verify widths, as one simdgroup's part
/// (`gx` x [R, GK] bf16, `gw` the gate's weight vector [GK] bf16, `gp` the 8 partials): MLX 0.32's `matmul` of x and the gate's
/// [GK, 1] at R = 2..4 (GK > 64, GK >= 16 R) runs gemv_bfloat16_bm1_bn8_sm1_sn32_tm{1,4}_tn4 (gemv.h GEMVKernel::run), whose
/// simdgroup gs of 8 and lane sum the columns (32 gs + lane) * 4 + 1024 i + tn in that order, the leftover block zero-filled
/// past GK (load_safe), then fold a shuffle-down tree; this simdgroup plays the gemv's simdgroup gs = tid / 32 - 1. The host's
/// router mode `softmax_gate_gemv` splices the same text (its test pins it to this one).
pub const text_gate_gemv = sources.gate_gemv;
/// The gate's sum of the 8 partials as the gemv's simdgroup 0 lane 0 folds them (its own first, then 1..7 in order), and its
/// bf16 store.
pub const text_gate_sum = sources.gate_sum;

/// The router kernel and form 3's prepare (`text_prep2`) as one launch, one threadgroup per (128-wide hidden block b, row)
/// of NSG = TG / 32 simdgroups (`Kernels3.routeSimdgroups`). The keys are staged as the router stages them; then phase
/// 1: simdgroup 0 runs the router's selection on them (its K rounds, verbatim; every block of the row repeats it, so no
/// block waits on another), and in block 0 simdgroup NSG - 1 runs the router's softmax tree (verbatim) beside it,
/// handing smax / snorm on through threadgroup memory, and with the gate simdgroups 1..8 take the shared gate's dot.
/// Phase 2: simdgroup j < K prepares row-slot (row, j) of block b with prep2's statements, its expert the selection's
/// j-th pick; in block 0 simdgroup NSG - 1 runs the router's weights and writes the slots and scores (verbatim), and
/// with the gate sums the gate's partials. Every value a statement of the router or of prep2 computes, on the same
/// lanes: their bytes.
fn routePrepText(comptime gate: bool) [:0]const u8 {
    return textJoin(&.{
        @embedFile("kernels/shared/route_prep_1.metal"),
        if (gate) @embedFile("kernels/shared/route_prep_gp.metal") else "",
        @embedFile("kernels/shared/route_prep_2.metal"),
        router_sel,
        @embedFile("kernels/shared/route_prep_3.metal"),
        router_softmax,
        @embedFile("kernels/shared/route_prep_4.metal"),
        if (gate) @embedFile("kernels/shared/route_prep_close_gate.metal") ++ text_gate_gemv else @embedFile("kernels/shared/route_prep_close.metal"),
        @embedFile("kernels/shared/route_prep_5.metal"),
        prep2_body,
        @embedFile("kernels/shared/route_prep_6.metal"),
        router_weights,
        if (gate) @embedFile("kernels/shared/route_prep_gsum_open.metal") ++ text_gate_sum ++ @embedFile("kernels/shared/route_prep_gsum_close.metal") else "",
        @embedFile("kernels/shared/route_prep_7.metal"),
    });
}
pub const text_route_prep: [:0]const u8 = routePrepText(false);
pub const text_route_prepg: [:0]const u8 = routePrepText(true);
const route_prep_inputs = [_][*:0]const u8{ "logits", "x", "gate_suh", "up_suh" };
const route_prep_outputs = [_][*:0]const u8{ "xg", "xu", "sg", "su", "inds", "scores" };
const route_prepg_inputs = route_prep_inputs ++ [_][*:0]const u8{ "gx", "gw" };
const route_prepg_outputs = route_prep_outputs ++ [_][*:0]const u8{"glogit"};

/// The shared expert `moeShared` folds into the routed launches: three dense EXL3 linears (gate, up: hidden -> inter;
/// down: inter -> hidden; trellis U16 [in/16, out/16, n], suh [in], svh [out]) as dense.zig serves them (`n_*` their
/// rates, `ks_*` `Dense.ksFor`'s K splits, `defer_affine` its MUL1 deferral), the host's SwiGLU table (OUT_T [65536]:
/// MLX's sigmoid of every 16-bit pattern) and the shared gate's logit per row (OUT_T [rows]).
pub const SharedBank = struct {
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    n_gu: u32,
    n_dn: u32,
    ks_gu: u32,
    ks_dn: u32,
    defer_affine: bool,
    sigtab: mlx.mlx_array,
    logit: mlx.mlx_array,
};
/// verify widths the shared heads carry (their GEMVs keep every row's accumulators in one threadgroup, as dgemv's RB)
pub const shared_rows_max: usize = 4;
const SKey = struct { n_gu: u32, n_dn: u32, ks_gu: u32, ks_dn: u32, def: bool, sp_gu: u32, sp_dn: u32 };
/// `want` slices per threadgroup when it divides the shared K split, else 1
fn sharedSpt(ks_: u32, want: u32) u32 {
    return if (want >= 1 and ks_ % want == 0) want else 1;
}
/// `vf`: the configs of gu3sf and of its counter zeroing (rows past gu8_rows)
const SCfg = struct { gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, gu8: bool, vf: ?VFold = null };
const VFold = struct { gu: mlx.mlx_fast_metal_kernel_config, zero: mlx.mlx_fast_metal_kernel_config };
fn freeSCfg(c: SCfg) void {
    inline for (.{ c.gu, c.fin, c.down, c.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
    if (c.vf) |vf| inline for (.{ vf.gu, vf.zero }) |k| {
        _ = mlx.mlx_fast_metal_kernel_config_free(k);
    };
}
const SSCfg = struct { gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
/// the v2 configs' key: the shared bank's shape and every v2 knob the launches read (`sots` / `sotsd` as resolved per row)
const TKey = struct { n_gu: u32, n_dn: u32, ks_gu: u32, ks_dn: u32, def: bool, ots: u32, otsd: u32, prep: bool, lm: bool, pre: u32, first: bool, gu8_rows: usize };
/// inside: the optional prepare (prep2t), gate|up (gu8t / gu3t / gu3p), fin7t, down3t, dfin7t; `gu8`: launch 1 is gu8t
const TCfg = struct { prep: ?mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, gu8: bool };
/// beside: the optional standalone prepare (sprep2, then sgu2p; else sgu2), sfin2, sdown2
const TSCfg = struct { prep: ?mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config };
fn freeTCfg(c: TCfg) void {
    if (c.prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
    inline for (.{ c.gu, c.fin, c.down, c.dfin }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
}
fn freeTSCfg(c: TSCfg) void {
    if (c.prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
    inline for (.{ c.gu, c.fin, c.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
}
/// `want` output tiles per v2 GEMV simdgroup at `rows` rows: a value whose simdgroup tiles would leave one projection
/// (`per_proj` tiles each) or whose groups of four would not tile the `nproj` projections runs 1 (always admitted: the
/// widths are 128-aligned); an admitted value whose accumulators would pass 64 floats per lane (the part's sums and the
/// slice's, rows x tiles x 2 columns each: rows x tiles > 16) runs the largest admitted value within that budget, so no
/// setting hands the back-end compiler a huge live state
pub fn sharedOts(per_proj: u32, nproj: u32, want: u32, rows: u32) u32 {
    const tiling = struct {
        fn ok(pp: u32, np: u32, c: u32) bool {
            return c >= 1 and pp % c == 0 and (np * pp) % (4 * c) == 0;
        }
    };
    if (!tiling.ok(per_proj, nproj, want)) return 1;
    var c: u32 = want;
    while (c > 1 and (rows * c > 16 or !tiling.ok(per_proj, nproj, c))) c -= 1;
    return c;
}


/// A geometry's shared heads beside its `Kernels3`
pub const SharedHeads = struct {
    /// the shared expert's K slices per GEMV threadgroup, gate|up (sspt, a divisor of its 20 slices) and down (ssptd, of its 5);
    /// launch shape only (every slice computed as a threadgroup of its own computes it). Set before the first call.
    sspt: u32 = 1,
    ssptd: u32 = 1,

    /// the shared expert inside form 9's launches (`moeShared`): the heads on gu3 / gu8 / fin7 / down3 and dfin7s
    gu3s: mlx.mlx_fast_metal_kernel,
    gu8s: mlx.mlx_fast_metal_kernel,
    fin7s: mlx.mlx_fast_metal_kernel,
    down3s: mlx.mlx_fast_metal_kernel,
    dfin7s: mlx.mlx_fast_metal_kernel,
    scfg: [shared_rows_max + 1]?SCfg = @splat(null),
    scfg_key: [shared_rows_max + 1]SKey = undefined,
    /// at the fused shared expert's rows past gu8_rows (the verify widths), gu3s and fin7s run as one launch (gu3sf: the
    /// threadgroup that completes a routed (row-slot, 128-block) runs fin7's statements for it, the one that completes a
    /// shared 128-block runs the shared fin head's for every row), after a counter zeroing that depends on nothing:
    /// fin7s's bytes, one dependent launch fewer (`Config.fused_fin_fold`). The kernels are built on first use.
    fused_fin_fold: bool = true,
    gu3sf: ?mlx.mlx_fast_metal_kernel = null,
    vzero: ?mlx.mlx_fast_metal_kernel = null,
    /// the shared expert in its own launches beside form 9's (`moeSharedSide`): the three heads standalone (RKC 0, so every
    /// grid row is the shared expert's) and the routed kernels untouched, each with its own threadgroup memory and registers
    sgu: mlx.mlx_fast_metal_kernel,
    sfin: mlx.mlx_fast_metal_kernel,
    sdown: mlx.mlx_fast_metal_kernel,
    sscfg: [shared_rows_max + 1]?SSCfg = @splat(null),
    sscfg_key: [shared_rows_max + 1]SKey = undefined,
    /// the v2 shared heads (`moeShared2` / `moeSharedSide2`, `sheads` 2; 1 runs `moeShared` / `moeSharedSide`). Launch shape
    /// only, the dense planes' bytes under every option: `sots` / `sotsd` the gate|up / down output tiles per GEMV simdgroup
    /// at 1, 2, 3 and 4+ rows (`sharedOts` clamps them); `sprep` the gate|up input prepared once in the routed prepare launch
    /// where form 9 runs one (else rotated in every threadgroup); `slm` lane-major planes (a float4 per plane and lane in the
    /// finishes); `spre` (per row count) a GEMV simdgroup's weight prefetch distance in input tiles, in whole parts of the
    /// slice, at most 3 parts and 16 tiles in flight; `sfirst` the shared threadgroups ahead of the routed ones in the gate|up
    /// and down grids. Set before the first call.
    sheads: u8 = 1,
    sots: [4]u32 = .{ 2, 2, 1, 1 },
    sotsd: [4]u32 = .{ 2, 2, 1, 1 },
    sprep: bool = true,
    slm: bool = true,
    spre: [4]u32 = .{ 2, 2, 2, 2 },
    sfirst: bool = false,
    /// the v2 heads on form 9's launches: gu8t (rotating head + gu8), gu3t (rotating head + gu3), gu3p (prepared-input head +
    /// gu3, after prep2t = prep2 + the shared prepare), fin7t, down3t, dfin7t; beside them: sgu2, sgu2p, sprep2, sfin2, sdown2
    gu8t: mlx.mlx_fast_metal_kernel,
    gu3t: mlx.mlx_fast_metal_kernel,
    gu3p: mlx.mlx_fast_metal_kernel,
    prep2t: mlx.mlx_fast_metal_kernel,
    fin7t: mlx.mlx_fast_metal_kernel,
    down3t: mlx.mlx_fast_metal_kernel,
    dfin7t: mlx.mlx_fast_metal_kernel,
    sgu2: mlx.mlx_fast_metal_kernel,
    sgu2p: mlx.mlx_fast_metal_kernel,
    sprep2: mlx.mlx_fast_metal_kernel,
    sfin2: mlx.mlx_fast_metal_kernel,
    sdown2: mlx.mlx_fast_metal_kernel,
    tcfg: [shared_rows_max + 1]?TCfg = @splat(null),
    tcfg_key: [shared_rows_max + 1]TKey = undefined,
    tscfg: [shared_rows_max + 1]?TSCfg = @splat(null),
    tscfg_key: [shared_rows_max + 1]TKey = undefined,
    /// The host's softmax router folded into form 9's routed prepare launch (`routeLaunch`, `moeSharedRouted`): the
    /// plain kernel and the one that also computes the shared gate's logit, built on first use, and one config per row count
    /// keyed on the launch's experts, scores dtype and gate width (`RouteKey`).
    route_prep: ?mlx.mlx_fast_metal_kernel = null,
    route_prepg: ?mlx.mlx_fast_metal_kernel = null,
    route_cfg: [Kernels.max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    route_key: [Kernels.max_rows + 1]RouteKey = undefined,

    /// The shared heads' kernels (the ones built on first use come later), launch-shape knobs at their defaults
    pub fn init() !SharedHeads {
        return .{
            .gu3s = ks.newKernel("turboexl3_gu3s", &gu3s_inputs, &gu3s_outputs, text_gu3s, header_s, true) orelse return error.KernelCreateFailed,
            .gu8s = ks.newKernel("turboexl3_gu8s", &gu8s_inputs, &gu8s_outputs, text_gu8s, header_s, true) orelse return error.KernelCreateFailed,
            .fin7s = ks.newKernel("turboexl3_fin7s", &fin7s_inputs, &fin7s_outputs, text_fin7s, header_s, true) orelse return error.KernelCreateFailed,
            .down3s = ks.newKernel("turboexl3_down3s", &down3s_inputs, &down3s_outputs, text_down3s, header_s, true) orelse return error.KernelCreateFailed,
            .dfin7s = ks.newKernel("turboexl3_dfin7s", &dfin7s_inputs, &down_outputs, text_dfin7s, header_s, true) orelse return error.KernelCreateFailed,
            .sgu = ks.newKernel("turboexl3_sgu", &shared_gu_inputs, &shared_gu_outputs, shared_gu_head, header_s, true) orelse return error.KernelCreateFailed,
            .sfin = ks.newKernel("turboexl3_sfin", &sfin_inputs, &sfin_outputs, shared_fin_head, header_s, true) orelse return error.KernelCreateFailed,
            .sdown = ks.newKernel("turboexl3_sdown", &sdown_inputs, &sdown_outputs, shared_down_head, header_s, true) orelse return error.KernelCreateFailed,
            .gu8t = ks.newKernel("turboexl3_gu8t", &gu8s_inputs, &gu8s_outputs, text_gu8t, header_t, true) orelse return error.KernelCreateFailed,
            .gu3t = ks.newKernel("turboexl3_gu3t", &gu3s_inputs, &gu3s_outputs, text_gu3t, header_t, true) orelse return error.KernelCreateFailed,
            .gu3p = ks.newKernel("turboexl3_gu3p", &gu3p_inputs, &gu3p_outputs, text_gu3p, header_t, true) orelse return error.KernelCreateFailed,
            .prep2t = ks.newKernel("turboexl3_prep2t", &prep2t_inputs, &prep2t_outputs, text_prep2t, header_t, true) orelse return error.KernelCreateFailed,
            .fin7t = ks.newKernel("turboexl3_fin7t", &fin7s_inputs, &fin7s_outputs, text_fin7t, header_t, true) orelse return error.KernelCreateFailed,
            .down3t = ks.newKernel("turboexl3_down3t", &down3s_inputs, &down3s_outputs, text_down3t, header_t, true) orelse return error.KernelCreateFailed,
            .dfin7t = ks.newKernel("turboexl3_dfin7t", &dfin7s_inputs, &down_outputs, text_dfin7t, header_t, true) orelse return error.KernelCreateFailed,
            .sgu2 = ks.newKernel("turboexl3_sgu2", &shared_gu_inputs, &shared_gu_outputs, shared_gu2_head, header_t, true) orelse return error.KernelCreateFailed,
            .sgu2p = ks.newKernel("turboexl3_sgu2p", &shared_gu2p_inputs, &shared_gu2p_outputs, shared_gu2p_head, header_t, true) orelse return error.KernelCreateFailed,
            .sprep2 = ks.newKernel("turboexl3_sprep2", &shared_prep_inputs, &shared_prep_outputs, shared_prep_head, header_t, true) orelse return error.KernelCreateFailed,
            .sfin2 = ks.newKernel("turboexl3_sfin2", &sfin_inputs, &sfin_outputs, shared_fin2_head, header_t, true) orelse return error.KernelCreateFailed,
            .sdown2 = ks.newKernel("turboexl3_sdown2", &sdown_inputs, &sdown_outputs, shared_down2_head, header_t, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *SharedHeads) void {
        for (&self.scfg) |*c| if (c.*) |cc| {
            freeSCfg(cc);
            c.* = null;
        };
        inline for (.{ self.gu3sf, self.vzero }) |k| if (k) |kk| {
            freeKernel(kk);
        };
        for (&self.sscfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.gu, cc.fin, cc.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.tcfg) |*c| if (c.*) |cc| {
            freeTCfg(cc);
            c.* = null;
        };
        for (&self.tscfg) |*c| if (c.*) |cc| {
            freeTSCfg(cc);
            c.* = null;
        };
        for (&self.route_cfg) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        inline for (.{ self.route_prep, self.route_prepg }) |k| if (k) |kk| {
            freeKernel(kk);
        };
        inline for (.{ self.gu3s, self.gu8s, self.fin7s, self.down3s, self.dfin7s, self.sgu, self.sfin, self.sdown }) |k| freeKernel(k);
        inline for (.{ self.gu8t, self.gu3t, self.gu3p, self.prep2t, self.fin7t, self.down3t, self.dfin7t, self.sgu2, self.sgu2p, self.sprep2, self.sfin2, self.sdown2 }) |k| freeKernel(k);
    }

    /// the v2 heads' resolved output tiles per simdgroup at `rows` rows: gate|up, down
    pub fn sharedTiles(self: *const SharedHeads, k3: *const Kernels3, rows: usize) [2]u32 {
        const ri: usize = @min(rows, 4) - 1;
        const r: u32 = @intCast(rows);
        return .{ sharedOts(k3.geo.inter / 16, 2, self.sots[ri], r), sharedOts(k3.geo.hidden / 16, 1, self.sotsd[ri], r) };
    }

    /// Whether `moeShared` carries this shared expert at `rows`: up to `shared_rows_max` rows, the routed width as its
    /// intermediate, gate|up K slices of exactly one 128-block of hidden (the heads rotate their block in-kernel: dense prep_s's
    /// span), a down split of whole 4-part slices, and the rows' bf16 or f16 output (the SwiGLU table's dtype). Never on a
    /// clamped geometry: the shared heads' table SwiGLU has no clamp (and the routed side launches run fin7's variant).
    pub fn sharedAdmitted(k3: *const Kernels3, rows: usize, sb: SharedBank) bool {
        const g = k3.geo;
        if (g.limit != 0) return false;
        if (rows == 0 or rows > shared_rows_max) return false;
        if (g.topk < 4) return false; // dfin7s and dfin7t write each block from their first 128 threads, as dfin7 does
        if (k3.out_dtype != .bfloat16 and k3.out_dtype != .float16) return false;
        const gs = mlx.getShape(sb.gate_t);
        const us = mlx.getShape(sb.up_t);
        const ds = mlx.getShape(sb.down_t);
        if (gs.len != 3 or us.len != 3 or ds.len != 3) return false;
        const ht: c_int = @intCast(g.hidden / 16);
        const it: c_int = @intCast(g.inter / 16);
        if (gs[0] != ht or gs[1] != it or us[0] != ht or us[1] != it or ds[0] != it or ds[1] != ht) return false;
        if (gs[2] != @as(c_int, @intCast(sb.n_gu)) or us[2] != gs[2] or ds[2] != @as(c_int, @intCast(sb.n_dn))) return false;
        if (sb.ks_gu == 0 or sb.ks_gu * 128 != g.hidden) return false;
        if (sb.ks_dn == 0 or (g.inter / 16) % (sb.ks_dn * 4) != 0) return false;
        return format.Rate.fromN(sb.n_gu) != null and format.Rate.fromN(sb.n_dn) != null;
    }

    /// The four launch configs (form 9's shape: bt1 / bt9, kdiv9) with the shared heads' extra grid rows and constants.
    fn sharedCfgs(self: *SharedHeads, k3: *Kernels3, rows: usize, sb: SharedBank) !SCfg {
        const key: SKey = .{ .n_gu = sb.n_gu, .n_dn = sb.n_dn, .ks_gu = sb.ks_gu, .ks_dn = sb.ks_dn, .def = sb.defer_affine, .sp_gu = sharedSpt(sb.ks_gu, self.sspt), .sp_dn = sharedSpt(sb.ks_dn, self.ssptd) };
        if (self.scfg[rows]) |c| {
            if (std.meta.eql(self.scfg_key[rows], key)) return c;
            freeSCfg(c);
            self.scfg[rows] = null;
        }
        const g = k3.geo;
        const bt: u32 = if (rows == 1) k3.bt1 else k3.bt9;
        const z_gu = k3.kThreadgroups(k3.ks_gu, rows, k3.kdiv9);
        const z_dn = k3.kThreadgroups(k3.ks_dn, rows, k3.kdiv9d orelse k3.kdiv9);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const ys_gu: u32 = (2 * (sb.ks_gu / key.sp_gu) + z_gu - 1) / z_gu;
        const ys_dn: u32 = (sb.ks_dn / key.sp_dn + z_dn - 1) / z_dn;
        const T = ks.TemplateArg;
        const nhw = T{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rkt = T{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = T{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = T{ .name = "KS", .value = .{ .int = @intCast(k3.ks_gu) } };
        const ks_dn_t = T{ .name = "KS", .value = .{ .int = @intCast(k3.ks_dn) } };
        const spt_gu_t = T{ .name = "SPT", .value = .{ .int = @intCast(k3.ks_gu / z_gu) } };
        const spt_dn_t = T{ .name = "SPT", .value = .{ .int = @intCast(k3.ks_dn / z_dn) } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = T{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) k3.group else 1) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = k3.out_dtype } };
        const def_t = T{ .name = "DEFER", .value = .{ .int = if (k3.defer_affine) 1 else 0 } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const nbi_t = T{ .name = "NBI", .value = .{ .int = nbi } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const sz_gu = T{ .name = "SZ", .value = .{ .int = @intCast(z_gu) } };
        const sz_dn = T{ .name = "SZ", .value = .{ .int = @intCast(z_dn) } };
        const ssp = T{ .name = "SSPT", .value = .{ .int = @intCast(key.sp_gu) } };
        const sspd = T{ .name = "SSPTD", .value = .{ .int = @intCast(key.sp_dn) } };
        const bw: u32 = 16 * bt;
        const use_gu8 = rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9);
        const sgu_shapes = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu = if (use_gu8)
            try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &([_][]const c_int{ &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh } } ++ sgu_shapes),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk, sng, sksg, srs, sdef, sz_gu, ssp },
            })
        else
            try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &([_][]const c_int{ &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I } } ++ sgu_shapes),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, sng, sksg, srs, sdef, sz_gu, ssp },
            });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32, .float16, .float32 },
            .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t, out_t, sksg, srs, sdef },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, rk + ys_dn, z_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(k3.ks_dn), rkc, H }, &[_]c_int{ @intCast(sb.ks_dn), R, H } },
            .out_dtypes = &.{ .float32, .float32 },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t, snd, sksd, srs, sdef, sz_dn, sspd },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(down);
        const dfin = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32 * (g.topk + 1), @intCast(rows), 1 },
            .threadgroup = .{ 32 * (g.topk + 1), 1, 1 },
            .out_shapes = &.{&[_]c_int{ R, H }},
            .out_dtypes = &.{k3.out_dtype},
            .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t, sksd, srs, sdef },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(dfin);
        // past gu8_rows: gu3s's grid and template (and fin7s's output dtype) for gu3sf, with fin7s's four outputs, then
        // the routed and shared planes and the shared input sums' block copies; one counter per routed (row-slot,
        // intermediate block), then one per shared block (at least 8: MLX binds an input of fewer than 8 elements in the
        // constant address space, no atomics there)
        const vf: ?VFold = if (use_gu8) null else blk: {
            if (self.gu3sf == null) self.gu3sf = ks.newKernel("turboexl3_gu3sf", &gu3sf_inputs, &gu3sf_outputs, text_gu3sf, header_s, true) orelse return error.KernelCreateFailed;
            if (self.vzero == null) self.vzero = ks.newKernel("turboexl3_vfold_zero", &vfold_zero_inputs, &vfold_zero_outputs, text_vfold_zero, header, true) orelse return error.KernelCreateFailed;
            const nbi_u: u32 = g.inter / 128;
            const ncnt: u32 = @max(rk * nbi_u + nbi_u, 8);
            const nsx: u32 = if (sb.defer_affine) @max(nbi_u * @as(u32, @intCast(rows)) * 2 * (g.hidden / 128), 8) else 8;
            const gcfg = try ks.buildConfig(.{
                .grid = .{ (g.inter / bw) * 128, rk + ys_gu, z_gu },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{@intCast(nsx)} },
                .out_dtypes = &.{ .float16, .float32, .float16, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, sng, sksg, srs, sdef, sz_gu, ssp, out_t },
            });
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gcfg);
            const zcfg = try ks.buildConfig(.{ .grid = .{ ((ncnt + 127) / 128) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{@intCast(ncnt)}}, .out_dtypes = &.{.uint32}, .template = &.{.{ .name = "NCNT", .value = .{ .int = @intCast(ncnt) } }} });
            break :blk .{ .gu = gcfg, .zero = zcfg };
        };
        const c: SCfg = .{ .gu = gu, .fin = fin, .down = down, .dfin = dfin, .gu8 = use_gu8, .vf = vf };
        self.scfg[rows] = c;
        self.scfg_key[rows] = key;
        return c;
    }

    /// Form 9 with the shared expert folded in: y = the routed sum in the output dtype plus sigmoid(logit) * shared(x), in the
    /// bytes of the host's chain (form 3's routed y, the three dense linears, the table SwiGLU, MLX's sigmoid, Multiply, Add)
    /// with no launch of its own: the shared GEMVs ride the gate|up and down launches as extra grid rows, its finishes ride
    /// launches 3 and 5. Past gu8_rows with `fused_fin_fold` launches 2 and 3 are one (`sharedFold`). The caller frees
    /// y; `sharedAdmitted` first.
    pub fn moeShared(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!SharedHeads.sharedAdmitted(k3, rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(k3, rows, sb);
        // routed: zg, zu, xsg, xsu; shared: szg, szu, sxsg, sxsu
        var pl: [8]mlx.mlx_array = undefined;
        if (sc.gu8) {
            try ks.applyConfig(self.gu8s, sc.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &pl, s);
        } else {
            const c = try k3.cfgs9(rows);
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(k3.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            if (self.fused_fin_fold and sc.vf != null) {
                defer for (xp) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                return self.sharedFold(s, sc, xp, x, slots, scores, bank, sb, sb.logit);
            }
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [6]mlx.mlx_array = undefined;
            try ks.applyConfig(self.gu3s, sc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], o[4], o[5] };
        }
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7s, sc.fin, &.{ pl[0], pl[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ hd[0], slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], slots, scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// `moeShared`'s launches past its prepare with gu3s and fin7s as one (gu3sf, after the counters' zeroing launch),
    /// then down3s and dfin7s unchanged; `xp` the routed prepare's xg, xu and block sums (the caller's), `logit` the shared
    /// gate's. The caller frees y.
    fn sharedFold(self: *SharedHeads, s: mlx.mlx_stream, sc: SCfg, xp: [4]mlx.mlx_array, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, sb: SharedBank, logit: mlx.mlx_array) !mlx.mlx_array {
        const vf = sc.vf orelse return error.BadExl3Shape;
        const cnt = try one(self.vzero.?, vf.zero, &.{bank.gate_svh}, s);
        defer _ = mlx.mlx_array_free(cnt);
        var o: [9]mlx.mlx_array = undefined; // hd, hs, sxpd, sxsd; zg, zu, szg, szu, sxsx (scratch)
        try ks.applyConfig(self.gu3sf.?, vf.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh, bank.gate_svh, bank.up_svh, bank.down_suh, xp[2], xp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab, cnt }, &o, s);
        defer for (o) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ o[0], slots, bank.down_t, o[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], slots, scores, bank.down_svh, o[1], zd[1], o[3], sb.down_svh, sb.sigtab, logit }, s);
    }

    /// `routeLaunch`'s config key: the experts, the scores' dtype, the gate's width (0: none).
    pub const RouteKey = struct { ne: u32, tout: mlx.mlx_dtype, gk: u32 };
    /// The merged launch's outputs, each the caller's: prep2's (the gate|up inputs prepared, their per-block sums), the
    /// router's (the slots, the renormalized weights) and the shared gate's logit per row (an empty handle without a gate).
    pub const RoutedPrep = struct { xg: mlx.mlx_array, xu: mlx.mlx_array, xsg: mlx.mlx_array, xsu: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, glogit: mlx.mlx_array };

    /// Simdgroups per merged threadgroup: one per row-slot (phase 2's prepares), one more for the router's softmax and weights,
    /// and with the gate at least nine below that one (simdgroup 0 selects while 1..8 take the gate's dot).
    pub fn routeSimdgroups(topk: u32, gate: bool) u32 {
        return @max(topk, if (gate) @as(u32, 9) else 1) + 1;
    }

    /// The router's top-k and weights and form 3's prepare in one launch (`text_route_prep`) from the router gemv's
    /// logits [rows, E] (the scores in their dtype) and x [rows, hidden]; with `gate_vec` (the shared gate's weight
    /// vector, bf16 [hidden], x bf16) also the shared gate's logit per row, in MLX's gemv order. The bytes of the
    /// router kernel (`text_router_softmax`) and prep2 (`routeChainPrep`), which the GPU test pins.
    pub fn routeLaunch(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, logits: mlx.mlx_array, x: mlx.mlx_array, bank: Kernels.Bank, rows: usize, gate_vec: ?mlx.mlx_array) !RoutedPrep {
        const g = k3.geo;
        if (rows == 0 or rows > Kernels.max_rows or g.topk > 31) return error.Exl3EngineNotImplemented;
        const lsh = mlx.getShape(logits);
        if (lsh.len != 2 or lsh[0] != @as(c_int, @intCast(rows)) or lsh[1] < @as(c_int, @intCast(g.topk)) or lsh[1] > 2048) return error.Exl3EngineNotImplemented;
        const tout = mlx.mlx_array_dtype(logits);
        if (tout != .bfloat16 and tout != .float16 and tout != .float32) return error.Exl3EngineNotImplemented;
        const gate = gate_vec != null;
        if (gate_vec) |gv| {
            if (mlx.mlx_array_dtype(gv) != .bfloat16 or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_size(gv) != g.hidden) return error.Exl3EngineNotImplemented;
        }
        const key: RouteKey = .{ .ne = @intCast(lsh[1]), .tout = tout, .gk = if (gate) g.hidden else 0 };
        if (gate) {
            if (self.route_prepg == null) self.route_prepg = ks.newKernel("turboexl3_route_prepg", &route_prepg_inputs, &route_prepg_outputs, text_route_prepg, "", true) orelse return error.KernelCreateFailed;
        } else if (self.route_prep == null) {
            self.route_prep = ks.newKernel("turboexl3_route_prep", &route_prep_inputs, &route_prep_outputs, text_route_prep, "", true) orelse return error.KernelCreateFailed;
        }
        if (self.route_cfg[rows]) |c| {
            if (!std.meta.eql(self.route_key[rows], key)) {
                _ = mlx.mlx_fast_metal_kernel_config_free(c);
                self.route_cfg[rows] = null;
            }
        }
        if (self.route_cfg[rows] == null) {
            const tg: u32 = 32 * routeSimdgroups(g.topk, gate);
            const rk: c_int = @intCast(rows * g.topk);
            const R: c_int = @intCast(rows);
            const H: c_int = @intCast(g.hidden);
            const nbh: c_int = @intCast(g.hidden / 128);
            const tmpl = [_]ks.TemplateArg{
                .{ .name = "NE", .value = .{ .int = @intCast(key.ne) } },
                .{ .name = "NK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "TG", .value = .{ .int = @intCast(tg) } },
                .{ .name = "TOUT", .value = .{ .dtype = tout } },
                .{ .name = "DIM", .value = .{ .int = H } },
                .{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } },
                .{ .name = "GK", .value = .{ .int = H } },
            };
            const shapes = [_][]const c_int{ &[_]c_int{ rk, H }, &[_]c_int{ rk, H }, &[_]c_int{ rk, nbh }, &[_]c_int{ rk, nbh }, &[_]c_int{rk}, &[_]c_int{rk}, &[_]c_int{R} };
            const dtypes = [_]mlx.mlx_dtype{ .float16, .float16, .float32, .float32, .uint32, tout, .bfloat16 };
            const n: usize = if (gate) 7 else 6;
            self.route_cfg[rows] = try ks.buildConfig(.{
                .grid = .{ (g.hidden / 128) * tg, @intCast(rows), 1 },
                .threadgroup = .{ tg, 1, 1 },
                .out_shapes = shapes[0..n],
                .out_dtypes = dtypes[0..n],
                .template = tmpl[0..n],
            });
            self.route_key[rows] = key;
        }
        var o: [7]mlx.mlx_array = undefined;
        if (gate_vec) |gv| {
            try ks.applyConfig(self.route_prepg.?, self.route_cfg[rows].?, &.{ logits, x, bank.gate_suh, bank.up_suh, x, gv }, o[0..7], s);
        } else {
            try ks.applyConfig(self.route_prep.?, self.route_cfg[rows].?, &.{ logits, x, bank.gate_suh, bank.up_suh }, o[0..6], s);
            o[6] = .{ .ctx = null };
        }
        return .{ .xg = o[0], .xu = o[1], .xsg = o[2], .xsu = o[3], .slots = o[4], .scores = o[5], .glogit = o[6] };
    }

    /// The chain the GPU test pins `routeLaunch` to, prep2's half: form 9's prepare launch (prep2 at `cfgs9`'s config,
    /// as `moeShared` runs it) over the slots the router kernel wrote. Outputs xg, xu, xsg, xsu (the caller's).
    pub fn routeChainPrep(k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Kernels.Bank, rows: usize) ![4]mlx.mlx_array {
        const c = try k3.cfgs9(rows);
        var xp: [4]mlx.mlx_array = undefined;
        try ks.applyConfig(k3.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
        return xp;
    }

    /// `moeShared` with the routing folded into its prepare launch: `routeLaunch` in prep2's place, then `moeShared`'s
    /// gu3s, fin7s, down3s and dfin7s (with `fused_fin_fold`: gu3sf, down3s and dfin7s) unchanged on the launch's slots
    /// and scores and, with `gate_vec`, its gate logit in `sb.logit`'s place. The router kernel's and `moeShared`'s
    /// bytes, one launch and one dependent level fewer. error.Exl3EngineNotImplemented at the rows whose prepare rides the
    /// gate|up launch (gu8s: no prepare launch to fold into); `sharedAdmitted` first. The caller frees y.
    pub fn moeSharedRouted(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, logits: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank, gate_vec: ?mlx.mlx_array) !mlx.mlx_array {
        if (!SharedHeads.sharedAdmitted(k3, rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(k3, rows, sb);
        if (sc.gu8) return error.Exl3EngineNotImplemented;
        const r = try self.routeLaunch(k3, s, logits, x, bank, rows, gate_vec);
        defer {
            _ = mlx.mlx_array_free(r.xg);
            _ = mlx.mlx_array_free(r.xu);
            _ = mlx.mlx_array_free(r.slots);
            _ = mlx.mlx_array_free(r.scores);
            if (r.glogit.ctx != null) _ = mlx.mlx_array_free(r.glogit);
        }
        if (self.fused_fin_fold and sc.vf != null) {
            defer _ = mlx.mlx_array_free(r.xsg);
            defer _ = mlx.mlx_array_free(r.xsu);
            return self.sharedFold(s, sc, .{ r.xg, r.xu, r.xsg, r.xsu }, x, r.slots, r.scores, bank, sb, if (gate_vec != null) r.glogit else sb.logit);
        }
        // moeShared's launches from here on, slots / scores / logit from the merged launch
        var o: [6]mlx.mlx_array = undefined;
        ks.applyConfig(self.gu3s, sc.gu, &.{ r.xg, r.xu, r.slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s) catch |e| {
            _ = mlx.mlx_array_free(r.xsg);
            _ = mlx.mlx_array_free(r.xsu);
            return e;
        };
        const pl = [8]mlx.mlx_array{ o[0], o[1], r.xsg, r.xsu, o[2], o[3], o[4], o[5] };
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7s, sc.fin, &.{ pl[0], pl[1], r.slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3s, sc.down, &.{ hd[0], r.slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7s, sc.dfin, &.{ zd[0], r.slots, r.scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, if (gate_vec != null) r.glogit else sb.logit }, s);
    }

    /// The standalone heads' configs for `moeSharedSide` (form 9's tiles per threadgroup): gate|up with every (K slice,
    /// projection) its own threadgroup along z, fin per (128-block, row), down with every K slice its own threadgroup.
    fn sideCfgs(self: *SharedHeads, k3: *Kernels3, rows: usize, sb: SharedBank) !SSCfg {
        const key: SKey = .{ .n_gu = sb.n_gu, .n_dn = sb.n_dn, .ks_gu = sb.ks_gu, .ks_dn = sb.ks_dn, .def = sb.defer_affine, .sp_gu = sharedSpt(sb.ks_gu, self.sspt), .sp_dn = sharedSpt(sb.ks_dn, self.ssptd) };
        if (self.sscfg[rows]) |c| {
            if (std.meta.eql(self.sscfg_key[rows], key)) return c;
            inline for (.{ c.gu, c.fin, c.down }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            self.sscfg[rows] = null;
        }
        const g = k3.geo;
        const bt: u32 = if (rows == 1) k3.bt1 else k3.bt9;
        const bw: u32 = 16 * bt;
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const T = ks.TemplateArg;
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rk0 = T{ .name = "RKC", .value = .{ .int = 0 } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = k3.out_dtype } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const gu = try ks.buildConfig(.{
            .grid = .{ (g.inter / bw) * 128, 1, 2 * (sb.ks_gu / key.sp_gu) },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float32, .float32, .float32, .float32 },
            .template = &.{ srs, bt_t, rk0, wb, sksg, T{ .name = "SZ", .value = .{ .int = @intCast(2 * (sb.ks_gu / key.sp_gu)) } }, hid, sng, cb, sdef, inter_t, T{ .name = "SSPT", .value = .{ .int = @intCast(key.sp_gu) } } },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32 },
            .template = &.{ inter_t, rk0, nbh_t, sdef, sksg, srs, out_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, 1, sb.ks_dn / key.sp_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(sb.ks_dn), R, H }},
            .out_dtypes = &.{.float32},
            .template = &.{ srs, bt_t, rk0, wb, sksd, inter_t, T{ .name = "SZ", .value = .{ .int = @intCast(sb.ks_dn / key.sp_dn) } }, snd, cb, hid, sdef, T{ .name = "SSPTD", .value = .{ .int = @intCast(key.sp_dn) } } },
        });
        const c: SSCfg = .{ .gu = gu, .fin = fin, .down = down };
        self.sscfg[rows] = c;
        self.sscfg_key[rows] = key;
        return c;
    }

    /// Form 9's launches 1-4 (`moeForm`'s, less the final reduce): the routed down planes and the down inputs' block sums;
    /// the caller frees both.
    fn routedPlanes9(k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !struct { zd: mlx.mlx_array, hs: mlx.mlx_array } {
        const c = try k3.cfgs9(rows);
        var planes: [4]mlx.mlx_array = undefined;
        if (rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9)) {
            try ks.applyConfig(k3.gu8, c.gu8, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &planes, s);
        } else {
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(k3.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            try ks.applyConfig(k3.gu, c.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t }, planes[0..2], s);
            planes[2] = xp[2];
            planes[3] = xp[3];
        }
        defer for (planes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [2]mlx.mlx_array = undefined; // the down's prepared input and its per-block sums
        try ks.applyConfig(k3.fin7, c.fin7, &.{ planes[0], planes[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, planes[2], planes[3] }, &hd, s);
        defer _ = mlx.mlx_array_free(hd[0]);
        errdefer _ = mlx.mlx_array_free(hd[1]);
        const zd = try one(k3.down, c.down, &.{ hd[0], slots, bank.down_t }, s);
        return .{ .zd = zd, .hs = hd[1] };
    }

    /// `moeShared`'s bytes with the shared expert in launches of its own beside form 9's (the routed kernels untouched; the
    /// fused heads enlarge every routed threadgroup's memory): gate|up (rotating in-kernel) beside launch 1, fin beside fin7,
    /// down beside down3, the reduce in dfin7s. The shared chain is as deep as the routed one, so MLX's tape (Kahn order from
    /// the outputs) puts each shared launch between the routed chain's own barriers. `sharedAdmitted` first; the caller frees y.
    pub fn moeSharedSide(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!SharedHeads.sharedAdmitted(k3, rows, sb)) return error.BadExl3Shape;
        const sc = try self.sharedCfgs(k3, rows, sb);
        const sd = try self.sideCfgs(k3, rows, sb);
        const rp = try SharedHeads.routedPlanes9(k3, s, x, slots, bank, rows);
        defer _ = mlx.mlx_array_free(rp.zd);
        defer _ = mlx.mlx_array_free(rp.hs);
        var sp: [4]mlx.mlx_array = undefined; // szg, szu, sxsg, sxsu
        try ks.applyConfig(self.sgu, sd.gu, &.{ x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &sp, s);
        defer for (sp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var sf: [2]mlx.mlx_array = undefined; // sxpd, sxsd
        try ks.applyConfig(self.sfin, sd.fin, &.{ sp[0], sp[1], sp[2], sp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &sf, s);
        defer for (sf) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const szd = try one(self.sdown, sd.down, &.{ sf[0], sb.down_t }, s);
        defer _ = mlx.mlx_array_free(szd);
        return one(self.dfin7s, sc.dfin, &.{ rp.zd, slots, scores, bank.down_svh, rp.hs, szd, sf[1], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// The v2 configs' key at `rows`: the shared bank's shape and the v2 knobs, the tiles per simdgroup resolved for the row
    /// count (`sharedOts`).
    fn sharedKey2(self: *const SharedHeads, k3: *const Kernels3, rows: usize, sb: SharedBank) TKey {
        const ri: usize = @min(rows, 4) - 1;
        const tiles = self.sharedTiles(k3, rows);
        return .{
            .n_gu = sb.n_gu,
            .n_dn = sb.n_dn,
            .ks_gu = sb.ks_gu,
            .ks_dn = sb.ks_dn,
            .def = sb.defer_affine,
            .ots = tiles[0],
            .otsd = tiles[1],
            .prep = self.sprep,
            .lm = self.slm,
            .pre = self.spre[ri],
            .first = self.sfirst,
            .gu8_rows = k3.gu8_rows,
        };
    }

    /// The v2 heads' shared threadgroup counts: gate|up K slices x groups of 4 ots tiles over both projections' inter / 16,
    /// down K slices x groups of 4 otsd of the hidden / 16 tiles.
    pub fn sharedTgs2(k3: *const Kernels3, ks_gu: u32, ks_dn: u32, ots: u32, otsd: u32) struct { gu: u32, dn: u32 } {
        const g = k3.geo;
        return .{ .gu = ks_gu * ((2 * (g.inter / 16)) / (4 * ots)), .dn = ks_dn * ((g.hidden / 16) / (4 * otsd)) };
    }

    /// The v2 inside configs (form 9's routed shape: bt1 / bt9, kdiv9) with the v2 heads' extra grid rows and constants: the
    /// gate|up grid's SX x SZ x SYS slots hold its shared threadgroups (the rows after the routed ones, or before them with
    /// `sfirst`), the same for the down; prep2t where form 9 runs its prepare launch and `sprep`.
    fn sharedCfgs2(self: *SharedHeads, k3: *Kernels3, rows: usize, sb: SharedBank) !TCfg {
        const key = self.sharedKey2(k3, rows, sb);
        if (self.tcfg[rows]) |c| {
            if (std.meta.eql(self.tcfg_key[rows], key)) return c;
            freeTCfg(c);
            self.tcfg[rows] = null;
        }
        const g = k3.geo;
        const bt: u32 = if (rows == 1) k3.bt1 else k3.bt9;
        const bw: u32 = 16 * bt;
        const z_gu = k3.kThreadgroups(k3.ks_gu, rows, k3.kdiv9);
        const z_dn = k3.kThreadgroups(k3.ks_dn, rows, k3.kdiv9d orelse k3.kdiv9);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const nt = SharedHeads.sharedTgs2(k3, key.ks_gu, key.ks_dn, key.ots, key.otsd);
        const x_gu: u32 = g.inter / bw;
        const x_dn: u32 = g.hidden / bw;
        const ys_gu: u32 = (nt.gu + x_gu * z_gu - 1) / (x_gu * z_gu);
        const ys_dn: u32 = (nt.dn + x_dn * z_dn - 1) / (x_dn * z_dn);
        const T = ks.TemplateArg;
        const nhw = T{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rkt = T{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = T{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = T{ .name = "KS", .value = .{ .int = @intCast(k3.ks_gu) } };
        const ks_dn_t = T{ .name = "KS", .value = .{ .int = @intCast(k3.ks_dn) } };
        const spt_gu_t = T{ .name = "SPT", .value = .{ .int = @intCast(k3.ks_gu / z_gu) } };
        const spt_dn_t = T{ .name = "SPT", .value = .{ .int = @intCast(k3.ks_dn / z_dn) } };
        const bt_t = T{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = T{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) k3.group else 1) } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = k3.out_dtype } };
        const def_t = T{ .name = "DEFER", .value = .{ .int = if (k3.defer_affine) 1 else 0 } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const nbi_t = T{ .name = "NBI", .value = .{ .int = nbi } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const slm = T{ .name = "SLM", .value = .{ .int = if (key.lm) 1 else 0 } };
        const spre = T{ .name = "SPRE", .value = .{ .int = @intCast(key.pre) } };
        const snp = T{ .name = "SNP", .value = .{ .int = if ((g.inter / 16) % (4 * key.ots) == 0) 1 else 2 } };
        const h_gu = [_]T{
            sng, sksg, srs, sdef, slm, spre, snp,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.ots) } },
            T{ .name = "SX", .value = .{ .int = @intCast(x_gu) } },
            T{ .name = "SYS", .value = .{ .int = @intCast(ys_gu) } },
            T{ .name = "SYOFF", .value = .{ .int = if (key.first) @as(i32, @intCast(ys_gu)) else 0 } },
        };
        const h_dn = [_]T{
            snd, sksd, srs, sdef, slm, spre,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.otsd) } },
            T{ .name = "SX", .value = .{ .int = @intCast(x_dn) } },
            T{ .name = "SYS", .value = .{ .int = @intCast(ys_dn) } },
            T{ .name = "SYOFF", .value = .{ .int = if (key.first) @as(i32, @intCast(ys_dn)) else 0 } },
        };
        const use_gu8 = rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9);
        const use_prep = !use_gu8 and key.prep;
        const prep: ?mlx.mlx_fast_metal_kernel_config = if (use_prep) try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh }, &[_]c_int{ R, H }, &[_]c_int{ R, H }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float16, .float16, .float32, .float32, .float16, .float16, .float32, .float32 },
            .template = &.{ T{ .name = "DIM", .value = .{ .int = H } }, topk, rkt },
        }) else null;
        errdefer {
            if (prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        }
        const zgu = [_][]const c_int{ &[_]c_int{ @intCast(k3.ks_gu), rkc, I }, &[_]c_int{ @intCast(k3.ks_gu), rkc, I } };
        const xsgu = [_][]const c_int{ &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh } };
        const szgu = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I } };
        const sxsgu = [_][]const c_int{ &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu_grid = [3]u32{ (g.inter / bw) * 128, rk + ys_gu, z_gu };
        const gu = if (use_gu8)
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ xsgu ++ szgu ++ sxsgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk } ++ h_gu),
            })
        else if (use_prep)
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ szgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } ++ h_gu),
            })
        else
            try ks.buildConfig(.{
                .grid = gu_grid,
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &(zgu ++ szgu ++ sxsgu),
                .out_dtypes = &.{ .float32, .float32, .float32, .float32, .float32, .float32 },
                .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } ++ h_gu),
            });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, rk + @as(u32, @intCast(rows)), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, nbi }, &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32, .float16, .float32 },
            .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t, out_t, sksg, srs, sdef, slm },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const down = try ks.buildConfig(.{
            .grid = .{ (g.hidden / bw) * 128, rk + ys_dn, z_dn },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(k3.ks_dn), rkc, H }, &[_]c_int{ @intCast(sb.ks_dn), R, H } },
            .out_dtypes = &.{ .float32, .float32 },
            .template = &([_]T{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t } ++ h_dn),
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(down);
        const dfin = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32 * (g.topk + 1), @intCast(rows), 1 },
            .threadgroup = .{ 32 * (g.topk + 1), 1, 1 },
            .out_shapes = &.{&[_]c_int{ R, H }},
            .out_dtypes = &.{k3.out_dtype},
            .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t, sksd, srs, sdef, slm },
        });
        const c: TCfg = .{ .prep = prep, .gu = gu, .fin = fin, .down = down, .dfin = dfin, .gu8 = use_gu8 };
        self.tcfg[rows] = c;
        self.tcfg_key[rows] = key;
        return c;
    }

    /// The v2 beside configs: the standalone heads (RKC 0: every grid row is the shared expert's) over one-dimensional grids
    /// of their shared threadgroups (SX = the count, SYS = 1, SYOFF = 0); sprep2 then sgu2p where the routed chain runs its
    /// prepare launch (rows above gu8_rows, so both chains are as deep) and `sprep`, else sgu2 rotating in-kernel.
    fn sideCfgs2(self: *SharedHeads, k3: *Kernels3, rows: usize, sb: SharedBank) !TSCfg {
        const key = self.sharedKey2(k3, rows, sb);
        if (self.tscfg[rows]) |c| {
            if (std.meta.eql(self.tscfg_key[rows], key)) return c;
            freeTSCfg(c);
            self.tscfg[rows] = null;
        }
        const g = k3.geo;
        const R: c_int = @intCast(rows);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const nbi: c_int = @intCast(g.inter / 128);
        const nt = SharedHeads.sharedTgs2(k3, key.ks_gu, key.ks_dn, key.ots, key.otsd);
        const T = ks.TemplateArg;
        const cb = T{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = T{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = T{ .name = "HID", .value = .{ .int = H } };
        const inter_t = T{ .name = "INTER", .value = .{ .int = I } };
        const rk0 = T{ .name = "RKC", .value = .{ .int = 0 } };
        const out_t = T{ .name = "OUT_T", .value = .{ .dtype = k3.out_dtype } };
        const nbh_t = T{ .name = "NBH", .value = .{ .int = nbh } };
        const sng = T{ .name = "SNG", .value = .{ .int = @intCast(sb.n_gu) } };
        const snd = T{ .name = "SND", .value = .{ .int = @intCast(sb.n_dn) } };
        const sksg = T{ .name = "SKSG", .value = .{ .int = @intCast(sb.ks_gu) } };
        const sksd = T{ .name = "SKSD", .value = .{ .int = @intCast(sb.ks_dn) } };
        const srs = T{ .name = "SRS", .value = .{ .int = R } };
        const sdef = T{ .name = "SDEF", .value = .{ .int = if (sb.defer_affine) 1 else 0 } };
        const slm = T{ .name = "SLM", .value = .{ .int = if (key.lm) 1 else 0 } };
        const spre = T{ .name = "SPRE", .value = .{ .int = @intCast(key.pre) } };
        const snp = T{ .name = "SNP", .value = .{ .int = if ((g.inter / 16) % (4 * key.ots) == 0) 1 else 2 } };
        const one_d = [_]T{ T{ .name = "SYS", .value = .{ .int = 1 } }, T{ .name = "SYOFF", .value = .{ .int = 0 } } };
        const use_gu8 = rows <= k3.gu8_rows and k3.gu8Admitted(rows, k3.kdiv9);
        const use_prep = !use_gu8 and key.prep;
        const prep: ?mlx.mlx_fast_metal_kernel_config = if (use_prep) try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, H }, &[_]c_int{ R, H }, &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } },
            .out_dtypes = &.{ .float16, .float16, .float32, .float32 },
            .template = &.{ T{ .name = "DIM", .value = .{ .int = H } }, rk0 },
        }) else null;
        errdefer {
            if (prep) |p| _ = mlx.mlx_fast_metal_kernel_config_free(p);
        }
        const h_gu = [_]T{
            wb, hid, inter_t, cb, rk0, sng, sksg, srs, sdef, slm, spre, snp,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.ots) } },
            T{ .name = "SX", .value = .{ .int = @intCast(nt.gu) } },
        } ++ one_d;
        const szgu = [_][]const c_int{ &[_]c_int{ @intCast(sb.ks_gu), R, I }, &[_]c_int{ @intCast(sb.ks_gu), R, I } };
        const sxsgu = [_][]const c_int{ &[_]c_int{ R, nbh }, &[_]c_int{ R, nbh } };
        const gu = if (use_prep)
            try ks.buildConfig(.{ .grid = .{ nt.gu * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &szgu, .out_dtypes = &.{ .float32, .float32 }, .template = &h_gu })
        else
            try ks.buildConfig(.{ .grid = .{ nt.gu * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &(szgu ++ sxsgu), .out_dtypes = &.{ .float32, .float32, .float32, .float32 }, .template = &h_gu });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const fin = try ks.buildConfig(.{
            .grid = .{ (g.inter / 128) * 32, @intCast(rows), 1 },
            .threadgroup = .{ 32, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ R, I }, &[_]c_int{ R, nbi } },
            .out_dtypes = &.{ .float16, .float32 },
            .template = &.{ inter_t, rk0, nbh_t, sdef, sksg, srs, out_t, slm },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(fin);
        const h_dn = [_]T{
            wb, hid, inter_t, cb, rk0, snd, sksd, srs, sdef, slm, spre,
            T{ .name = "SOTS", .value = .{ .int = @intCast(key.otsd) } },
            T{ .name = "SX", .value = .{ .int = @intCast(nt.dn) } },
        } ++ one_d;
        const down = try ks.buildConfig(.{
            .grid = .{ nt.dn * 128, 1, 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(sb.ks_dn), R, H }},
            .out_dtypes = &.{.float32},
            .template = &h_dn,
        });
        const c: TSCfg = .{ .prep = prep, .gu = gu, .fin = fin, .down = down };
        self.tscfg[rows] = c;
        self.tscfg_key[rows] = key;
        return c;
    }

    /// `moeShared`'s bytes with the v2 heads (one simdgroup per (K slice, sots output tiles) writing dgemv's plane values with
    /// no threadgroup reduction; the gate|up input prepared once in prep2t where form 9 runs a prepare launch and `sprep`, else
    /// rotated per threadgroup; planes in layout `slm`). `sharedAdmitted` first; the caller frees y.
    pub fn moeShared2(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!SharedHeads.sharedAdmitted(k3, rows, sb)) return error.BadExl3Shape;
        const tc = try self.sharedCfgs2(k3, rows, sb);
        // routed: zg, zu, xsg, xsu; shared: szg, szu, sxsg, sxsu
        var pl: [8]mlx.mlx_array = undefined;
        if (tc.gu8) {
            try ks.applyConfig(self.gu8t, tc.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &pl, s);
        } else if (tc.prep) |pc| {
            var xp: [8]mlx.mlx_array = undefined; // xg, xu and their block sums; the shared sxg, sxu and theirs
            try ks.applyConfig(self.prep2t, pc, &.{ x, slots, bank.gate_suh, bank.up_suh, sb.gate_suh, sb.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            defer _ = mlx.mlx_array_free(xp[4]);
            defer _ = mlx.mlx_array_free(xp[5]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            errdefer _ = mlx.mlx_array_free(xp[6]);
            errdefer _ = mlx.mlx_array_free(xp[7]);
            var o: [4]mlx.mlx_array = undefined; // zg, zu, szg, szu
            try ks.applyConfig(self.gu3p, tc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, xp[4], xp[5], sb.gate_t, sb.up_t }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], xp[6], xp[7] };
        } else {
            const c = try k3.cfgs9(rows);
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(k3.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [6]mlx.mlx_array = undefined; // zg, zu, szg, szu, sxsg, sxsu
            try ks.applyConfig(self.gu3t, tc.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t, x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &o, s);
            pl = .{ o[0], o[1], xp[2], xp[3], o[2], o[3], o[4], o[5] };
        }
        defer for (pl) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [4]mlx.mlx_array = undefined; // hd, hs; the shared down's prepared input sxpd and its block sums sxsd
        try ks.applyConfig(self.fin7t, tc.fin, &.{ pl[0], pl[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var zd: [2]mlx.mlx_array = undefined; // the routed down planes, the shared down planes
        try ks.applyConfig(self.down3t, tc.down, &.{ hd[0], slots, bank.down_t, hd[2], sb.down_t }, &zd, s);
        defer for (zd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        return one(self.dfin7t, tc.dfin, &.{ zd[0], slots, scores, bank.down_svh, hd[1], zd[1], hd[3], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

    /// `moeShared2`'s bytes with the v2 heads in launches of their own beside form 9's untouched routed launches (sprep2 then
    /// sgu2p, or sgu2; sfin2; sdown2), the reduce in dfin7t. `sharedAdmitted` first; the caller frees y.
    pub fn moeSharedSide2(self: *SharedHeads, k3: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, sb: SharedBank) !mlx.mlx_array {
        if (!SharedHeads.sharedAdmitted(k3, rows, sb)) return error.BadExl3Shape;
        const tc = try self.sharedCfgs2(k3, rows, sb);
        const sd = try self.sideCfgs2(k3, rows, sb);
        const rp = try SharedHeads.routedPlanes9(k3, s, x, slots, bank, rows);
        defer _ = mlx.mlx_array_free(rp.zd);
        defer _ = mlx.mlx_array_free(rp.hs);
        var sp: [4]mlx.mlx_array = undefined; // szg, szu, sxsg, sxsu
        if (sd.prep) |pc| {
            var xp: [4]mlx.mlx_array = undefined; // sxg, sxu, sxsg, sxsu
            try ks.applyConfig(self.sprep2, pc, &.{ x, sb.gate_suh, sb.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            var o: [2]mlx.mlx_array = undefined; // szg, szu
            try ks.applyConfig(self.sgu2p, sd.gu, &.{ xp[0], xp[1], sb.gate_t, sb.up_t }, &o, s);
            sp = .{ o[0], o[1], xp[2], xp[3] };
        } else {
            try ks.applyConfig(self.sgu2, sd.gu, &.{ x, sb.gate_t, sb.up_t, sb.gate_suh, sb.up_suh }, &sp, s);
        }
        defer for (sp) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var sf: [2]mlx.mlx_array = undefined; // sxpd, sxsd
        try ks.applyConfig(self.sfin2, sd.fin, &.{ sp[0], sp[1], sp[2], sp[3], sb.gate_svh, sb.up_svh, sb.down_suh, sb.sigtab }, &sf, s);
        defer for (sf) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const szd = try one(self.sdown2, sd.down, &.{ sf[0], sb.down_t }, s);
        defer _ = mlx.mlx_array_free(szd);
        return one(self.dfin7t, tc.dfin, &.{ rp.zd, slots, scores, bank.down_svh, rp.hs, szd, sf[1], sb.down_svh, sb.sigtab, sb.logit }, s);
    }

};

/// A `Kernels3` with its shared heads, for tests that make their own
pub const Owned = struct {
    k: Kernels3,
    h: SharedHeads,

    pub fn deinit(self: *Owned) void {
        self.h.deinit();
        self.k.deinit();
    }
};

// ── the process's shared heads, one per `Kernels3` (root's cache of geometries) ──

const Entry = struct { key: *const Kernels3, h: SharedHeads };
const cache_lock = Global(std.atomic.Value(u32), "qwen38_shared_lock", std.atomic.Value(u32).init(0));
const cache = Global([16]?Entry, "qwen38_shared_cache", @splat(null));

/// The shared heads beside `k3` (root's `Kernels3` of the call's geometry), made at its first shared call:
/// `Config.fused_fin_fold` and `shared_spt` as configured then
pub fn forKernels(k3: *const Kernels3) !*SharedHeads {
    while (cache_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.get().*.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache.get().*, 0..) |*slot, i| {
        if (slot.*) |*e| {
            if (e.key == k3) return &e.h;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    var h = try SharedHeads.init();
    h.fused_fin_fold = config.get().fused_fin_fold;
    // the shared expert's gate|up K slices per GEMV threadgroup (of its 20; a value that does not divide them runs 1).
    // Launch shape only: the planes, so the bytes, stay.
    h.sspt = config.get().shared_spt;
    cache.get().*[i] = .{ .key = k3, .h = h };
    return &cache.get().*[i].?.h;
}

/// The process-wide state here (root's `shared_state` lists it)
pub const shared_state = .{ cache_lock, cache };

/// The lines of `text` (no empty ones), for the fold test's line-for-line checks
fn nonEmptyLines(a: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > 0) try out.append(a, l);
    return out.toOwnedSlice(a);
}

test "turboexl3 decode fused fin fold texts: gu3sf is gu3s but its plane and sums stores, then fin7's and the shared fin head's statements" {
    const t = std.testing;
    const a = t.allocator;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "SNG", "SKSG", "SRS", "SDEF", "SZ", "SSPT", "OUT_T" }) |name| try t.expect(std.mem.indexOf(u8, text_gu3sf, name) != null);
    for (gu3sf_inputs ++ gu3sf_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3sf, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_vfold_zero, "NCNT") != null);
    // gu3s and fin7s are their pieces; every piece gu3sf keeps is in it line for line, every piece it replaces is gone and
    // its replacement there once
    try t.expect(std.mem.eql(u8, text_gu3s, sgh_1 ++ sgh_2 ++ sgh_3 ++ sgh_z ++ sgh_4 ++ sgh_sums ++ sgh_5 ++ sgh_end ++ text_gu3));
    try t.expect(std.mem.eql(u8, text_fin7s, sfh_1 ++ sfh_2 ++ sfh_xg ++ sfh_xu ++ sfh_3 ++ sfh_g ++ sfh_u ++ sfh_4 ++ sfh_end ++ text_fin7));
    const kept = [_][]const u8{ sgh_1, sgh_2, sgh_3, sgh_4, sgh_5, sgh_end, sfh_2, sfh_3, sfh_4, gu3_1, rk_line, gu3_2, gu3_3, gu3_4, fin7_2, fin7_3, fin7_sums, fin7_4 };
    for (kept) |p| {
        const ls = try nonEmptyLines(a, p);
        defer a.free(ls);
        for (ls) |l| try t.expect(std.mem.indexOf(u8, text_gu3sf, l) != null);
    }
    for ([_][]const u8{ sgh_z, sgh_sums, sfh_xg, sfh_xu, sfh_g, sfh_u, gu3_stores, fin7_planes }) |p| try t.expect(std.mem.indexOf(u8, text_gu3sf, p) == null);
    for ([_][]const u8{ vf_head_zc, vf_head_sumsc, vf_sfin_xgc, vf_sfin_xuc, vf_sfin_g, vf_sfin_u, vf_gu3_stores, vf_fin7_planes }) |p| try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_gu3sf, p));
    // the inline finishes: turboexl3_sfinish's 12 statements each, its parameters bound by name, in a block of their own
    for ([_][]const u8{ vf_sfin_g, vf_sfin_u }) |p| {
        const ls = try nonEmptyLines(a, p);
        defer a.free(ls);
        try t.expectEqual(@as(usize, 14), ls.len);
    }
}

test "turboexl3 decode shared-expert texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "SRS", "BT", "RKC", "WBITS", "HID", "SKSG", "SZ", "INTER", "SNG", "CODEBOOK", "SDEF", "SSPT" }) |name| {
        try t.expect(std.mem.indexOf(u8, shared_gu_head, name) != null);
    }
    for ([_][]const u8{ "RKC", "INTER", "NBH", "SDEF", "SKSG", "SRS", "OUT_T" }) |name| try t.expect(std.mem.indexOf(u8, shared_fin_head, name) != null);
    for ([_][]const u8{ "SRS", "BT", "RKC", "WBITS", "SKSD", "INTER", "SZ", "SND", "CODEBOOK", "HID", "SDEF", "SSPTD" }) |name| try t.expect(std.mem.indexOf(u8, shared_down_head, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "KS", "DEFER", "NBI", "RKC", "OUT_T", "SKSD", "SRS", "SDEF" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7s, name) != null);
    for (gu3s_inputs ++ gu3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3s, std.mem.span(n)) != null);
    for (gu8s_inputs ++ gu8s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8s, std.mem.span(n)) != null);
    for (fin7s_inputs ++ fin7s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7s, std.mem.span(n)) != null);
    for (down3s_inputs ++ down3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_down3s, std.mem.span(n)) != null);
    for (dfin7s_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7s, std.mem.span(n)) != null);
}

test "turboexl3 decode v2 shared-head texts: every template constant read is passed, every signature name is used" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/decode_shared2.metal"));
    const t = std.testing;
    const gu_names = [_][]const u8{ "SRS", "SYOFF", "RKC", "SYS", "SX", "WBITS", "SNG", "HID", "SKSG", "INTER", "SOTS", "SPRE", "CODEBOOK", "SDEF", "SLM" };
    for (gu_names ++ [_][]const u8{"SNP"}) |name| try t.expect(std.mem.indexOf(u8, shared_gu2_head, name) != null);
    for (gu_names) |name| try t.expect(std.mem.indexOf(u8, shared_gu2p_head, name) != null);
    for ([_][]const u8{ "RKC", "DIM" }) |name| try t.expect(std.mem.indexOf(u8, shared_prep_head, name) != null);
    for ([_][]const u8{ "RKC", "INTER", "NBH", "SDEF", "SKSG", "SRS", "OUT_T", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, shared_fin2_head, name) != null);
    for ([_][]const u8{ "SRS", "SYOFF", "RKC", "SYS", "SX", "WBITS", "SND", "HID", "SKSD", "INTER", "SOTS", "SPRE", "CODEBOOK", "SDEF", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, shared_down2_head, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "KS", "DEFER", "NBI", "RKC", "OUT_T", "SKSD", "SRS", "SDEF", "SLM" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7t, name) != null);
    // the routed bodies under the v2 heads take rk = y - SYOFF, their one rk line spliced
    for ([_][]const u8{ text_gu8_y, text_gu3_y, text_down3_y }) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, pin[0]) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[1]) == null);
    }
    for (gu8s_inputs ++ gu8s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8t, std.mem.span(n)) != null);
    for (gu3s_inputs ++ gu3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3t, std.mem.span(n)) != null);
    for (gu3p_inputs ++ gu3p_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu3p, std.mem.span(n)) != null);
    for (prep2t_inputs ++ prep2t_outputs) |n| try t.expect(std.mem.indexOf(u8, text_prep2t, std.mem.span(n)) != null);
    for (fin7s_inputs ++ fin7s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7t, std.mem.span(n)) != null);
    for (down3s_inputs ++ down3s_outputs) |n| try t.expect(std.mem.indexOf(u8, text_down3t, std.mem.span(n)) != null);
    for (dfin7s_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7t, std.mem.span(n)) != null);
    for (shared_gu_inputs ++ shared_gu_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_gu2_head, std.mem.span(n)) != null);
    for (shared_gu2p_inputs ++ shared_gu2p_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_gu2p_head, std.mem.span(n)) != null);
    for (shared_prep_inputs ++ shared_prep_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_prep_head, std.mem.span(n)) != null);
    for (sfin_inputs ++ sfin_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_fin2_head, std.mem.span(n)) != null);
    for (sdown_inputs ++ sdown_outputs) |n| try t.expect(std.mem.indexOf(u8, shared_down2_head, std.mem.span(n)) != null);
    // the parts take every codeword from the window's 32-bit words (no 64-bit window anywhere in the v2 helpers)
    try t.expect(std.mem.indexOf(u8, metal_shared2, pin[2]) != null);
    try t.expect(std.mem.indexOf(u8, metal_shared2, pin[3]) == null);
    try t.expect(std.mem.indexOf(u8, metal_shared2, "Exl3BitsM<exl3_rmode<N>()> ww;") == null);
    // each launch's text (the v2 kernels, the routed down body under the rk splice, the standalone heads) uses every name of
    // its signature
    const Sig = struct { text: []const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8 };
    const sigs = [_]Sig{
        .{ .text = text_down3t, .inputs = &down3s_inputs, .outputs = &down3s_outputs },
        .{ .text = shared_down2_head, .inputs = &sdown_inputs, .outputs = &sdown_outputs },
        .{ .text = text_down3_y, .inputs = &down3_inputs, .outputs = &down3_outputs },
        .{ .text = text_down3s, .inputs = &down3s_inputs, .outputs = &down3s_outputs },
        .{ .text = shared_down_head, .inputs = &sdown_inputs, .outputs = &sdown_outputs },
        .{ .text = text_gu3p, .inputs = &gu3p_inputs, .outputs = &gu3p_outputs },
        .{ .text = text_gu3t, .inputs = &gu3s_inputs, .outputs = &gu3s_outputs },
        .{ .text = text_gu8t, .inputs = &gu8s_inputs, .outputs = &gu8s_outputs },
        .{ .text = shared_gu2_head, .inputs = &shared_gu_inputs, .outputs = &shared_gu_outputs },
        .{ .text = shared_gu2p_head, .inputs = &shared_gu2p_inputs, .outputs = &shared_gu2p_outputs },
        .{ .text = text_dfin7t, .inputs = &dfin7s_inputs, .outputs = &down_outputs },
    };
    for (sigs) |p| {
        for (p.inputs) |n| try t.expect(std.mem.indexOf(u8, p.text, std.mem.span(n)) != null);
        for (p.outputs) |n| try t.expect(std.mem.indexOf(u8, p.text, std.mem.span(n)) != null);
    }
    // tiles per simdgroup: whole groups of four simdgroups within a projection, else 1; within 64 accumulators per lane
    try t.expectEqual(@as(u32, 2), sharedOts(40, 2, 2, 1));
    try t.expectEqual(@as(u32, 4), sharedOts(40, 2, 4, 1));
    try t.expectEqual(@as(u32, 1), sharedOts(40, 2, 8, 1));
    try t.expectEqual(@as(u32, 8), sharedOts(160, 1, 8, 1));
    try t.expectEqual(@as(u32, 1), sharedOts(160, 1, 3, 1));
    try t.expectEqual(@as(u32, 10), sharedOts(40, 2, 10, 1));
    try t.expectEqual(@as(u32, 5), sharedOts(40, 2, 10, 2));
    try t.expectEqual(@as(u32, 5), sharedOts(40, 2, 5, 3));
    try t.expectEqual(@as(u32, 4), sharedOts(40, 2, 10, 4));
    try t.expectEqual(@as(u32, 4), sharedOts(160, 1, 8, 4));
    try t.expectEqual(@as(u32, 2), sharedOts(160, 1, 8, 8));
}

test "turboexl3 decode router-folded prepare texts: the merged launch carries the router's and prep2's statements verbatim, every name is used" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/decode_route_prep.metal"));
    const t = std.testing;
    const texts = [_][]const u8{ text_route_prep, text_route_prepg };
    // the pieces are pieces of the router copy and of prep2, and each lands once in each merged text
    for ([_][]const u8{ router_sel, router_softmax, router_weights }) |piece| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax, piece) != null);
        for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, piece));
    }
    try t.expect(std.mem.indexOf(u8, text_prep2, prep2_body) != null);
    for (texts) |txt| try t.expectEqual(@as(usize, 1), std.mem.count(u8, txt, prep2_body));
    // the value-setting lines of the router's rounds, softmax and weights, and of prep2's rotation
    const lines = [_][]const u8{
        pin[0],
        pin[1],
        pin[2],
        pin[3],
        pin[4],
        pin[5],
        pin[6],
        pin[7],
        pin[8],
        pin[9],
        pin[10],
        pin[11],
        pin[12],
        pin[13],
        pin[14],
    };
    for (lines) |line| {
        try t.expect(std.mem.indexOf(u8, text_router_softmax ++ text_prep2, line) != null);
        for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, line) != null);
    }
    // the hand-offs: smax / snorm through threadgroup memory, the expert from the selection, the row-slot of the simdgroup
    for (texts) |txt| {
        try t.expect(std.mem.indexOf(u8, txt, pin[15]) != null);
        try t.expect(std.mem.indexOf(u8, txt, @embedFile("kernels/test/pins/decode_route_prep_1.metal")) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[16]) != null);
        try t.expect(std.mem.indexOf(u8, txt, pin[17]) != null);
        try t.expectEqual(@as(usize, 2), std.mem.count(u8, txt, pin[18]));
    }
    // the shared gate's dot and sum only in the gate kernel
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_gemv));
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, text_route_prepg, text_gate_sum));
    try t.expect(std.mem.indexOf(u8, text_route_prep, pin[19]) == null);
    // every input / output name and template constant is read or written
    for (route_prep_inputs ++ route_prep_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prep, std.mem.span(n)) != null);
    for (route_prepg_inputs ++ route_prepg_outputs) |n| try t.expect(std.mem.indexOf(u8, text_route_prepg, std.mem.span(n)) != null);
    for ([_][]const u8{ "NE", "NK", "TG", "TOUT", "DIM", "TOPK" }) |name| for (texts) |txt| try t.expect(std.mem.indexOf(u8, txt, name) != null);
    try t.expect(std.mem.indexOf(u8, text_route_prepg, "GK") != null);
    for (router_inputs ++ router_outputs) |n| try t.expect(std.mem.indexOf(u8, text_router_softmax, std.mem.span(n)) != null);
    for ([_][]const u8{ "gx", "gw", "GK", pin[20], pin[21] }) |name| try t.expect(std.mem.indexOf(u8, text_gate_gemv, name) != null);
    // one simdgroup per slot plus the weights', with the gate's eight beside the selection
    try t.expectEqual(@as(u32, 11), SharedHeads.routeSimdgroups(10, false));
    try t.expectEqual(@as(u32, 11), SharedHeads.routeSimdgroups(10, true));
    try t.expectEqual(@as(u32, 5), SharedHeads.routeSimdgroups(4, false));
    try t.expectEqual(@as(u32, 10), SharedHeads.routeSimdgroups(4, true));
}
