//! The GatedDeltaNet layer's tail at prefill rows on EXL3 linears, in one launch: z's finish, the gated norm and
//! out_proj's prepare per (row, value head). The host stops in_proj_z at its inner product (`linearInnerSplit`), runs
//! `prepare` on it and the recurrence's output, and continues out_proj from the prepared rows (`linearPrepared`). Neither
//! z nor the gated rows are written. Qwen3.8-Flash-Next's GatedDeltaNet layers are the client.
//!
//! Every element is the chain's bytes:
//! - z's finish is `linearFinish` (dense `text_finish`'s statements): f16 read, H128, x svh, f16 round, the activation
//!   dtype's round.
//! - The gated norm is qwen4_exp's (`output_gate_type` sigmoid). y's RMS over each 128-wide head is summed as MLX's
//!   rms_norm sums it at axis 128 (one simdgroup, lane L over channels 4L..4L+3 in order, then simd_sum), with
//!   precise::rsqrt(sum / 128 + eps) and w * T(y * inv). The product with sigmoid(z) is in T, the sigmoid by the host's
//!   table of MLX's sigmoid (as ATTN-OUT takes it), so it is MLX's Sigmoid and Multiply.
//! - out_proj's prepare is `linearPrep` (dense `text_prep`'s statements, natural order): f16(gated) * suh, H128, x
//!   HAD_SCALE, f16.
//! The two butterflies are `sources.had128`. The swish gate and the formula sigmoid are not served.
const std = @import("std");
const Global = @import("../../core/global.zig").Global;
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const decode = @import("../../core/decode.zig");
const sources = @import("../../core/sources.zig");

const had128 = sources.had128;

/// The tail's one launch: xp half [rows, HV DV] (out_proj's prepared rows, natural order) from y T [rows, HV, DV] (the
/// recurrence's output), zq half [rows, HV DV] (in_proj_z's inner product) with z's finish vector svh half [HV DV], the
/// norm weight norm_w T [DV], eps f32 (0-dim), the sigmoid table sigtab T [65536] and out_proj's prepare vector suh half
/// [HV DV]. DV is 128: one simdgroup per (row, value head), NSG heads per threadgroup, grid (32 NSG, rows, HV / NSG).
pub const text_prep: [:0]const u8 = prep_1 ++ had128 ++ prep_2 ++ had128 ++ prep_3;

/// z's finish up to its butterfly: the inner product's f16 read
const prep_1 = @embedFile("kernels/gdn_out/prep_1.metal");
/// z's finish store, the gated norm, and the prepare's input up to its butterfly
const prep_2 = @embedFile("kernels/gdn_out/prep_2.metal");
/// the prepare's store, natural order
const prep_3 = @embedFile("kernels/gdn_out/prep_3.metal");

const prep_in = [_][*:0]const u8{ "y", "zq", "svh", "norm_w", "eps", "sigtab", "suh" };
const prep_out = [_][*:0]const u8{"xp"};

/// The value-head width the tail serves: one simdgroup per head, lane L holding channels L + 32 j.
pub const head_dim = 128;

/// The gated norm's operands, in y's dtype unless noted.
pub const Norm = struct {
    /// the norm weight [128]
    w: mlx.mlx_array,
    /// 0-dim f32: the norm's eps
    eps: mlx.mlx_array,
    /// [65536]: MLX's sigmoid of every 16-bit pattern (the host's table)
    sigtab: mlx.mlx_array,
};

/// Heads per threadgroup. This sets the launch shape only: every element's arithmetic is the same at any NSG.
pub fn headsPerGroup(hv: u32) u32 {
    return if (hv % 4 == 0) 4 else if (hv % 2 == 0) 2 else 1;
}

const kernel_lock = Global(std.atomic.Value(u32), "gdn_out_kernel_lock", std.atomic.Value(u32).init(0));
const kernel = Global(?mlx.mlx_fast_metal_kernel, "gdn_out_kernel", null);

/// The tail's kernel, made at its first call (a process that never takes it never builds it).
fn kernelGet() !mlx.mlx_fast_metal_kernel {
    while (kernel_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer kernel_lock.get().*.store(0, .release);
    if (kernel.get().*) |k| return k;
    const k = ks.newKernel("turboexl3_gdn_out_prep", &prep_in, &prep_out, text_prep, decode.header, true) orelse return error.KernelCreateFailed;
    kernel.get().* = k;
    return k;
}

/// The tail's launch (`text_prep`): xp half [rows, hv 128] from y [rows, hv, 128] (any leading shape whose product is
/// rows), zq half [rows, hv 128], z's `svh`, the norm and out_proj's `suh`. `root.gdnOutPrep` checks the operands.
pub fn prepare(s: mlx.mlx_stream, y: mlx.mlx_array, zq: mlx.mlx_array, svh: mlx.mlx_array, norm: Norm, suh: mlx.mlx_array, hv: u32, rows: usize) !mlx.mlx_array {
    const k = try kernelGet();
    const nsg = headsPerGroup(hv);
    const cfg = try ks.buildConfig(.{
        .grid = .{ 32 * nsg, @intCast(rows), hv / nsg },
        .threadgroup = .{ 32 * nsg, 1, 1 },
        .out_shapes = &.{&[_]c_int{ @intCast(rows), @intCast(hv * head_dim) }},
        .out_dtypes = &.{.float16},
        .template = &.{
            .{ .name = "T", .value = .{ .dtype = mlx.mlx_array_dtype(y) } },
            .{ .name = "HV", .value = .{ .int = @intCast(hv) } },
            .{ .name = "DV", .value = .{ .int = head_dim } },
            .{ .name = "NSG", .value = .{ .int = @intCast(nsg) } },
        },
    });
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    var xp: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(k, cfg, &.{ y, zq, svh, norm.w, norm.eps, norm.sigtab, suh }, &xp, s);
    return xp[0];
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernel_lock, kernel };

// ── tests ──

const testing = std.testing;
const dense = @import("../../core/dense.zig");
const attn_io = @import("attn_io.zig");

fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

test "turboexl3 gdn_out text: dense's butterfly twice, z's finish and out_proj's prepare as dense writes them, the sigmoid by the table" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("kernels/test/pins/gdn_out.metal"));
    // the butterfly spliced is dense's (its prepare's and its finish's statements, verbatim): once for z's finish, once
    // for out_proj's prepare
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep, had128));
    try testing.expectEqual(@as(usize, 1), count(dense.text_finish, had128));
    try testing.expectEqual(@as(usize, 2), count(text_prep, had128));
    // the scale both apply, z's finish value (dense's finish, before its OUT_T round) and the prepare's store value
    try testing.expectEqual(@as(usize, 1), count(dense.text_finish, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(text_prep, pin[0]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_finish, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(text_prep, pin[1]));
    try testing.expectEqual(@as(usize, 1), count(dense.text_prep, pin[2]));
    try testing.expectEqual(@as(usize, 1), count(text_prep, pin[2]));
    // the sigmoid by the host's table, as ATTN-OUT's gated prepare reads it
    try testing.expectEqual(@as(usize, 1), count(attn_io.text_prep_gate, pin[3]));
    try testing.expectEqual(@as(usize, 1), count(text_prep, pin[3]));
    try testing.expectEqual(@as(u32, 4), headsPerGroup(48));
    try testing.expectEqual(@as(u32, 2), headsPerGroup(6));
    try testing.expectEqual(@as(u32, 1), headsPerGroup(5));
}
