//! The routed decode kernels (rows <= DECODE_ROWS_MAX). `moe` (root.zig) runs the form `Config.decode_form` names, each
//! with form 3's bytes (config.zig describes them): form 3's split-K chain and its variants 7, 8 and 9 (the default) in
//! `Kernels3`, forms 5 and 6 in `Kernels5`. Form 1 (`Kernels`), described here, is the plainest and is not served: two
//! launches per MoE layer, any admitted rate, codebook, window and 128-aligned geometry, every constant a template int of
//! one Metal text per kernel; the GPU tests hold it to the host reference.
//!
//! Launch 1, `turboexl3_gu`, one 128-thread threadgroup per (128-wide intermediate block, slot, row): the four simdgroups
//! rotate the row's input for the slot's expert (f16 round, suh, H128 per block as a 5-shuffle + 2-register butterfly,
//! 1/sqrt(128), f16 round: `format.prepareInput`) into threadgroup memory once, then split the input tiles four ways and
//! stream the gate and up trellis tiles of the block's 8 output tiles through the lane plan (`lane.zig`), decode in
//! registers and accumulate f32; the partial sums meet in threadgroup memory, simdgroup 0 rounds to f16 (`innerGemv`),
//! rotates the block's 128 outputs (H128), scales by svh and rounds (`finishOutput`), and writes h = silu(gate) * up as f32.
//!
//! Launch 2, `turboexl3_down`, one threadgroup per (128-wide hidden block, row): for each of the row's slots in order it
//! rotates that slot's h (f16 round, down suh, H128, f16 round), streams the down tiles of its 8 output tiles, reduces,
//! rounds, rotates, scales by svh, rounds, and accumulates score_k * y_k in registers; one f32 write per output at the end.
//! No atomics, no cross-threadgroup traffic: the sum over slots is in slot order, so the bytes are deterministic.
//!
//! Rounding points are Sushi's host reference's (`ref.zig`); the reductions run in a different order, so parity is a
//! tolerance, not bit identity (the GPU test states it).
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const format = @import("format.zig");
const lane = @import("lane.zig");
const sources = @import("sources.zig");

pub const Geometry = struct {
    hidden: u32,
    inter: u32,
    topk: u32,
    rate: format.Rate,
    dec: format.Decode,
    /// timing probes only (never a served kernel): 1 skips the codebook hash, 2 skips the funnel too
    probe: u8 = 0,
    /// the SwiGLU's clamp (`root.moeClamped`: DeepSeek-V4's and GLM's `swiglu_limit`; 0 none): each row-slot's gate clamped
    /// from above at `limit` and its up to [-limit, limit] before the SiLU. `Kernels3` carries it in fin7's variant
    /// (`text_fin7c`, forms 7, 8 and 9); `Kernels` and `Kernels5` refuse a clamped geometry.
    limit: u32 = 0,

    pub fn admitted(g: Geometry) bool {
        return g.hidden % 128 == 0 and g.inter % 128 == 0 and g.hidden > 0 and g.inter > 0 and g.topk > 0 and g.topk <= 32;
    }
};

/// The Metal every decode kernel shares after the helpers: `turboexl3_row_of`, the tile row of each of a lane's eight
/// weights (quantize.py's tensor-core layout).
const metal_common = @embedFile("kernels/decode/common.metal");

/// Launch 1's text.
pub const text_gu = @embedFile("kernels/decode/gu.metal");

/// Launch 2's text.
pub const text_down = @embedFile("kernels/decode/down.metal");

/// The GEMV core every form-3 kernel inlines: lane L of a simdgroup accumulates its 8 tiles x 2 columns over input
/// tiles [tk0, tk1) of expert `wbase` (the stacked bank's expert base, in uint words), x in lane order.
const metal_gemv_core = @embedFile("kernels/decode/gemv_core.metal");

pub const header: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core;

const gu_inputs = [_][*:0]const u8{ "x", "slots", "gate_t", "gate_suh", "gate_svh", "up_t", "up_suh", "up_svh" };
const gu_outputs = [_][*:0]const u8{"h"};
const down_inputs = [_][*:0]const u8{ "h", "slots", "scores", "down_t", "down_suh", "down_svh" };
pub const down_outputs = [_][*:0]const u8{"y"};

/// Drops a kernel `ks.newKernel` made, by the consumer's kernel API (README.md's Wiring). A `kernel_launch` that declares
/// `mlx_fast_metal_kernel_free` hands each kernel to its caller and frees it there (mlx-stream's: the kernel's launch-cache
/// entry, then mlx-c's free); one that does not keeps every kernel itself (the plugin's: the server's kernel cache, one
/// kernel per name and text for the process), so nothing is freed. Every kernel turboexl3 drops goes through here.
pub fn freeKernel(k: mlx.mlx_fast_metal_kernel) void {
    if (@hasDecl(ks, "mlx_fast_metal_kernel_free")) _ = ks.mlx_fast_metal_kernel_free(k);
}

/// The two kernels of one geometry (compiled once per (rate, codebook, window, hidden, inter, topk)); configs per row count.
pub const Kernels = struct {
    geo: Geometry,
    gu: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    cfg_gu: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),
    cfg_down: [max_rows + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null),

    pub const max_rows: usize = 16;

    pub fn init(geo: Geometry) !Kernels {
        if (!geo.admitted()) return error.BadExl3Shape;
        if (geo.limit != 0) return error.Exl3EngineNotImplemented; // the clamped SwiGLU is Kernels3's
        const gu = ks.newKernel("turboexl3_gu", &gu_inputs, &gu_outputs, text_gu, header, true) orelse return error.KernelCreateFailed;
        const down = ks.newKernel("turboexl3_down", &down_inputs, &down_outputs, text_down, header, true) orelse return error.KernelCreateFailed;
        return .{ .geo = geo, .gu = gu, .down = down };
    }

    pub fn deinit(self: *Kernels) void {
        for (&self.cfg_gu) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        for (&self.cfg_down) |*c| if (c.*) |cc| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cc);
            c.* = null;
        };
        freeKernel(self.gu);
        freeKernel(self.down);
    }

    fn template(self: *const Kernels, buf: *[7]ks.TemplateArg) []const ks.TemplateArg {
        buf.* = .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.geo.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.geo.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.geo.dec.window.bits()) } },
            .{ .name = "HID", .value = .{ .int = @intCast(self.geo.hidden) } },
            .{ .name = "INTER", .value = .{ .int = @intCast(self.geo.inter) } },
            .{ .name = "TOPK", .value = .{ .int = @intCast(self.geo.topk) } },
            .{ .name = "PROBE", .value = .{ .int = self.geo.probe } },
        };
        return buf;
    }

    fn configs(self: *Kernels, rows: usize) !struct { gu: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config } {
        if (rows == 0 or rows > max_rows) return error.BadExl3Shape;
        if (self.cfg_gu[rows] == null) {
            var tbuf: [7]ks.TemplateArg = undefined;
            const t = self.template(&tbuf);
            const g = self.geo;
            const r: c_int = @intCast(rows);
            const h_shape = [_]c_int{ r * @as(c_int, @intCast(g.topk)), @intCast(g.inter) };
            self.cfg_gu[rows] = try ks.buildConfig(.{
                .grid = .{ (g.inter / 128) * 128, g.topk, @intCast(rows) },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&h_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
            const y_shape = [_]c_int{ r, @intCast(g.hidden) };
            self.cfg_down[rows] = try ks.buildConfig(.{
                .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 },
                .threadgroup = .{ 128, 1, 1 },
                .out_shapes = &.{&y_shape},
                .out_dtypes = &.{.float32},
                .template = t,
            });
        }
        return .{ .gu = self.cfg_gu[rows].?, .down = self.cfg_down[rows].? };
    }

    pub const Bank = struct {
        gate_t: mlx.mlx_array,
        gate_suh: mlx.mlx_array,
        gate_svh: mlx.mlx_array,
        up_t: mlx.mlx_array,
        up_suh: mlx.mlx_array,
        up_svh: mlx.mlx_array,
        down_t: mlx.mlx_array,
        down_suh: mlx.mlx_array,
        down_svh: mlx.mlx_array,
    };

    /// Launch 1: h f32 [rows * topk, inter] from x f32 [rows, hidden] and slots u32 [rows * topk].
    pub fn gateUp(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.gu, c.gu, &.{ x, slots, bank.gate_t, bank.gate_suh, bank.gate_svh, bank.up_t, bank.up_suh, bank.up_svh }, &outs, s);
        return outs[0];
    }

    /// Launch 2: y f32 [rows, hidden] from h, slots and scores f32 [rows * topk].
    pub fn downReduce(self: *Kernels, s: mlx.mlx_stream, h: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !mlx.mlx_array {
        const c = try self.configs(rows);
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.down, c.down, &.{ h, slots, scores, bank.down_t, bank.down_suh, bank.down_svh }, &outs, s);
        return outs[0];
    }

    /// The layer: both launches; the caller frees h and y.
    pub fn moe(self: *Kernels, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Bank, rows: usize) !struct { h: mlx.mlx_array, y: mlx.mlx_array } {
        const h = try self.gateUp(s, x, slots, bank, rows);
        errdefer _ = mlx.mlx_array_free(h);
        const y = try self.downReduce(s, h, slots, scores, bank, rows);
        return .{ .h = h, .y = y };
    }
};

test "turboexl3 decode texts: every template constant the texts read is one the launch passes" {
    const pin = comptime @import("test_support.zig").pins(@embedFile("kernels/test/pins/decode_texts.metal"));
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "TOPK", "PROBE" }) |name| {
        try t.expect(std.mem.indexOf(u8, header ++ text_gu, name) != null);
        try t.expect(std.mem.indexOf(u8, header ++ text_down, name) != null);
    }
    // every input the signatures name is read by its text, and the outputs are written
    for (gu_inputs) |n| try t.expect(std.mem.indexOf(u8, text_gu, std.mem.span(n)) != null);
    for (down_inputs) |n| try t.expect(std.mem.indexOf(u8, text_down, std.mem.span(n)) != null);
    try t.expect(std.mem.indexOf(u8, text_gu, pin[0]) != null);
    try t.expect(std.mem.indexOf(u8, text_down, pin[1]) != null);
}

test "turboexl3 decode geometry admission: 128-aligned widths, top-k within the reduce bank" {
    const t = std.testing;
    const ok: Geometry = .{ .hidden = 2560, .inter = 640, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 };
    try t.expect(ok.admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 600, .topk = 10, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
    try t.expect(!(Geometry{ .hidden = 2560, .inter = 640, .topk = 33, .rate = format.Rate.fromK(5), .dec = .mul1 }).admitted());
}

/// Form 3 (split-K planes). Launch 2: one threadgroup per (128-wide intermediate block, row-slot, group of SPT K slices);
/// the K partition into KS slices is the geometry's alone (a row's summation order never depends on how many rows share
/// the call), the row count only sets how many threadgroups share a row-slot's slices (KS / SPT). The four simdgroups are
/// projection x half of a slice; each slice's partial sums go to its plane of zg / zu f32 [KS, RK, inter] (written once
/// each; no atomics).
pub const text_gu3 = gu3_1 ++ rk_line ++ gu3_body;
/// gu3's pieces after its rk line, cut where gu3sf differs: after its reduction buffer (its declarations) and at the
/// planes' stores; and where the banked text differs: at the weight base (`gu3_w`)
pub const gu3_1 = @embedFile("kernels/decode/gu3_1.metal");
pub const gu3_2 = @embedFile("kernels/decode/gu3_2.metal");
pub const gu3_w = @embedFile("kernels/decode/gu3_w.metal");
pub const gu3_3 = @embedFile("kernels/decode/gu3_3.metal");
pub const gu3_stores = @embedFile("kernels/decode/gu3_stores.metal");
pub const gu3_4 = @embedFile("kernels/decode/gu3_4.metal");
pub const gu3_body = gu3_2 ++ gu3_w ++ gu3_3 ++ gu3_stores ++ gu3_4;

/// Form 3, launch 3: per (128-wide intermediate block, row-slot): sum the planes, f16 round, H128, svh, f16 round,
/// SwiGLU, then the down's input rotation (f16 round, down suh, H128, f16 round) into hd in lane order.
pub const text_fin3 = @embedFile("kernels/decode/fin3.metal");

/// Form 3, launch 4: the down GEMV as split-K planes zd f32 [KS2, RK, hidden], the partition fixed per geometry and SPT
/// slices per threadgroup as in launch 2.
pub const text_down3 = down3_1 ++ rk_line ++ down3_2 ++ down3_w ++ down3_3;
/// down3's pieces: its constants and grid line, then its body cut at the weight base (`down3_w`; the banked text's
/// differs)
pub const down3_1 = @embedFile("kernels/decode/down3_1.metal");
pub const down3_2 = @embedFile("kernels/decode/down3_2.metal");
pub const down3_w = @embedFile("kernels/decode/down3_w.metal");
pub const down3_3 = @embedFile("kernels/decode/down3_3.metal");

/// Form 3, launch 5: per (128-wide hidden block, row): for each slot in order sum the planes, f16 round, H128, svh,
/// f16 round, times the score; y f32 [rows, hidden].
pub const text_dfin3 = dfin3_1 ++ dfin3_expert ++ dfin3_2;
/// dfin3's pieces, cut at the slot's expert line (the banked text's differs)
pub const dfin3_1 = @embedFile("kernels/decode/dfin3_1.metal");
const dfin3_expert = @embedFile("kernels/decode/dfin3_expert.metal");
pub const dfin3_2 = @embedFile("kernels/decode/dfin3_2.metal");

/// Form 3's prepare for gate and up in one launch: xg, xu half [RK, DIM] in lane order from x [rows, DIM] and the two suh.
pub const text_prep2 = prep2_1 ++ expert_line ++ prep2_row ++ prep2_body;
/// prep2's constants and grid lines, and its row line after the expert line
pub const prep2_1 = @embedFile("kernels/decode/prep2_1.metal");
pub const prep2_row = @embedFile("kernels/decode/prep2_row.metal");
/// A row-slot's expert, read where the kernel starts (prep2, fin7); the banked texts read a bank and a slot row there
/// (`prep2b_expert`, `fin7b_expert`)
pub const expert_line = @embedFile("kernels/decode/expert_line.metal");
pub const prep2_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh" };
pub const prep2_outputs = [_][*:0]const u8{ "xg", "xu", "sg", "su" };

pub const gu3_inputs = [_][*:0]const u8{ "xgp", "xup", "slots", "gate_t", "up_t" };
pub const gu3_outputs = [_][*:0]const u8{ "zg", "zu" };
pub const fin3_inputs = [_][*:0]const u8{ "zg", "zu", "slots", "gate_svh", "up_svh", "down_suh", "xsg", "xsu" };
pub const fin3_outputs = [_][*:0]const u8{ "hd", "hs" };
pub const down3_inputs = [_][*:0]const u8{ "hdp", "slots", "down_t" };
pub const down3_outputs = [_][*:0]const u8{"zd"};
pub const dfin3_inputs = [_][*:0]const u8{ "zd", "slots", "scores", "down_svh", "hsx" };


/// Form 3: five launches (prepare gate and up, split-K gate/up planes, finish + SwiGLU + down prepare, split-K down
/// planes, finish + reduce); KS slices fill the GPU at one row.
pub const Kernels3 = struct {
    geo: Geometry,
    ks_gu: u32,
    ks_dn: u32,
    /// output tiles per GEMV threadgroup (8, 4, 2 or 1): 2 bt f32 accumulators per lane, 128 / (16 bt) x more threadgroups
    bt: u32,
    /// row-slots per GEMV lead threadgroup from two rows up (2: slots that chose the same expert share one decode)
    group: u32,
    /// how many threadgroups share a row-slot's K slices: `adaptive` (all KS at one row, half at 2-3, a quarter from 4: the
    /// row-slots fill the GPU by themselves), `full` (one threadgroup per slice at every row count); either way the K
    /// partition, so every row's bytes, is the geometry's alone
    kpar: KPar = .adaptive,
    /// `adaptive`'s divisors of the K slices for 1, 2, 3 and 4+ rows (1, 2, 4, 8 each): z = max(1, KS / kdiv) threadgroups
    /// along K per row-slot; set before the first call
    kdiv: [4]u32 = .{ 1, 2, 2, 4 },
    /// y's dtype (the x the host passes: bf16 served; f32 in the tests and the bench); set before the first call
    out_dtype: mlx.mlx_dtype = .float32,
    /// MUL1: the GEMVs sum x times the codeword's byte sum and the finishes apply the affine map once per output
    /// (false decodes every weight to its f16 value, the reference's rounding)
    defer_affine: bool = false,
    prep: mlx.mlx_fast_metal_kernel,
    gu: mlx.mlx_fast_metal_kernel,
    fin: mlx.mlx_fast_metal_kernel,
    down: mlx.mlx_fast_metal_kernel,
    dfin: mlx.mlx_fast_metal_kernel,
    /// form 7's launches 3 and 5 (fin3 and dfin3 with the H128s in registers, dfin's slots side by side): the same bytes
    fin7: mlx.mlx_fast_metal_kernel,
    dfin7: mlx.mlx_fast_metal_kernel,
    /// form 8's launch 1 (prep2 inside gu3, each threadgroup rotating its own K span): the same bytes, four launches
    gu8: mlx.mlx_fast_metal_kernel,
    /// form 9's launch shape, its own (forms 3/7/8 keep bt and kdiv): output tiles per GEMV threadgroup at one row (bt1) and
    /// from two rows (bt9), and the K-slice divisors (kdiv9); 1, 2, 4, 8 each. The K partition, so the bytes, do not depend
    /// on any of them. Set before the first call.
    bt1: u32 = 2,
    bt9: u32 = 2,
    kdiv9: [4]u32 = .{ 1, 1, 1, 1 },
    /// form 9's down-GEMV K-slice divisors when they differ from kdiv9 (null: kdiv9)
    kdiv9d: ?[4]u32 = null,
    /// form 9 runs the prepare inside the gate|up launch (gu8) up to this many rows (where `gu8Admitted`), prep2 + gu3 above
    gu8_rows: usize = 1,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),
    /// form 9's configs
    cfg9: [Kernels.max_rows + 1]?Cfg = @splat(null),


    pub const Cfg = struct { prep_h: mlx.mlx_fast_metal_kernel_config, gu: mlx.mlx_fast_metal_kernel_config, fin: mlx.mlx_fast_metal_kernel_config, down: mlx.mlx_fast_metal_kernel_config, dfin: mlx.mlx_fast_metal_kernel_config, fin7: mlx.mlx_fast_metal_kernel_config, dfin7: mlx.mlx_fast_metal_kernel_config, gu8: mlx.mlx_fast_metal_kernel_config };
    const prep_in = [_][*:0]const u8{ "x", "slots", "suh" };
    const prep_out = [_][*:0]const u8{"xp"};

    pub const KPar = enum { adaptive, full };

    /// K slices, fixed per geometry: gate/up over hidden/16 tiles in KS x 2 parts, down over inter/16 tiles in KS2 x 4
    /// parts; each part at least 4 tiles.
    pub fn init(geo: Geometry, bt: u32, ks_max: u32) !Kernels3 {
        return initWith(geo, bt, ks_max, 2);
    }

    /// `group` 2 pairs the row-slots that chose the same expert under one threadgroup from two rows up (the weights
    /// read and decoded once, fed to both members' accumulators; Sushi's grouped kernels do the same); 1 keeps every
    /// row-slot its own threadgroup.
    pub fn initWith(geo: Geometry, bt: u32, ks_max: u32, group: u32) !Kernels3 {
        if (!geo.admitted() or (bt != 8 and bt != 4 and bt != 2 and bt != 1) or (ks_max != 1 and ks_max != 2 and ks_max != 4 and ks_max != 8) or (group != 1 and group != 2)) return error.BadExl3Shape;
        var ks_gu: u32 = ks_max;
        while (ks_gu > 1 and (geo.hidden / 16) % (ks_gu * 2) != 0) ks_gu /= 2;
        var ks_dn: u32 = ks_max;
        while (ks_dn > 1 and ((geo.inter / 16) % (ks_dn * 4) != 0 or (geo.inter / 16) / (ks_dn * 4) < 2)) ks_dn /= 2;
        return .{
            .geo = geo,
            .ks_gu = ks_gu,
            .ks_dn = ks_dn,
            .bt = bt,
            .group = group,
            .defer_affine = geo.dec.codebook == .mul1,
            .prep = ks.newKernel("turboexl3_prep2", &prep2_inputs, &prep2_outputs, text_prep2, header, true) orelse return error.KernelCreateFailed,
            .gu = ks.newKernel("turboexl3_gu3", &gu3_inputs, &gu3_outputs, text_gu3, header, true) orelse return error.KernelCreateFailed,
            .fin = ks.newKernel("turboexl3_fin3", &fin3_inputs, &fin3_outputs, text_fin3, header, true) orelse return error.KernelCreateFailed,
            .down = ks.newKernel("turboexl3_down3", &down3_inputs, &down3_outputs, text_down3, header, true) orelse return error.KernelCreateFailed,
            .dfin = ks.newKernel("turboexl3_dfin3", &dfin3_inputs, &down_outputs, text_dfin3, header, true) orelse return error.KernelCreateFailed,
            // a clamped geometry's fin7 is the clamped variant, so every launch of fin7 (forms 7, 8 and 9) clamps
            .fin7 = (if (geo.limit == 0)
                ks.newKernel("turboexl3_fin7", &fin3_inputs, &fin3_outputs, text_fin7, header5, true)
            else
                ks.newKernel("turboexl3_fin7c", &fin3_inputs, &fin3_outputs, text_fin7c, header5, true)) orelse return error.KernelCreateFailed,
            .dfin7 = ks.newKernel("turboexl3_dfin7", &dfin3_inputs, &down_outputs, text_dfin7, header5, true) orelse return error.KernelCreateFailed,
            .gu8 = ks.newKernel("turboexl3_gu8", &gu8_inputs, &gu8_outputs, text_gu8, header5, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels3) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        for (&self.cfg9) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep_h, cc.gu, cc.fin, cc.down, cc.dfin, cc.fin7, cc.dfin7, cc.gu8 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.prep, self.gu, self.fin, self.down, self.dfin, self.fin7, self.dfin7, self.gu8 }) |k| freeKernel(k);
    }

    /// Form 8's gate|up launch rotates whole 128-blocks: a threadgroup's span (its SPT slices of hidden / KS values) must be
    /// 128-aligned; otherwise form 8 runs form 7's launches.
    pub fn gu8Admitted(self: *const Kernels3, rows: usize, kdiv: [4]u32) bool {
        const spt = self.ks_gu / self.kThreadgroups(self.ks_gu, rows, kdiv);
        if ((self.geo.hidden / self.ks_gu * spt) % 128 != 0 or self.geo.hidden % self.ks_gu != 0) return false;
        // gu8's threadgroup memory (gu8_1.metal, gu8_2.metal) within Metal's 32 KB: xs, the group members' prepared gate and
        // up spans (GROUP x 2 x SPT x hidden / KS halves), and red (SPT x 4 x 2 x 16 x BT floats) at the widest BT a form
        // runs. DeepSeek-V4.1's 5120-wide hidden at two rows (GROUP 2, SPT 4) needs 43,008 bytes: its prepare runs in prep2.
        const grp: u32 = if (rows >= 2) self.group else 1;
        const bt = @max(self.bt, @max(self.bt1, self.bt9));
        return 4 * grp * spt * (self.geo.hidden / self.ks_gu) + 512 * spt * bt <= 32768;
    }

    /// Threadgroups along K per row-slot for a row count (`adaptive`, `kdiv`'s defaults): all ks slices apart at one row, two
    /// slices per threadgroup at 2-3 rows, four from 4 rows (the row-slots fill the GPU by themselves). The slices themselves,
    /// and so the summation order, do not change: a threadgroup loops over its ks / z slices and writes each slice's plane.
    pub fn kThreadgroups(self: *const Kernels3, ks_fixed: u32, rows: usize, kdiv: [4]u32) u32 {
        if (self.kpar == .full) return ks_fixed;
        const div: u32 = kdiv[@min(rows, 4) - 1];
        return @max(1, ks_fixed / @max(1, div));
    }

    fn cfgs(self: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = try self.buildCfgs(rows, self.bt, self.kdiv);
        self.cfg[rows] = c;
        return c;
    }

    /// Form 9's configs: bt1 tiles per GEMV threadgroup at one row, bt9 from two, kdiv9's K threadgroups.
    pub fn cfgs9(self: *Kernels3, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg9[rows]) |c| return c;
        const bt = if (rows == 1) self.bt1 else self.bt9;
        if (bt != 1 and bt != 2 and bt != 4 and bt != 8) return error.BadExl3Shape;
        const c = try self.buildCfgsSplit(rows, bt, self.kdiv9, self.kdiv9d orelse self.kdiv9);
        self.cfg9[rows] = c;
        return c;
    }

    fn buildCfgs(self: *Kernels3, rows: usize, bt: u32, kdiv: [4]u32) !Cfg {
        return self.buildCfgsSplit(rows, bt, kdiv, kdiv);
    }

    pub fn buildCfgsSplit(self: *Kernels3, rows: usize, bt: u32, kdiv: [4]u32, kdiv_dn: [4]u32) !Cfg {
        return self.buildCfgsGroup(rows, bt, kdiv, kdiv_dn, self.group);
    }

    /// `buildCfgsSplit` at `group` row-slots per lead (the GEMVs' GROUP from two rows up)
    pub fn buildCfgsGroup(self: *Kernels3, rows: usize, bt: u32, kdiv: [4]u32, kdiv_dn: [4]u32, group: u32) !Cfg {
        const g = self.geo;
        const z_gu = self.kThreadgroups(self.ks_gu, rows, kdiv);
        const z_dn = self.kThreadgroups(self.ks_dn, rows, kdiv_dn);
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
        const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
        const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_gu_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks_dn_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_dn) } };
        const spt_gu_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(self.ks_gu / z_gu) } };
        const spt_dn_t = ks.TemplateArg{ .name = "SPT", .value = .{ .int = @intCast(self.ks_dn / z_dn) } };
        const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(bt) } };
        const grp_t = ks.TemplateArg{ .name = "GROUP", .value = .{ .int = @intCast(if (rows >= 2) group else 1) } };
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const nbh_t = ks.TemplateArg{ .name = "NBH", .value = .{ .int = @intCast(g.hidden / 128) } };
        const nbi_t = ks.TemplateArg{ .name = "NBI", .value = .{ .int = @intCast(g.inter / 128) } };
        // fin7's constants, the clamp last where the geometry clamps (an unclamped fin7 takes exactly its five)
        const fin7_t = [_]ks.TemplateArg{ inter_t, rkt, ks_gu_t, def_t, nbh_t, .{ .name = "LIMIT", .value = .{ .int = @intCast(g.limit) } } };
        const fin7_n: usize = if (g.limit != 0) fin7_t.len else fin7_t.len - 1;
        const bw: u32 = 16 * bt;
        return .{
            .prep_h = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ rkc, H }, &[_]c_int{ rkc, @intCast(g.hidden / 128) }, &[_]c_int{ rkc, @intCast(g.hidden / 128) } }, .out_dtypes = &.{ .float16, .float16, .float32, .float32 }, .template = &.{ .{ .name = "DIM", .value = .{ .int = H } }, topk } }),
            .gu = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, z_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I } }, .out_dtypes = &.{ .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t } }),
            .fin = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 128, rk, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{ inter_t, rkt, ks_gu_t, def_t, nbh_t } }),
            .down = try ks.buildConfig(.{ .grid = .{ (g.hidden / bw) * 128, rk, z_dn }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.ks_dn), rkc, H }}, .out_dtypes = &.{.float32}, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_dn_t, spt_dn_t, bt_t, grp_t, def_t } }),
            .dfin = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{self.out_dtype}, .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t } }),
            .fin7 = try ks.buildConfig(.{ .grid = .{ (g.inter / 128) * 32, rk, 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ rkc, I }, &[_]c_int{ rkc, @intCast(g.inter / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = fin7_t[0..fin7_n] }),
            .dfin7 = try ks.buildConfig(.{ .grid = .{ (g.hidden / 128) * 32 * g.topk, @intCast(rows), 1 }, .threadgroup = .{ 32 * g.topk, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(rows), H }}, .out_dtypes = &.{self.out_dtype}, .template = &.{ hid, topk, rkt, ks_dn_t, out_t, def_t, nbi_t } }),
            .gu8 = try ks.buildConfig(.{ .grid = .{ (g.inter / bw) * 128, rk, z_gu }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ rkc, @intCast(g.hidden / 128) }, &[_]c_int{ rkc, @intCast(g.hidden / 128) } }, .out_dtypes = &.{ .float32, .float32, .float32, .float32 }, .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_gu_t, spt_gu_t, bt_t, grp_t, def_t, topk } }),
        };
    }

    pub fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// The five launches (one prepare for both projections); the caller frees y.
    pub fn moe(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize) !mlx.mlx_array {
        return self.moeForm(s, x, slots, scores, bank, rows, 3);
    }

    /// `form` 3: the five launches above; 7: the same GEMVs with fin7 / dfin7 as launches 3 and 5; 8: form 7 with the
    /// prepare inside the gate|up launch (gu8, four launches) where `gu8Admitted`; 9: form 8 at one row, form 7 from two
    /// rows (gu8 re-rotates its whole span in every output-tile threadgroup, a cost that grows once a threadgroup carries
    /// several slices), in its own launch shape (bt1, bt9, kdiv9). Every form, form 3's bytes. A clamped geometry runs forms
    /// 7, 8 and 9 (fin7's clamped variant) and refuses form 3 (error.Exl3EngineNotImplemented: fin3 has no clamp).
    pub fn moeForm(self: *Kernels3, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, form: u8) !mlx.mlx_array {
        if (self.geo.limit != 0 and form < 7) return error.Exl3EngineNotImplemented;
        const one_row9 = form == 9 and rows <= self.gu8_rows;
        const c = if (form == 9) try self.cfgs9(rows) else try self.cfgs(rows);
        const f7 = form >= 7;
        // dfin7 runs one simdgroup per routed slot and writes each block's 128 outputs from its first 128 threads, so it needs
        // a top-k of 4 or more; below that form 3's dfin (the same bytes) finishes the down projection.
        const f7d = f7 and self.geo.topk >= 4;
        // planes[0..2] = zg, zu (the split-K gate|up planes), planes[2..4] = the inputs' per-block sums
        var planes: [4]mlx.mlx_array = undefined;
        if ((form == 8 and self.gu8Admitted(rows, self.kdiv)) or (one_row9 and self.gu8Admitted(rows, self.kdiv9))) {
            try ks.applyConfig(self.gu8, c.gu8, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &planes, s);
        } else {
            var xp: [4]mlx.mlx_array = undefined; // xg, xu, and their per-block input sums
            try ks.applyConfig(self.prep, c.prep_h, &.{ x, slots, bank.gate_suh, bank.up_suh }, &xp, s);
            defer _ = mlx.mlx_array_free(xp[0]);
            defer _ = mlx.mlx_array_free(xp[1]);
            errdefer _ = mlx.mlx_array_free(xp[2]);
            errdefer _ = mlx.mlx_array_free(xp[3]);
            try ks.applyConfig(self.gu, c.gu, &.{ xp[0], xp[1], slots, bank.gate_t, bank.up_t }, planes[0..2], s);
            planes[2] = xp[2];
            planes[3] = xp[3];
        }
        defer for (planes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var hd: [2]mlx.mlx_array = undefined; // the down's prepared input and its per-block sums
        try ks.applyConfig(if (f7) self.fin7 else self.fin, if (f7) c.fin7 else c.fin, &.{ planes[0], planes[1], slots, bank.gate_svh, bank.up_svh, bank.down_suh, planes[2], planes[3] }, &hd, s);
        defer for (hd) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const zd = try one(self.down, c.down, &.{ hd[0], slots, bank.down_t }, s);
        defer _ = mlx.mlx_array_free(zd);
        return one(if (f7d) self.dfin7 else self.dfin, if (f7d) c.dfin7 else c.dfin, &.{ zd, slots, scores, bank.down_svh, hd[1] }, s);
    }


};

/// Form 5's helpers, appended to `header` for its three kernels (`header5`): expert groups of up to GM row-slots, the GEMV over
/// prepared inputs held in threadgroup memory, and the H128 of a simdgroup's 128 values held 4 per lane.
const metal_form5 = @embedFile("kernels/decode/form5.metal");

pub const header5: [:0]const u8 = lane.metal_helpers ++ metal_common ++ metal_gemv_core ++ metal_form5;

/// Form 5, launch 1: per (BT output tiles, lead row-slot, K slice): the expert group's members' rotated inputs over the slice
/// (prep2's arithmetic, into threadgroup memory: no prepare launch, no device round trip), then gu3's GEMV for every member
/// at once (each weight decoded once per group) into the same split-K planes zg / zu f32 [KS, RK, inter], and each 128-block's
/// input sum (xsg / xsu [RK, hidden / 128], written by the blk 0 threadgroups) for the deferred MUL1 map.
pub const text_gu5 = @embedFile("kernels/decode/gu5.metal");

/// Form 5, launch 2: per (128-wide hidden block, lead row-slot), 8 simdgroups: every member's mid over the whole intermediate
/// (fin3's arithmetic: the planes' sum, the deferred map, f16 round, H128, svh, f16 round, SwiGLU, the down's suh, H128, f16
/// round, lane order, and each 128-block's sum) into threadgroup memory, then the down GEMV of the block's 8 output tiles
/// (one per simdgroup) for every member in down3's K partition, then dfin3's per-slot finish up to the score: the deferred
/// map, f16 round, H128, svh, f16 round -> wv half [RK, hidden].
const dn5_head = @embedFile("kernels/decode/dn5_head.metal");
/// Form 5's down tail: the finished values to wv for launch 3.
const dn5_tail = @embedFile("kernels/decode/dn5_tail.metal");
/// Form 6's down tail: the finished values stored through coherent pointers, then member j's row counts this 128-block's
/// arrival (device-scope seq_cst fences around a relaxed fetch_add; no waiting) and the simdgroup that brings
/// the row's TOPK-th slot reduces the block in slot order (red5's fma chain, so red5's bytes) straight into y.
const dn6_tail = @embedFile("kernels/decode/dn6_tail.metal");

pub const text_dn5: [:0]const u8 = dn5_head ++ dn5_tail;
pub const text_dn6: [:0]const u8 = dn5_head ++ dn6_tail;

/// Form 5, launch 3: y [rows, hidden] = the score-weighted sum of a row's slots in slot order (dfin3's fma chain).
pub const text_red5 = @embedFile("kernels/decode/red5.metal");

const gu5_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh", "gate_t", "up_t" };
const gu5_outputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "cnt" };
const dn5_inputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "slots", "gate_svh", "up_svh", "down_suh", "down_t", "down_svh" };
const dn5_outputs = [_][*:0]const u8{"wv"};
const dn6_inputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu", "slots", "gate_svh", "up_svh", "down_suh", "down_t", "down_svh", "cnt", "scores" };
const dn6_outputs = [_][*:0]const u8{ "wv", "y" };
const red5_inputs = [_][*:0]const u8{ "wv", "scores" };
const red5_outputs = [_][*:0]const u8{"y"};

/// Form 7, launch 3 (fin3 with the H128s in registers): one simdgroup per (128-wide intermediate block, row-slot), lane L
/// holding elements L + 32 q; fin3's arithmetic in fin3's order (`turboexl3_h128r` is `turboexl3_h128`'s stages, the block sum
/// the same four simd sums), so fin3's bytes without its twelve threadgroup barriers.
pub const text_fin7 = fin7_1 ++ expert_line ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu ++ fin7_5;
/// fin7's pieces: its constants and grid lines; the expert line (`expert_line`: the banked text reads a bank there); its
/// body, cut at the planes' read (gu3sf reads the planes through coherent pointers), at the input sums' read and at the
/// SwiGLU's line (`text_fin7c` clamps there)
pub const fin7_1 = @embedFile("kernels/decode/fin7_1.metal");
pub const fin7_2 = @embedFile("kernels/decode/fin7_2.metal");
pub const fin7_planes = @embedFile("kernels/decode/fin7_planes.metal");
pub const fin7_3 = @embedFile("kernels/decode/fin7_3.metal");
pub const fin7_sums = @embedFile("kernels/decode/fin7_sums.metal");
pub const fin7_4 = @embedFile("kernels/decode/fin7_4.metal");
pub const fin7_swiglu = @embedFile("kernels/decode/fin7_swiglu.metal");
pub const fin7_5 = @embedFile("kernels/decode/fin7_5.metal");

/// Form 7's launch 3 with the clamped SwiGLU (`Geometry.limit`, the template int LIMIT; `root.moeClamped`): fin7 with its
/// SwiGLU line replaced by the clamp (the gate clamped from above at LIMIT and up to [-LIMIT, LIMIT], then fin7's SwiGLU
/// on them), every other statement fin7's, so where no value passes the limit it gives fin7's bytes.
pub const text_fin7c = fin7_1 ++ expert_line ++ fin7_2 ++ fin7_planes ++ fin7_3 ++ fin7_sums ++ fin7_4 ++ fin7_swiglu_clamp ++ fin7_5;
pub const fin7_swiglu_clamp = @embedFile("kernels/decode/fin7_swiglu_clamp.metal");

/// Form 7, launch 5 (dfin3 with the slots side by side): per (128-wide hidden block, row), simdgroup k finishes slot k (the
/// planes' sum, the deferred map, f16 round, H128 in registers, svh, f16 round) into threadgroup memory, then 128 threads
/// run dfin3's score fma chain in slot order: dfin3's bytes, one H128 deep where dfin3 runs TOPK of them in a row.
pub const text_dfin7 = @embedFile("kernels/decode/dfin7.metal");

/// Form 8, launch 1 (prep2 + gu3 in one launch): gu3's threadgroup over its SPT slices, whose span (SPT hidden / KS values,
/// 128-aligned: the host admits it only then) the four simdgroups first rotate for every member and both projections in
/// prep2's arithmetic into threadgroup memory; then gu3's GEMV reads the input from there. The blk 0 threadgroups write
/// each block's input sum (xsg / xsu [RK, hidden / 128]) for fin's deferred map. gu3's planes, so form 3's bytes.
pub const text_gu8 = @embedFile("kernels/decode/gu8_1.metal") ++ rk_line ++ @embedFile("kernels/decode/gu8_2.metal");
pub const gu8_inputs = [_][*:0]const u8{ "x", "slots", "gate_suh", "up_suh", "gate_t", "up_t" };
pub const gu8_outputs = [_][*:0]const u8{ "zg", "zu", "xsg", "xsu" };

/// The routed bodies under the v2 heads: rk = y - SYOFF (SYOFF 0: v1's rows; > 0: the shared rows first).
pub const rk_line = @embedFile("kernels/decode/rk_line.metal");
/// prep2's body after its index lines (the router fold splices it)
pub const prep2_body = @embedFile("kernels/decode/prep2_2.metal");


/// Form 5: three launches (prepare + gate|up GEMV, mid + down GEMV + per-slot finish, the slot reduce) where form 3 has five,
/// in form 3's K partitions and rounding points, so a row's bytes are form 3's; the row-slots that chose one expert (up to
/// `gmax`, members from other rows) share one threadgroup, so that expert's weights are read and decoded once per group.
pub const Kernels5 = struct {
    geo: Geometry,
    ks_gu: u32,
    ks_dn: u32,
    /// output tiles per gate|up threadgroup (1, 2, 4, 8): the inline prepare is shared by bt tiles
    bt: u32,
    /// row-slots per expert group (1..4); a call of r rows groups up to min(gmax, r)
    gmax: u32,
    /// the integer-rate reader (n % 16 == 0; the same codewords as the general reader)
    intr: bool,
    /// y's dtype (the x the host passes: bf16 served; f32 in the tests and the bench); set before the first call
    out_dtype: mlx.mlx_dtype = .float32,
    /// MUL1's deferred affine map (false decodes every weight to its f16 value)
    defer_affine: bool = false,
    gu: mlx.mlx_fast_metal_kernel,
    dn: mlx.mlx_fast_metal_kernel,
    red: mlx.mlx_fast_metal_kernel,
    /// form 6's down (the slot reduce fused by arrival counting): two launches
    dn6: mlx.mlx_fast_metal_kernel,
    cfg: [Kernels.max_rows + 1]?Cfg = @splat(null),

    const Cfg = struct { gu: mlx.mlx_fast_metal_kernel_config, dn: mlx.mlx_fast_metal_kernel_config, red: mlx.mlx_fast_metal_kernel_config, dn6: mlx.mlx_fast_metal_kernel_config };

    /// The K partitions are form 3's (`Kernels3.initWith` at the same ks_max): gate|up hidden / 16 tiles in KS x 2 parts,
    /// down inter / 16 tiles in KS2 x 4 parts of at least 2 tiles.
    pub fn init(geo: Geometry, bt: u32, ks_max: u32, gmax: u32) !Kernels5 {
        if (!geo.admitted() or (bt != 1 and bt != 2 and bt != 4 and bt != 8) or (ks_max != 1 and ks_max != 2 and ks_max != 4 and ks_max != 8) or gmax < 1 or gmax > 4) return error.BadExl3Shape;
        if ((geo.inter / 16) % bt != 0) return error.BadExl3Shape;
        if (geo.limit != 0) return error.Exl3EngineNotImplemented; // the clamped SwiGLU is Kernels3's (dn5 has no clamp)
        var ks_gu: u32 = ks_max;
        while (ks_gu > 1 and (geo.hidden / 16) % (ks_gu * 2) != 0) ks_gu /= 2;
        var ks_dn: u32 = ks_max;
        while (ks_dn > 1 and ((geo.inter / 16) % (ks_dn * 4) != 0 or (geo.inter / 16) / (ks_dn * 4) < 2)) ks_dn /= 2;
        return .{
            .geo = geo,
            .ks_gu = ks_gu,
            .ks_dn = ks_dn,
            .bt = bt,
            .gmax = gmax,
            .intr = geo.rate.n % 16 == 0,
            .defer_affine = geo.dec.codebook == .mul1,
            .gu = ks.newKernel("turboexl3_gu5", &gu5_inputs, &gu5_outputs, text_gu5, header5, true) orelse return error.KernelCreateFailed,
            .dn = ks.newKernel("turboexl3_dn5", &dn5_inputs, &dn5_outputs, text_dn5, header5, true) orelse return error.KernelCreateFailed,
            .red = ks.newKernel("turboexl3_red5", &red5_inputs, &red5_outputs, text_red5, header5, true) orelse return error.KernelCreateFailed,
            .dn6 = ks.newKernel("turboexl3_dn6", &dn6_inputs, &dn6_outputs, text_dn6, header5, true) orelse return error.KernelCreateFailed,
        };
    }

    pub fn deinit(self: *Kernels5) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.gu, cc.dn, cc.red, cc.dn6 }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        inline for (.{ self.gu, self.dn, self.red, self.dn6 }) |k| freeKernel(k);
    }

    fn cfgs(self: *Kernels5, rows: usize) !Cfg {
        if (rows == 0 or rows > Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const g = self.geo;
        const rk: u32 = @as(u32, @intCast(rows)) * g.topk;
        const rkc: c_int = @intCast(rk);
        const H: c_int = @intCast(g.hidden);
        const I: c_int = @intCast(g.inter);
        const nbh: c_int = @intCast(g.hidden / 128);
        const gm: u32 = @min(self.gmax, @as(u32, @intCast(rows)));
        const nhw = ks.TemplateArg{ .name = "NHW", .value = .{ .int = @intCast(g.rate.n) } };
        const cb = ks.TemplateArg{ .name = "CODEBOOK", .value = .{ .int = @backingInt(g.dec.codebook) } };
        const wb = ks.TemplateArg{ .name = "WBITS", .value = .{ .int = @intCast(g.dec.window.bits()) } };
        const hid = ks.TemplateArg{ .name = "HID", .value = .{ .int = H } };
        const inter_t = ks.TemplateArg{ .name = "INTER", .value = .{ .int = I } };
        const rkt = ks.TemplateArg{ .name = "RKC", .value = .{ .int = rkc } };
        const topk = ks.TemplateArg{ .name = "TOPK", .value = .{ .int = @intCast(g.topk) } };
        const ks_t = ks.TemplateArg{ .name = "KS", .value = .{ .int = @intCast(self.ks_gu) } };
        const ks2_t = ks.TemplateArg{ .name = "KS2", .value = .{ .int = @intCast(self.ks_dn) } };
        const bt_t = ks.TemplateArg{ .name = "BT", .value = .{ .int = @intCast(self.bt) } };
        const gm_t = ks.TemplateArg{ .name = "GMAX", .value = .{ .int = @intCast(gm) } };
        const def_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        const ir_t = ks.TemplateArg{ .name = "IR", .value = .{ .int = if (self.intr) 1 else 0 } };
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.out_dtype } };
        const gu = try ks.buildConfig(.{
            .grid = .{ (g.inter / (16 * self.bt)) * 128, rk, self.ks_gu },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ @intCast(self.ks_gu), rkc, I }, &[_]c_int{ rkc, nbh }, &[_]c_int{ rkc, nbh }, &[_]c_int{ @as(c_int, @intCast(rows)) * nbh } },
            .out_dtypes = &.{ .float32, .float32, .float32, .float32, .uint32 },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, bt_t, gm_t, topk, def_t, ir_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(gu);
        const dn = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 256, rk, 1 },
            .threadgroup = .{ 256, 1, 1 },
            .out_shapes = &.{&[_]c_int{ rkc, H }},
            .out_dtypes = &.{.float16},
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, ks2_t, gm_t, def_t, ir_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(dn);
        const red = try ks.buildConfig(.{
            .grid = .{ g.hidden, @intCast(rows), 1 },
            .threadgroup = .{ 128, 1, 1 },
            .out_shapes = &.{&[_]c_int{ @intCast(rows), H }},
            .out_dtypes = &.{self.out_dtype},
            .template = &.{ hid, topk, out_t },
        });
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(red);
        const dn6 = try ks.buildConfig(.{
            .grid = .{ (g.hidden / 128) * 256, rk, 1 },
            .threadgroup = .{ 256, 1, 1 },
            .out_shapes = &.{ &[_]c_int{ rkc, H }, &[_]c_int{ @intCast(rows), H } },
            .out_dtypes = &.{ .float16, self.out_dtype },
            .template = &.{ nhw, cb, wb, hid, inter_t, rkt, ks_t, ks2_t, gm_t, def_t, ir_t, topk, out_t },
        });
        const c: Cfg = .{ .gu = gu, .dn = dn, .red = red, .dn6 = dn6 };
        self.cfg[rows] = c;
        return c;
    }

    /// The three launches (`fused`: form 6, the two with the slot reduce in the down launch); the caller frees y. x [rows,
    /// hidden], slots and scores [rows * topk], as `Kernels3.moe`.
    pub fn moe(self: *Kernels5, s: mlx.mlx_stream, x: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, bank: Kernels.Bank, rows: usize, fused: bool) !mlx.mlx_array {
        const c = try self.cfgs(rows);
        var z: [5]mlx.mlx_array = undefined; // zg, zu, the inputs' per-block sums, form 6's zeroed counters
        try ks.applyConfig(self.gu, c.gu, &.{ x, slots, bank.gate_suh, bank.up_suh, bank.gate_t, bank.up_t }, &z, s);
        defer for (z) |a| {
            _ = mlx.mlx_array_free(a);
        };
        if (fused) {
            var o6: [2]mlx.mlx_array = undefined; // wv (the exchange), y
            try ks.applyConfig(self.dn6, c.dn6, &.{ z[0], z[1], z[2], z[3], slots, bank.gate_svh, bank.up_svh, bank.down_suh, bank.down_t, bank.down_svh, z[4], scores }, &o6, s);
            _ = mlx.mlx_array_free(o6[0]);
            return o6[1];
        }
        var wv: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.dn, c.dn, &.{ z[0], z[1], z[2], z[3], slots, bank.gate_svh, bank.up_svh, bank.down_suh, bank.down_t, bank.down_svh }, &wv, s);
        defer _ = mlx.mlx_array_free(wv[0]);
        var y: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(self.red, c.red, &.{ wv[0], scores }, &y, s);
        return y[0];
    }
};

test "turboexl3 decode forms 7 and 8 texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "INTER", "RKC", "KS", "DEFER", "NBH" }) |name| try t.expect(std.mem.indexOf(u8, text_fin7, name) != null);
    for ([_][]const u8{ "HID", "TOPK", "RKC", "KS", "OUT_T", "DEFER", "NBI" }) |name| try t.expect(std.mem.indexOf(u8, text_dfin7, name) != null);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "SPT", "BT", "GROUP", "DEFER", "TOPK" }) |name| try t.expect(std.mem.indexOf(u8, text_gu8, name) != null);
    for (fin3_inputs ++ fin3_outputs) |n| try t.expect(std.mem.indexOf(u8, text_fin7, std.mem.span(n)) != null);
    for (dfin3_inputs ++ down_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dfin7, std.mem.span(n)) != null);
    for (gu8_inputs ++ gu8_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu8, std.mem.span(n)) != null);
}





test "turboexl3 decode form 5 texts: every template constant read is passed, every signature name is used" {
    const t = std.testing;
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS", "BT", "GMAX", "TOPK", "DEFER", "IR" }) |name| try t.expect(std.mem.indexOf(u8, text_gu5, name) != null);
    for ([_][]const u8{ "NHW", "CODEBOOK", "WBITS", "HID", "INTER", "RKC", "KS2", "GMAX", "DEFER", "IR" }) |name| try t.expect(std.mem.indexOf(u8, text_dn5, name) != null);
    for (gu5_inputs ++ gu5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_gu5, std.mem.span(n)) != null);
    for (dn5_inputs ++ dn5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dn5, std.mem.span(n)) != null);
    for (red5_inputs ++ red5_outputs) |n| try t.expect(std.mem.indexOf(u8, text_red5, std.mem.span(n)) != null);
    for (dn6_inputs ++ dn6_outputs) |n| try t.expect(std.mem.indexOf(u8, text_dn6, std.mem.span(n)) != null);
}

