//! Decode-row merges at a gated, QK-normed attention layer's EXL3 linears (Qwen3.8-Flash-Next's 12 full-attention layers
//! and its MTP head). Rows 1..16 (the GEMV path; the served verify is 4 rows, the head's draft steps 1). Each entry point
//! gives the bytes of the chain it replaces, at every rounding point of that chain, with DEC-ONE (`Dense.decTakes`: one
//! launch per dense decode linear) taking the call or not:
//!
//! - `gatedDecode`: o_proj over T(attn * sigmoid(gate)) with the gate product inside o_proj's prepare
//!   (`text_prep_gate_sc`: the host's strided gate kernel's statement, then `dense.text_prep_s`'s statements in lane order,
//!   the block sums and DEC-ONE's counter zeroing, as `dense.text_prep_sc`), feeding DEC-ONE's `dec0` (or the chain's
//!   dgemv + dfinish when DEC-ONE does not take the call). The host's chain is its strided sigmoid-gate kernel then
//!   `Dense.linear`: one level less per layer.
//! - `mergeGatedDecode`: the same with the fused QSA core's split-K merge in that launch too (`text_merge_gate_sc`: the
//!   host's `msv_attn_qsa256_qmerge` statements, its T rounding, the gate product, the prepare): two levels less.
//! - `indexerQk`: the indexer projection (q heads | raw key, 128 wide each) with DEC-ONE's last-arriver finish (or the
//!   chain's dfinish) normalizing and rotating each query head (the host's hd-128 QK-norm+RoPE statements: MLX's
//!   rms_single_row order at axis 128, its rope's floats from the host's angle rows) and building the one pooled block key
//!   a call completes (the host's `QSA_POOL_ROPE_SOURCE` statements: the f32 block mean in row order, bf16, rms_norm,
//!   the bf16 cos / sin rotation), so the index-q copy / rms / copy / rope launches and the pooled-key copy / concat /
//!   kernel launches never run.
//! - `qPlanes`: q_proj with its finish writing each head's query and gate halves as two contiguous planes (the
//!   finish's values relaid out), so the hd-256 QK-norm+RoPE kernel's row-contiguous copy of the strided query view
//!   never runs.
//!
//! The value-setting statements are dense's / the host's, spliced or copied verbatim; the CPU tests below pin them.
const std = @import("std");
const Global = @import("../../core/global.zig").Global;
const mlx = @import("sdk").mlx;
const ks = @import("sdk").kernel_launch;
const decode = @import("../../core/decode.zig");
const dense = @import("../../core/dense.zig");
const sources = @import("../../core/sources.zig");

/// One linear of a merged call: the `Dense` `linear` uses for its shape (`denseFor`'s) and the host's arrays
pub const Member = struct { d: *dense.Dense, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };

/// Where o_proj's prepare reads the gate in place: element (row, head h, d) at base[row row_stride + h head_stride + off + d]
pub const Gate = struct { base: mlx.mlx_array, row_stride: u32, head_stride: u32, off: u32 };

/// dense.zig's 128-point Hadamard butterfly: the two statements every prep and finish there runs on v[4] (lane L holds
/// element L + 32 j of the block), verbatim; a CPU test below pins them inside text_prep_s and dfinish_v
const had128 = sources.had128;

// ------------------------------------------------------------------------- o_proj over the gated output: the producers

/// o_proj's prepare at decode rows over the gated attention output, the bytes of the host's strided gate kernel
/// (`y[i] = xv * sigtab[as_type<ushort>(gv)]`, T, its names kept) followed by `dense.text_prep_sc` (= text_prep_s + the
/// counter zeroing): yv = T(attn[h, row, d] * sigtab[gate(row, h, d)]) with i = h HD + d, then f16(yv) * suh, the butterfly, x HAD_SCALE, f16,
/// lane order (LANE_ORDER 1), the 128-block sums xs and DEC-ONE's arrival counters zeroed. attn T [1, H, rows, HD], the
/// gate at gate[row GROW + h GHEAD + GOFF + d]; one simdgroup per (block, row), prep_sc's launch shape.
pub const text_prep_gate_sc: [:0]const u8 = @embedFile("kernels/decb/prep_gate_sc.metal") ++ had128 ++ prep_sc_tail;

/// text_prep_s's lane-order store and block sum, then text_prep_sc's counter zeroing (their statements verbatim)
const prep_sc_tail = @embedFile("kernels/decb/prep_sc_tail.metal");

/// The qmerge form: the QSA core's split-K merge (the host's `msv_attn_qsa256_qmerge` text up to its store, verbatim),
/// its value rounded to T where the merge stores it, the gate product in T, staged; then simdgroups 0 and 1 run o_proj's
/// prepare (text_prep_s's statements) on the head's two 128-blocks of the row. Grid (256, rows, heads), one threadgroup
/// per (row, head) as the merge's; pacc f32 [heads, rows, NS, 256], pml f32 [heads, rows, NS, 2].
pub const text_merge_gate_sc: [:0]const u8 = merge_body ++
    @embedFile("kernels/decb/merge_gate_sc_1.metal") ++
    had128 ++
    @embedFile("kernels/decb/merge_gate_sc_2.metal");

/// The host's `QSA_ATTN_MERGE_SOURCE` (transformer.zig) without its final store line, verbatim (the host's CPU test pins
/// this against its own text)
pub const merge_body = @embedFile("kernels/decb/merge_body.metal");
/// The merge's store (the host's last line), whose value the qmerge form keeps: T(acc / Z)
pub const merge_store = @embedFile("kernels/decb/merge_store.metal");

// ------------------------------------------------------------------------------------------------ the indexer epilogue

/// The host's hd-128 QK-norm+RoPE statements (`QK_NORM_ROPE_SOURCE`, transformer.zig: MLX's rms_single_row at axis 128,
/// N_READS 4, one simdgroup; `rope_single_impl` / `rope_impl`'s rotation against the angle row) on one 128-wide index
/// query head: `src` its finished values (threadgroup), `arow` the row's cos | sin, `dst` its q_rope row. Names mapped:
/// lane -> L, HD -> IHD, eps -> qeps, w -> qn, angles -> arow, T -> OUT_T.
const idx_qrope = @embedFile("kernels/decb/idx_qrope.metal");

/// The host's pooled-key statements (`QSA_POOL_ROPE_SOURCE`, transformer.zig) for the one block a call completes: rows
/// r < ln of the block are the ring's leftover (`ring`, bf16 as the ring holds them), the rest the call's first rows
/// (`ist`, the finish's bf16 values); one simdgroup, lane owns dims [4 lane, 4 lane + 4). Names mapped: xs -> pxs, inv ->
/// pinv, eps -> peps, inv_r -> invr, norm_w -> kn, cosv / sinv row 0 -> pcos / psin, out -> op, R -> RATIO, d -> pd,
/// a / b -> pa / pb, o -> po; the read kb[kbase + r * 128 + j] -> the block's row r.
const idx_pool = @embedFile("kernels/decb/idx_pool.metal");

/// The indexer epilogue's threadgroup staging in DEC-ONE's launch: each finished row's 128-block (the chunk's rows,
/// rr = sg)
const idx_dec_decl = @embedFile("kernels/decb/idx_dec_decl.metal");

/// The indexer epilogue in DEC-ONE's last-arriver finish: dense.dec_finish_v (the planes summed in slice order, the
/// deferred map, f16, the butterfly), dec_finish_store's value (x HAD_SCALE x svh, f16, OUT_T) kept and stored, then the
/// query-head epilogue (blocks < NQB) on the row; after the row loop the key block's chunk-0 last arriver builds the
/// pooled key from the rows its simdgroups staged
const idx_dec_finish = dense.dec_finish_v ++
    @embedFile("kernels/decb/idx_dec_finish_1.metal") ++
    idx_qrope ++
    @embedFile("kernels/decb/idx_dec_finish_2.metal") ++
    idx_pool ++
    @embedFile("kernels/decb/idx_dec_finish_3.metal");

/// The indexer epilogue on the chain's finish (DEC-ONE off): dense.dfinish_v, dfinish_store's value kept and stored, the
/// query-head epilogue on the (block, row) simdgroup, and the pooled key in the key block's row-0 simdgroup, which
/// finishes the block's rows 1.. itself (dfinish_v's statements on row rq: the same values the other simdgroups store)
const idx_dfin_tail = @embedFile("kernels/decb/idx_dfin_tail_1.metal") ++
    idx_qrope ++
    @embedFile("kernels/decb/idx_dfin_tail_2.metal") ++
    had128 ++
    @embedFile("kernels/decb/idx_dfin_tail_3.metal") ++
    idx_pool ++
    @embedFile("kernels/decb/idx_dfin_tail_4.metal");

/// The indexer projection over prep_sc's rows (DEC-ONE, prepare in its own launch): y, z, oq, op
pub const text_dec0_idx: [:0]const u8 = dense.dec_head ++ idx_dec_decl ++ dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ idx_dec_finish;
/// The indexer projection with DEC-ONE's prepare inside: y, z, xsx, oq, op
pub const text_dec1_idx: [:0]const u8 = dense.dec_head ++ idx_dec_decl ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive ++ dense.dec_xsum1 ++ idx_dec_finish;
/// The indexer projection on the chain's finish: y, oq, op
pub const text_dfinish_idx: [:0]const u8 = dense.dfinish_v ++ idx_dfin_tail;

// ------------------------------------------------------------------------------------------------- the q / gate planes

/// The planes' store: dec_finish_store's value written to its head's query plane (columns [0, PH) of the head's PW) or
/// gate plane, [rows, NHP, PH] each, instead of y [rows, OUT]
const planes_store_dec = @embedFile("kernels/decb/planes_store_dec.metal");
const planes_store_dfin = @embedFile("kernels/decb/planes_store_dfin.metal");

/// The planes over prep_sc's rows (DEC-ONE, prepare in its own launch): yq, yg, z
pub const text_dec0_planes: [:0]const u8 = dense.dec_head ++ dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ dense.dec_finish_v ++ planes_store_dec;
/// The planes with DEC-ONE's prepare inside: yq, yg, z, xsx
pub const text_dec1_planes: [:0]const u8 = dense.dec_head ++ dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive ++ dense.dec_xsum1 ++ dense.dec_finish_v ++ planes_store_dec;
/// The planes on the chain's finish: yq, yg
pub const text_dfinish_planes: [:0]const u8 = dense.dfinish_v ++ planes_store_dfin;

// ------------------------------------------------------------------------------------------------- kernels and configs

const prep_gate_sc_in = [_][*:0]const u8{ "attn", "gate", "sigtab", "suh" };
const merge_gate_in = [_][*:0]const u8{ "pacc", "pml", "gate", "sigtab", "suh" };
const prep_sc_out = [_][*:0]const u8{ "xp", "xs", "cnt" };
const dec0_planes_in = [_][*:0]const u8{ "xp", "xs", "w", "svh", "cnt" };
const dec0_planes_out = [_][*:0]const u8{ "yq", "yg", "z" };
const dec1_planes_in = [_][*:0]const u8{ "x", "suh", "w", "svh", "cnt" };
const dec1_planes_out = [_][*:0]const u8{ "yq", "yg", "z", "xsx" };
const dfin_planes_in = [_][*:0]const u8{ "z", "svh", "xs" };
const dfin_planes_out = [_][*:0]const u8{ "yq", "yg" };
const idx_extra_in = [_][*:0]const u8{ "qn", "qeps", "angles", "kn", "ring", "pcos", "psin", "peps", "invr", "ln" };
const dec0_idx_in = dec0_planes_in ++ idx_extra_in;
const dec0_idx_out = [_][*:0]const u8{ "y", "z", "oq", "op" };
const dec1_idx_in = dec1_planes_in ++ idx_extra_in;
const dec1_idx_out = [_][*:0]const u8{ "y", "z", "xsx", "oq", "op" };
const dfin_idx_in = dfin_planes_in ++ idx_extra_in;
const dfin_idx_out = [_][*:0]const u8{ "y", "oq", "op" };

const Kernels = struct {
    prep_gate: mlx.mlx_fast_metal_kernel,
    merge_gate: mlx.mlx_fast_metal_kernel,
    dec0_planes: mlx.mlx_fast_metal_kernel,
    dec1_planes: mlx.mlx_fast_metal_kernel,
    dfin_planes: mlx.mlx_fast_metal_kernel,
    dec0_idx: mlx.mlx_fast_metal_kernel,
    dec1_idx: mlx.mlx_fast_metal_kernel,
    dfin_idx: mlx.mlx_fast_metal_kernel,
};

const kernels_lock = Global(std.atomic.Value(u32), "decb_kernels_lock", std.atomic.Value(u32).init(0));
const kernels = Global(?Kernels, "decb_kernels", null);

fn lock() void {
    while (kernels_lock.get().*.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}
fn unlock() void {
    kernels_lock.get().*.store(0, .release);
}

fn kernelsGet() !*const Kernels {
    lock();
    defer unlock();
    if (kernels.get().*) |*k| return k;
    // the DEC-ONE variants take DEC-ONE's header (header5: the coherent partial-plane pointers), the others dense's
    const pg = ks.newKernel("turboexl3_decb_prep_gate", &prep_gate_sc_in, &prep_sc_out, text_prep_gate_sc, decode.header, true) orelse return error.KernelCreateFailed;
    const mg = ks.newKernel("turboexl3_decb_merge_gate", &merge_gate_in, &prep_sc_out, text_merge_gate_sc, decode.header, true) orelse return error.KernelCreateFailed;
    const d0p = ks.newKernel("turboexl3_decb_dec0_planes", &dec0_planes_in, &dec0_planes_out, text_dec0_planes, decode.header5, true) orelse return error.KernelCreateFailed;
    const d1p = ks.newKernel("turboexl3_decb_dec1_planes", &dec1_planes_in, &dec1_planes_out, text_dec1_planes, decode.header5, true) orelse return error.KernelCreateFailed;
    const dfp = ks.newKernel("turboexl3_decb_dfinish_planes", &dfin_planes_in, &dfin_planes_out, text_dfinish_planes, decode.header, true) orelse return error.KernelCreateFailed;
    const d0i = ks.newKernel("turboexl3_decb_dec0_idx", &dec0_idx_in, &dec0_idx_out, text_dec0_idx, decode.header5, true) orelse return error.KernelCreateFailed;
    const d1i = ks.newKernel("turboexl3_decb_dec1_idx", &dec1_idx_in, &dec1_idx_out, text_dec1_idx, decode.header5, true) orelse return error.KernelCreateFailed;
    const dfi = ks.newKernel("turboexl3_decb_dfinish_idx", &dfin_idx_in, &dfin_idx_out, text_dfinish_idx, decode.header, true) orelse return error.KernelCreateFailed;
    kernels.get().* = .{ .prep_gate = pg, .merge_gate = mg, .dec0_planes = d0p, .dec1_planes = d1p, .dfin_planes = dfp, .dec0_idx = d0i, .dec1_idx = d1i, .dfin_idx = dfi };
    return &kernels.get().*.?;
}

/// What a cached launch config serves: the Dense's geometry and template knobs (`denseTag`: a config bakes them in, and a
/// Dense freed and remade at the same address may differ), the row count, the launch and the per-call template values
const Kind = enum { dec0, prep_gate, merge_gate, planes_dec, planes_dfin, idx_dec, idx_dfin };
const CfgKey = struct {
    d: u64,
    rows: u32,
    kind: Kind,
    /// prep_gate / merge_gate: the gate layout; idx: QEPI, POOL, heads, rope dims, ratio; planes: heads, hd
    a: u32 = 0,
    b: u32 = 0,
    c: u32 = 0,
    e: u32 = 0,
    f: u32 = 0,
    inside: bool = false,
};
const CfgSlot = struct { key: CfgKey, cfg: mlx.mlx_fast_metal_kernel_config };
const cfg_cache = Global([512]?CfgSlot, "decb_cfg_cache", @splat(null));

fn cfgFind(key: CfgKey) ?mlx.mlx_fast_metal_kernel_config {
    lock();
    defer unlock();
    for (cfg_cache.get().*) |slot| if (slot) |sl| {
        if (std.meta.eql(sl.key, key)) return sl.cfg;
    };
    return null;
}

/// Caches `cfg` under `key` (a full cache frees nothing: the caller frees its own config after the call)
fn cfgPut(key: CfgKey, cfg: mlx.mlx_fast_metal_kernel_config) bool {
    lock();
    defer unlock();
    for (&cfg_cache.get().*) |*slot| if (slot.* == null) {
        slot.* = .{ .key = key, .cfg = cfg };
        return true;
    };
    return false;
}

/// A config from the cache or `build`'s; `owned` tells the caller to free it after its launch (the cache was full)
const Got = struct { cfg: mlx.mlx_fast_metal_kernel_config, owned: bool };
fn cfgGet(key: CfgKey, built: ks.Launch) !Got {
    if (cfgFind(key)) |c| return .{ .cfg = c, .owned = false };
    const c = try ks.buildConfig(built);
    if (cfgPut(key, c)) return .{ .cfg = c, .owned = false };
    return .{ .cfg = c, .owned = true };
}
fn cfgDone(g: Got) void {
    if (g.owned) _ = mlx.mlx_fast_metal_kernel_config_free(g.cfg);
}

fn ci(v: anytype) c_int {
    return @intCast(v);
}

/// Everything of a Dense a launch config bakes in: its shape (rate, codebook, window, output dtype), its K split and the
/// deferred MUL1 map (DEC-ONE's row tiling is the same for every shape)
fn denseTag(d: *const dense.Dense) u64 {
    var h = std.hash.Wyhash.init(0x4DEC8);
    const dt: u64 = switch (d.shape.out_dtype) {
        .bfloat16 => 1,
        .float16 => 2,
        .float32 => 3,
        else => 4,
    };
    const vals = [_]u64{ d.shape.in, d.shape.out, d.shape.rate.n, @backingInt(d.shape.dec.codebook), d.shape.dec.window.bits(), dt, d.ks_split, @intFromBool(d.defer_affine) };
    h.update(std.mem.asBytes(&vals));
    return h.final();
}

/// DEC-ONE's row tiling for `rows` (Dense.decCfgs's arithmetic): rows per chunk, output tiles per threadgroup, chunks.
/// Cached per row count.
const Tiling = struct { rb: u32, bt: u32, chunks: u32 };
const tiling_cache = Global([decode.Kernels.max_rows + 1]?Tiling, "decb_tiling_cache", @splat(null));
fn tiling(rows: usize) !Tiling {
    if (rows < 1 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
    lock();
    defer unlock();
    if (tiling_cache.get().*[rows]) |t| return t;
    const rows_u: u32 = @intCast(rows);
    const rb: u32 = dense.Dense.rbFor(rows_u);
    const bt = dense.Dense.btFor(rb);
    if (8 % bt != 0) return error.BadExl3Shape;
    const t: Tiling = .{ .rb = rb, .bt = bt, .chunks = (rows_u + rb - 1) / rb };
    tiling_cache.get().*[rows] = t;
    return t;
}

/// Whether the indexer epilogue's staging (RB rows of one bf16 128-block) still fits beside DEC-ONE's own threadgroup
/// memory with the prepare inside (Dense.decPrepFits's budget plus `ist`)
fn idxInsideFits(d: *const dense.Dense, t: Tiling) bool {
    const per = (d.shape.in / 16) / (d.ks_split * 4);
    const span = 64 * per;
    return t.rb * span * 2 + 4 * t.rb * 16 * t.bt * 4 + 16 + t.rb * 128 * 2 <= 32768;
}

/// DEC-ONE's arrival-counter count at `rows` (Dense.decCfgs's: one per (output block, row chunk), at least 8)
pub fn decCounters(d: *const dense.Dense, rows: usize) !u32 {
    const t = try tiling(rows);
    return @max((d.shape.out / 128) * t.chunks, 8);
}

/// DEC-ONE's GEMV launch at `rows` (Dense.decCfgs's grid, threadgroup and template arguments) with `extra` template
/// arguments and the variant's outputs
fn decLaunch(d: *dense.Dense, rows: usize, extra: []const ks.TemplateArg, tmpl_buf: []ks.TemplateArg, out_shapes: []const []const c_int, out_dtypes: []const mlx.mlx_dtype) !ks.Launch {
    const t = try tiling(rows);
    const tp = d.templ();
    const base = [_]ks.TemplateArg{
        tp[0], tp[1], tp[2],
        .{ .name = "IN", .value = .{ .int = ci(d.shape.in) } },
        .{ .name = "OUT", .value = .{ .int = ci(d.shape.out) } },
        .{ .name = "KS", .value = .{ .int = ci(d.ks_split) } },
        .{ .name = "BT", .value = .{ .int = ci(t.bt) } },
        .{ .name = "RB", .value = .{ .int = ci(t.rb) } },
        .{ .name = "ROWS", .value = .{ .int = ci(rows) } },
        .{ .name = "DEFER", .value = .{ .int = if (d.defer_affine) 1 else 0 } },
        .{ .name = "OUT_T", .value = .{ .dtype = d.shape.out_dtype } },
    };
    if (base.len + extra.len > tmpl_buf.len) return error.BadExl3Shape;
    @memcpy(tmpl_buf[0..base.len], &base);
    @memcpy(tmpl_buf[base.len..][0..extra.len], extra);
    return .{
        .grid = .{ (d.shape.out / 16 / t.bt) * 128, t.chunks, d.ks_split },
        .threadgroup = .{ 128, 1, 1 },
        .out_shapes = out_shapes,
        .out_dtypes = out_dtypes,
        .template = tmpl_buf[0 .. base.len + extra.len],
    };
}

/// The chain's finish launch at `rows` (Dense.buildMultiCfg's finish: grid, threadgroup, template) with `extra` template
/// arguments and the variant's outputs
fn dfinLaunch(d: *const dense.Dense, rows: usize, extra: []const ks.TemplateArg, tmpl_buf: []ks.TemplateArg, out_shapes: []const []const c_int, out_dtypes: []const mlx.mlx_dtype) !ks.Launch {
    const base = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = ci(d.shape.out) } },
        .{ .name = "OUT_T", .value = .{ .dtype = d.shape.out_dtype } },
        .{ .name = "KS", .value = .{ .int = ci(d.ks_split) } },
        .{ .name = "ROWS", .value = .{ .int = ci(rows) } },
        .{ .name = "INBLK", .value = .{ .int = ci(d.shape.in / 128) } },
        .{ .name = "DEFER", .value = .{ .int = if (d.defer_affine) 1 else 0 } },
    };
    if (base.len + extra.len > tmpl_buf.len) return error.BadExl3Shape;
    @memcpy(tmpl_buf[0..base.len], &base);
    @memcpy(tmpl_buf[base.len..][0..extra.len], extra);
    return .{
        .grid = .{ (d.shape.out / 128) * 32, @intCast(rows), 1 },
        .threadgroup = .{ 32, 1, 1 },
        .out_shapes = out_shapes,
        .out_dtypes = out_dtypes,
        .template = tmpl_buf[0 .. base.len + extra.len],
    };
}

/// The size of DEC-ONE's xsx scratch with the prepare inside (Dense.decCfgs's)
fn xsxCount(d: *const dense.Dense, rows: usize) !u32 {
    const t = try tiling(rows);
    return if (d.defer_affine) @max((d.shape.out / 128) * t.chunks * t.rb * (d.shape.in / 128), 8) else 8;
}

fn freeAll(xs: []const mlx.mlx_array) void {
    for (xs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

// -------------------------------------------------------------- o_proj over the gated output: the consumer and entries

/// `linear`'s inner product and finish on rows a producer prepared as prep_sc prepares them (xp half [rows, in] in lane
/// order, xs f32 [rows, in / 128], cnt u32 zeroed): DEC-ONE's dec0 when `dec_on`, else the chain's dgemv + dfinish (the
/// configs `linear` uses at this row count). y [rows, out] in `shape.out_dtype`.
fn preparedRows(d: *dense.Dense, s: mlx.mlx_stream, xp: mlx.mlx_array, xs: mlx.mlx_array, cnt: mlx.mlx_array, trellis: mlx.mlx_array, svh: mlx.mlx_array, rows: usize, dec_on: bool) !mlx.mlx_array {
    if (dec_on) {
        const r = ci(rows);
        const OUT = ci(d.shape.out);
        var tb: [16]ks.TemplateArg = undefined;
        const y_shape = [_]c_int{ r, OUT };
        const z_shape = [_]c_int{ ci(d.ks_split) * r, OUT };
        const shapes = [_][]const c_int{ &y_shape, &z_shape };
        const dtypes = [_]mlx.mlx_dtype{ d.shape.out_dtype, .float32 };
        const no_extra = [_]ks.TemplateArg{};
        const launch = try decLaunch(d, rows, &no_extra, &tb, &shapes, &dtypes);
        const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .dec0 }, launch);
        defer cfgDone(got);
        const ins = [_]mlx.mlx_array{ xp, xs, trellis, svh, cnt };
        var o: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(d.dec0, got.cfg, &ins, &o, s);
        _ = mlx.mlx_array_free(o[1]);
        return o[0];
    }
    const c = try d.cfgs(rows);
    var z: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(if (c.lean) d.dgemv_lean else d.dgemv, c.gemv, &.{ xp, trellis }, &z, s);
    defer _ = mlx.mlx_array_free(z[0]);
    var y: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(d.dfinish, c.finish, &.{ z[0], svh, xs }, &y, s);
    return y[0];
}

/// Refusals both gated entries share: rows on the GEMV path, the row-slot kernels off, the gate's layout in bounds
fn gatedCheck(d: *const dense.Dense, gate: Gate, heads: u32, hd: u32, rows: usize) !void {
    if (rows < 1 or rows > decode.Kernels.max_rows or d.rowslot) return error.Exl3EngineNotImplemented;
    if (d.shape.in != heads * hd or hd % 128 != 0 or heads == 0) return error.BadExl3Shape;
    if (gate.off + (heads - 1) * gate.head_stride + hd > gate.row_stride or mlx.mlx_array_size(gate.base) != rows * gate.row_stride) return error.BadExl3Shape;
}

/// y [rows, out] = o_proj(T(attn * sigmoid(gate))) at decode rows for attn T [1, heads, rows, hd] (head-major, the
/// attention core's output), the gate read in place, `sigtab` T [65536] (MLX's sigmoid of every 16-bit pattern): the
/// bytes of the host's strided gate kernel + `Dense.linear` (DEC-ONE's or the chain's) on its product.
pub fn gatedDecode(s: mlx.mlx_stream, attn: mlx.mlx_array, gate: Gate, sigtab: mlx.mlx_array, o: Member, heads: u32, hd: u32, rows: usize) !mlx.mlx_array {
    const d = o.d;
    try gatedCheck(d, gate, heads, hd, rows);
    const kz = try kernelsGet();
    const dec_on = d.decTakes(rows);
    const nb = d.shape.in / 128;
    const ncnt: u32 = if (dec_on) try decCounters(d, rows) else 8;
    const nth: u32 = nb * 32 * @as(u32, @intCast(rows));
    const r = ci(rows);
    const xp_shape = [_]c_int{ r, ci(d.shape.in) };
    const xs_shape = [_]c_int{ r, ci(nb) };
    const cnt_shape = [_]c_int{ci(ncnt)};
    const shapes = [_][]const c_int{ &xp_shape, &xs_shape, &cnt_shape };
    const dtypes = [_]mlx.mlx_dtype{ .float16, .float32, .uint32 };
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = ci(d.shape.in) } },
        .{ .name = "HD", .value = .{ .int = ci(hd) } },
        .{ .name = "GROW", .value = .{ .int = ci(gate.row_stride) } },
        .{ .name = "GHEAD", .value = .{ .int = ci(gate.head_stride) } },
        .{ .name = "GOFF", .value = .{ .int = ci(gate.off) } },
        .{ .name = "T", .value = .{ .dtype = d.shape.out_dtype } },
        .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
        .{ .name = "NCNT", .value = .{ .int = ci(ncnt) } },
        .{ .name = "NTH", .value = .{ .int = ci(nth) } },
    };
    const launch: ks.Launch = .{ .grid = .{ nb * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &shapes, .out_dtypes = &dtypes, .template = &tmpl };
    const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .prep_gate, .a = gate.row_stride, .b = gate.head_stride, .c = gate.off, .e = hd, .f = ncnt }, launch);
    defer cfgDone(got);
    const ins = [_]mlx.mlx_array{ attn, gate.base, sigtab, o.suh };
    var po: [3]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.prep_gate, got.cfg, &ins, &po, s);
    defer freeAll(&po);
    return preparedRows(d, s, po[0], po[1], po[2], o.trellis, o.svh, rows, dec_on);
}

/// `gatedDecode`'s qmerge form: y [rows, out] = o_proj(T(merge(pacc, pml) * sigmoid(gate))) for the fused QSA core's
/// split-K partials (pacc f32 [heads, rows, NS, 256], pml f32 [heads, rows, NS, 2]): the bytes of the host's merge launch,
/// its strided gate kernel and `Dense.linear`.
pub fn mergeGatedDecode(s: mlx.mlx_stream, pacc: mlx.mlx_array, pml: mlx.mlx_array, gate: Gate, sigtab: mlx.mlx_array, o: Member, heads: u32, rows: usize) !mlx.mlx_array {
    const d = o.d;
    const hd: u32 = 256; // the merge's BD
    try gatedCheck(d, gate, heads, hd, rows);
    const psh = mlx.getShape(pacc);
    const msh = mlx.getShape(pml);
    if (psh.len != 4 or msh.len != 4 or psh[0] != ci(heads) or psh[1] != ci(rows) or psh[3] != 256 or msh[0] != psh[0] or msh[1] != psh[1] or msh[2] != psh[2] or msh[3] != 2) return error.BadExl3Shape;
    if (psh[2] < 1 or psh[2] > 64 or mlx.mlx_array_dtype(pacc) != .float32 or mlx.mlx_array_dtype(pml) != .float32) return error.Exl3EngineNotImplemented;
    const kz = try kernelsGet();
    const dec_on = d.decTakes(rows);
    const nb = d.shape.in / 128;
    const ncnt: u32 = if (dec_on) try decCounters(d, rows) else 8;
    const nth: u32 = 256 * @as(u32, @intCast(rows)) * heads;
    const r = ci(rows);
    const xp_shape = [_]c_int{ r, ci(d.shape.in) };
    const xs_shape = [_]c_int{ r, ci(nb) };
    const cnt_shape = [_]c_int{ci(ncnt)};
    const shapes = [_][]const c_int{ &xp_shape, &xs_shape, &cnt_shape };
    const dtypes = [_]mlx.mlx_dtype{ .float16, .float32, .uint32 };
    const tmpl = [_]ks.TemplateArg{
        .{ .name = "DIM", .value = .{ .int = ci(d.shape.in) } },
        .{ .name = "GROW", .value = .{ .int = ci(gate.row_stride) } },
        .{ .name = "GHEAD", .value = .{ .int = ci(gate.head_stride) } },
        .{ .name = "GOFF", .value = .{ .int = ci(gate.off) } },
        .{ .name = "T", .value = .{ .dtype = d.shape.out_dtype } },
        .{ .name = "LANE_ORDER", .value = .{ .int = 1 } },
        .{ .name = "NCNT", .value = .{ .int = ci(ncnt) } },
        .{ .name = "NTH", .value = .{ .int = ci(nth) } },
    };
    const launch: ks.Launch = .{ .grid = .{ 256, @intCast(rows), heads }, .threadgroup = .{ 256, 1, 1 }, .out_shapes = &shapes, .out_dtypes = &dtypes, .template = &tmpl };
    const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .merge_gate, .a = gate.row_stride, .b = gate.head_stride, .c = gate.off, .e = heads, .f = ncnt }, launch);
    defer cfgDone(got);
    const ins = [_]mlx.mlx_array{ pacc, pml, gate.base, sigtab, o.suh };
    var po: [3]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.merge_gate, got.cfg, &ins, &po, s);
    defer freeAll(&po);
    return preparedRows(d, s, po[0], po[1], po[2], o.trellis, o.svh, rows, dec_on);
}

// ------------------------------------------------------------------------------------------ the q / gate planes: entry

/// q_proj (x2 [rows, in]) with each head's [query | gate] columns written as two planes [rows, heads, hd] each
/// (returned as [rows, heads * hd]): the finish's values of `Dense.linear` (DEC-ONE's or the chain's), relaid out.
pub fn qPlanes(s: mlx.mlx_stream, x2: mlx.mlx_array, rows: usize, q: Member, heads: u32, hd: u32) ![2]mlx.mlx_array {
    const d = q.d;
    if (rows < 1 or rows > decode.Kernels.max_rows or d.rowslot) return error.Exl3EngineNotImplemented;
    if (heads == 0 or hd % 128 != 0 or d.shape.out != 2 * heads * hd) return error.BadExl3Shape;
    const kz = try kernelsGet();
    const r = ci(rows);
    const plane = [_]c_int{ r, ci(heads * hd) };
    const extra = [_]ks.TemplateArg{
        .{ .name = "PW", .value = .{ .int = ci(2 * hd) } },
        .{ .name = "PH", .value = .{ .int = ci(hd) } },
        .{ .name = "NHP", .value = .{ .int = ci(heads) } },
    };
    const odt = d.shape.out_dtype;
    var tb: [20]ks.TemplateArg = undefined;
    if (d.decTakes(rows)) {
        const c = try d.decCfgs(rows);
        const z_shape = [_]c_int{ ci(d.ks_split) * r, ci(d.shape.out) };
        if (c.prep_in) {
            const nx = try xsxCount(d, rows);
            const x_shape = [_]c_int{ci(nx)};
            const shapes = [_][]const c_int{ &plane, &plane, &z_shape, &x_shape };
            const dtypes = [_]mlx.mlx_dtype{ odt, odt, .float32, .float32 };
            const launch = try decLaunch(d, rows, &extra, &tb, &shapes, &dtypes);
            const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .planes_dec, .a = heads, .b = hd, .inside = true }, launch);
            defer cfgDone(got);
            const zin = [_]mlx.mlx_array{q.svh};
            var cnt: [1]mlx.mlx_array = undefined;
            try ks.applyConfig(d.dec_zero, c.prep, &zin, &cnt, s);
            defer _ = mlx.mlx_array_free(cnt[0]);
            const ins = [_]mlx.mlx_array{ x2, q.suh, q.trellis, q.svh, cnt[0] };
            var o: [4]mlx.mlx_array = undefined;
            try ks.applyConfig(kz.dec1_planes, got.cfg, &ins, &o, s);
            freeAll(o[2..4]);
            return .{ o[0], o[1] };
        }
        const shapes = [_][]const c_int{ &plane, &plane, &z_shape };
        const dtypes = [_]mlx.mlx_dtype{ odt, odt, .float32 };
        const launch = try decLaunch(d, rows, &extra, &tb, &shapes, &dtypes);
        const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .planes_dec, .a = heads, .b = hd, .inside = false }, launch);
        defer cfgDone(got);
        const pin = [_]mlx.mlx_array{ x2, q.suh };
        var po: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(d.prep_sc, c.prep, &pin, &po, s);
        defer freeAll(&po);
        const ins = [_]mlx.mlx_array{ po[0], po[1], q.trellis, q.svh, po[2] };
        var o: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(kz.dec0_planes, got.cfg, &ins, &o, s);
        _ = mlx.mlx_array_free(o[2]);
        return .{ o[0], o[1] };
    }
    const c = try d.cfgs(rows);
    const pin = [_]mlx.mlx_array{ x2, q.suh };
    var po: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(d.prep_s, c.prep, &pin, &po, s);
    defer freeAll(&po);
    const gin = [_]mlx.mlx_array{ po[0], q.trellis };
    var z: [1]mlx.mlx_array = undefined;
    try ks.applyConfig(if (c.lean) d.dgemv_lean else d.dgemv, c.gemv, &gin, &z, s);
    defer _ = mlx.mlx_array_free(z[0]);
    const shapes = [_][]const c_int{ &plane, &plane };
    const dtypes = [_]mlx.mlx_dtype{ odt, odt };
    const launch = try dfinLaunch(d, rows, &extra, &tb, &shapes, &dtypes);
    const got = try cfgGet(.{ .d = denseTag(d), .rows = @intCast(rows), .kind = .planes_dfin, .a = heads, .b = hd }, launch);
    defer cfgDone(got);
    const fin = [_]mlx.mlx_array{ z[0], q.svh, po[1] };
    var o: [2]mlx.mlx_array = undefined;
    try ks.applyConfig(kz.dfin_planes, got.cfg, &fin, &o, s);
    return o;
}

// -------------------------------------------------------------------------------------------------- the indexer: entry

/// The indexer projection's epilogue: its first `heads` 128-blocks are its query heads, the next its raw key.
pub const IdxEpilogue = struct {
    /// index query heads (each one 128-block of the projection; the key block follows)
    heads: u32,
    /// the rotated leading dims of each 128-wide head (a multiple of 8, at most 128)
    rope_dims: u32,
    /// the block ratio (rows per pooled key, at most 8: the f32 block sum's MLX order)
    ratio: u32,
    /// normalize and rotate the query heads (`q_rope` out)
    qepi: bool,
    /// bf16 [128]: the index query norm's weight (MLX rms_norm's)
    q_norm: mlx.mlx_array,
    /// f32 [rows * rope_dims]: row p holds cos then sin at the call's position p (the host's `ropeAngleRows`); any f32
    /// array of at least 8 elements when !qepi
    angles: mlx.mlx_array,
    /// build the one pooled key block this call completes (`pooled` out)
    pool: bool,
    /// bf16 [128]: the pooled key norm's weight
    k_norm: mlx.mlx_array,
    /// bf16 [1, ln, 128]: the ring's leftover rows of the block (any bf16 array of at least 8 elements when ln == 0)
    ring: mlx.mlx_array,
    /// the block's rows that come from the ring (< ratio; the block's other ratio - ln rows are the call's first rows)
    ln: u32,
    /// bf16 [1, rope_dims]: the block's cos / sin row (the host's `qsaPooledCosSin`)
    pcos: mlx.mlx_array,
    psin: mlx.mlx_array,
    /// the norms' eps (both the index query's and the pooled key's: the config's rms_norm_eps)
    eps: f32,
};

/// `indexerQk`'s results (each the caller's): qk [rows, (heads + 1) 128] as `linear` gives it, q_rope [1, heads, rows,
/// 128] (null-ctx without `qepi`), pooled [1, 1, 128] (null-ctx without `pool`)
pub const IdxOut = struct {
    qk: mlx.mlx_array,
    q_rope: mlx.mlx_array = .{ .ctx = null },
    pooled: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *IdxOut) void {
        inline for (.{ &self.qk, &self.q_rope, &self.pooled }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
        }
    }
};

/// An f32 [8] array (MLX binds an input under 8 elements in the constant address space): the angle input when !qepi
const dummy_f32 = Global(?mlx.mlx_array, "decb_dummy_f32", null);
fn dummyF32() !mlx.mlx_array {
    lock();
    defer unlock();
    if (dummy_f32.get().*) |a| return a;
    const zeros: [8]f32 = @splat(0);
    const a = mlx.mlx_array_new_data(&zeros, &[_]c_int{8}, 1, .float32);
    if (a.ctx == null) return error.MlxError;
    try mlx.check(mlx.mlx_array_eval(a));
    dummy_f32.get().* = a;
    return a;
}

/// The indexer projection of x2 [rows, in] with its finish's epilogue (see the file's doc). Refuses
/// (error.Exl3EngineNotImplemented) what it does not serve, before any launch.
pub fn indexerQk(s: mlx.mlx_stream, x2: mlx.mlx_array, rows: usize, idx: Member, ep: IdxEpilogue) !IdxOut {
    const d = idx.d;
    if (rows < 1 or rows > decode.Kernels.max_rows or d.rowslot) return error.Exl3EngineNotImplemented;
    if (!ep.qepi and !ep.pool) return error.Exl3EngineNotImplemented;
    if (ep.heads == 0 or d.shape.out != (ep.heads + 1) * 128) return error.BadExl3Shape;
    if (d.shape.out_dtype != .bfloat16) return error.Exl3EngineNotImplemented;
    if (ep.rope_dims == 0 or ep.rope_dims > 128 or ep.rope_dims % 8 != 0) return error.Exl3EngineNotImplemented;
    if (ep.ratio < 1 or ep.ratio > 8 or ep.ln >= ep.ratio) return error.Exl3EngineNotImplemented;
    if (ep.pool and ep.ratio - ep.ln > rows) return error.Exl3EngineNotImplemented;
    for ([_]mlx.mlx_array{ ep.q_norm, ep.k_norm }) |w| {
        if (mlx.mlx_array_dtype(w) != .bfloat16 or mlx.mlx_array_size(w) != 128) return error.Exl3EngineNotImplemented;
    }
    if (ep.qepi and (mlx.mlx_array_dtype(ep.angles) != .float32 or mlx.mlx_array_size(ep.angles) != rows * ep.rope_dims)) return error.Exl3EngineNotImplemented;
    if (ep.pool) {
        if (mlx.mlx_array_dtype(ep.pcos) != .bfloat16 or mlx.mlx_array_dtype(ep.psin) != .bfloat16 or mlx.mlx_array_size(ep.pcos) != ep.rope_dims or mlx.mlx_array_size(ep.psin) != ep.rope_dims) return error.Exl3EngineNotImplemented;
        if (ep.ln > 0 and (mlx.mlx_array_dtype(ep.ring) != .bfloat16 or mlx.mlx_array_size(ep.ring) != ep.ln * 128)) return error.Exl3EngineNotImplemented;
    }
    const dec_on = d.decTakes(rows);
    if (dec_on and ep.pool) {
        // the pooled key reads the rows the key block's chunk-0 last arriver finished
        const t = try tiling(rows);
        if (ep.ratio - ep.ln > t.rb) return error.Exl3EngineNotImplemented;
    }
    const kz = try kernelsGet();
    const af = if (ep.qepi) ep.angles else try dummyF32();
    // the rope tables and the ring only where read (bf16 stand-ins otherwise: the norm weights, 128 elements)
    const ring = if (ep.pool and ep.ln > 0) ep.ring else ep.k_norm;
    const pcos = if (ep.pool) ep.pcos else ep.k_norm;
    const psin = if (ep.pool) ep.psin else ep.k_norm;
    const qeps = mlx.mlx_array_new_float(ep.eps);
    defer _ = mlx.mlx_array_free(qeps);
    const peps = mlx.mlx_array_new_float(ep.eps);
    defer _ = mlx.mlx_array_free(peps);
    // MLX's mean multiplies by NumberOfElements: f32(1.0 / n) computed in f64 (as the host's pooled-key kernel)
    const invr = mlx.mlx_array_new_float(@floatCast(1.0 / @as(f64, @floatFromInt(ep.ratio))));
    defer _ = mlx.mlx_array_free(invr);
    const lna = mlx.mlx_array_new_int(@intCast(ep.ln));
    defer _ = mlx.mlx_array_free(lna);
    const extra_in = [_]mlx.mlx_array{ ep.q_norm, qeps, af, ep.k_norm, ring, pcos, psin, peps, invr, lna };
    const extra_t = [_]ks.TemplateArg{
        .{ .name = "NQB", .value = .{ .int = ci(ep.heads) } },
        .{ .name = "QEPI", .value = .{ .int = if (ep.qepi) 1 else 0 } },
        .{ .name = "POOL", .value = .{ .int = if (ep.pool) 1 else 0 } },
        .{ .name = "RD", .value = .{ .int = ci(ep.rope_dims) } },
        .{ .name = "HALF", .value = .{ .int = ci(ep.rope_dims / 2) } },
        .{ .name = "RATIO", .value = .{ .int = ci(ep.ratio) } },
    };
    const r = ci(rows);
    const y_shape = [_]c_int{ r, ci(d.shape.out) };
    // unread outputs keep a minimal buffer
    const oq_full = [_]c_int{ 1, ci(ep.heads), r, 128 };
    const op_full = [_]c_int{ 1, 1, 128 };
    const minimal = [_]c_int{8};
    const oq_shape: []const c_int = if (ep.qepi) &oq_full else &minimal;
    const op_shape: []const c_int = if (ep.pool) &op_full else &minimal;
    const bf: mlx.mlx_dtype = .bfloat16;
    var tb: [24]ks.TemplateArg = undefined;
    const key: CfgKey = .{ .d = denseTag(d), .rows = @intCast(rows), .kind = if (dec_on) .idx_dec else .idx_dfin, .a = @intFromBool(ep.qepi), .b = @intFromBool(ep.pool), .c = ep.heads, .e = ep.rope_dims, .f = ep.ratio };
    var out: IdxOut = .{ .qk = .{ .ctx = null } };
    if (dec_on) {
        const c = try d.decCfgs(rows);
        const z_shape = [_]c_int{ ci(d.ks_split) * r, ci(d.shape.out) };
        if (c.prep_in) {
            if (!idxInsideFits(d, try tiling(rows))) return error.Exl3EngineNotImplemented;
            const nx = try xsxCount(d, rows);
            const x_shape = [_]c_int{ci(nx)};
            var k2 = key;
            k2.inside = true;
            const shapes = [_][]const c_int{ &y_shape, &z_shape, &x_shape, oq_shape, op_shape };
            const dtypes = [_]mlx.mlx_dtype{ bf, .float32, .float32, bf, bf };
            const launch = try decLaunch(d, rows, &extra_t, &tb, &shapes, &dtypes);
            const got = try cfgGet(k2, launch);
            defer cfgDone(got);
            const zin = [_]mlx.mlx_array{idx.svh};
            var cnt: [1]mlx.mlx_array = undefined;
            try ks.applyConfig(d.dec_zero, c.prep, &zin, &cnt, s);
            defer _ = mlx.mlx_array_free(cnt[0]);
            const ins = [_]mlx.mlx_array{ x2, idx.suh, idx.trellis, idx.svh, cnt[0] } ++ extra_in;
            var o: [5]mlx.mlx_array = undefined;
            try ks.applyConfig(kz.dec1_idx, got.cfg, &ins, &o, s);
            freeAll(o[1..3]);
            out = .{ .qk = o[0], .q_rope = o[3], .pooled = o[4] };
        } else {
            const shapes = [_][]const c_int{ &y_shape, &z_shape, oq_shape, op_shape };
            const dtypes = [_]mlx.mlx_dtype{ bf, .float32, bf, bf };
            const launch = try decLaunch(d, rows, &extra_t, &tb, &shapes, &dtypes);
            const got = try cfgGet(key, launch);
            defer cfgDone(got);
            const pin = [_]mlx.mlx_array{ x2, idx.suh };
            var po: [3]mlx.mlx_array = undefined;
            try ks.applyConfig(d.prep_sc, c.prep, &pin, &po, s);
            defer freeAll(&po);
            const ins = [_]mlx.mlx_array{ po[0], po[1], idx.trellis, idx.svh, po[2] } ++ extra_in;
            var o: [4]mlx.mlx_array = undefined;
            try ks.applyConfig(kz.dec0_idx, got.cfg, &ins, &o, s);
            _ = mlx.mlx_array_free(o[1]);
            out = .{ .qk = o[0], .q_rope = o[2], .pooled = o[3] };
        }
    } else {
        const c = try d.cfgs(rows);
        const pin = [_]mlx.mlx_array{ x2, idx.suh };
        var po: [2]mlx.mlx_array = undefined;
        try ks.applyConfig(d.prep_s, c.prep, &pin, &po, s);
        defer freeAll(&po);
        const gin = [_]mlx.mlx_array{ po[0], idx.trellis };
        var z: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(if (c.lean) d.dgemv_lean else d.dgemv, c.gemv, &gin, &z, s);
        defer _ = mlx.mlx_array_free(z[0]);
        const shapes = [_][]const c_int{ &y_shape, oq_shape, op_shape };
        const dtypes = [_]mlx.mlx_dtype{ bf, bf, bf };
        const launch = try dfinLaunch(d, rows, &extra_t, &tb, &shapes, &dtypes);
        const got = try cfgGet(key, launch);
        defer cfgDone(got);
        const ins = [_]mlx.mlx_array{ z[0], idx.svh, po[1] } ++ extra_in;
        var o: [3]mlx.mlx_array = undefined;
        try ks.applyConfig(kz.dfin_idx, got.cfg, &ins, &o, s);
        out = .{ .qk = o[0], .q_rope = o[1], .pooled = o[2] };
    }
    if (!ep.qepi) {
        _ = mlx.mlx_array_free(out.q_rope);
        out.q_rope = .{ .ctx = null };
    }
    if (!ep.pool) {
        _ = mlx.mlx_array_free(out.pooled);
        out.pooled = .{ .ctx = null };
    }
    return out;
}

// ---------------------------------------------------------------------------------------------- CPU tests (no device)

const testing = std.testing;

fn has(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}
fn count(hay: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, hay, needle);
}

/// Each line of `text` that holds a statement (trimmed, non-empty, not a comment), in order
fn stmtLines(a: std.mem.Allocator, text: []const u8) ![][]const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " ");
        if (t.len == 0 or std.mem.startsWith(u8, t, "//")) continue;
        try out.append(a, t);
    }
    return out.toOwnedSlice(a);
}

test "turboexl3 decode merges texts: dense's split finishes are its old texts, and every variant carries the chain's value-setting statements" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("../../core/kernels/test/pins/decb_dense_statements.metal"));
    const a = testing.allocator;
    // the butterfly the merges splice is dense's: its prep's and its finish's statements, verbatim
    try testing.expect(has(dense.text_prep_s, had128) and has(dense.dfinish_v, had128));
    // the splits keep dense's kernels' bytes: text_dfinish and DEC-ONE's texts end in the store they always had
    try testing.expect(std.mem.eql(u8, dense.text_dfinish, dense.dfinish_v ++ dense.dfinish_store));
    try testing.expect(std.mem.endsWith(u8, dense.text_dec0, dense.dec_finish_v ++ dense.dec_finish_store));
    try testing.expect(std.mem.endsWith(u8, dense.text_dec1, dense.dec_finish_v ++ dense.dec_finish_store));
    // every DEC-ONE variant is DEC-ONE's GEMV and arrival, spliced verbatim, then dec_finish_v verbatim
    for ([_][]const u8{ text_dec0_idx, text_dec0_planes }) |t| {
        try testing.expect(std.mem.startsWith(u8, t, dense.dec_head));
        try testing.expect(has(t, dense.dec_gemv0 ++ dense.dec_gemv_arrive ++ dense.dec_xsum0 ++ dense.dec_finish_v));
    }
    for ([_][]const u8{ text_dec1_idx, text_dec1_planes }) |t| {
        try testing.expect(std.mem.startsWith(u8, t, dense.dec_head));
        try testing.expect(has(t, dense.dec_prep1 ++ dense.dec_gemv1 ++ dense.dec_gemv_arrive ++ dense.dec_xsum1 ++ dense.dec_finish_v));
    }
    for ([_][]const u8{ text_dfinish_idx, text_dfinish_planes }) |t| try testing.expect(std.mem.startsWith(u8, t, dense.dfinish_v));
    // the finish's output value, as the chain's store forms it, in every variant
    const store_dec = pin[0];
    for ([_][]const u8{ text_dec0_idx, text_dec1_idx, text_dfinish_idx, text_dec0_planes, text_dec1_planes, text_dfinish_planes }) |t| try testing.expect(has(t, store_dec));
    try testing.expect(has(dense.dec_finish_store, store_dec) and has(dense.dfinish_store, store_dec));
    // the indexer's chain variant finishes the key block's rows 1.. with dfinish_v's statements (row -> rq)
    const Pair = struct { chain: []const u8, rows1: []const u8 };
    const pairs = [_]Pair{
        .{ .chain = pin[1], .rows1 = pin[2] },
        .{ .chain = pin[3], .rows1 = pin[4] },
        .{ .chain = pin[5], .rows1 = pin[6] },
        .{ .chain = pin[7], .rows1 = pin[7] },
        .{ .chain = pin[8], .rows1 = pin[8] },
        .{ .chain = pin[9], .rows1 = pin[9] },
    };
    for (pairs) |p| {
        try testing.expect(has(dense.dfinish_v, p.chain));
        try testing.expect(has(text_dfinish_idx, p.rows1));
    }
    // and every statement of dfinish_v that sets a value is one of those, the butterfly or a constant (none is missed)
    const v_lines = try stmtLines(a, dense.dfinish_v);
    defer a.free(v_lines);
    for (v_lines) |l| {
        var known = has(l, pin[10]) or has(l, pin[11]) or has(l, pin[12]) or has(l, "MUL1_") or
            std.mem.eql(u8, l, pin[13]) or std.mem.eql(u8, l, pin[14]) or std.mem.eql(u8, l, pin[15]) or std.mem.eql(u8, l, "}") or
            has(had128, l);
        for (pairs) |p| known = known or std.mem.eql(u8, l, p.chain);
        try testing.expect(known);
    }
    // the gated prepares: the gate product as the host's strided gate kernel forms it, then text_prep_s's statements
    // (the host's CPU test pins `xv * sigtab[as_type<ushort>(gv)]` to its strided gate kernel's store)
    try testing.expect(has(text_prep_gate_sc, pin[16]));
    try testing.expect(has(text_prep_gate_sc, pin[17]));
    try testing.expect(has(text_merge_gate_sc, pin[18]));
    try testing.expect(has(text_merge_gate_sc, pin[19]));
    try testing.expect(has(text_merge_gate_sc, pin[20]));
    try testing.expect(has(merge_store, pin[21]));
    const ps_lines = try stmtLines(a, dense.text_prep_s);
    defer a.free(ps_lines);
    for (ps_lines) |l| {
        // text_prep_s's statements after its input read (x's own read is where the gate's product replaces it)
        if (has(l, pin[22]) or has(l, pin[11]) or has(l, pin[12])) continue;
        try testing.expect(has(text_prep_gate_sc, l));
        if (!has(l, pin[10]) and !has(l, pin[14])) try testing.expect(has(text_merge_gate_sc, l));
    }
    try testing.expect(has(dense.text_prep_sc, pin[23]));
    try testing.expect(has(text_prep_gate_sc, pin[23]));
    for ([_][]const u8{ text_prep_gate_sc, text_merge_gate_sc }) |t| try testing.expectEqual(@as(usize, 1), count(t, had128));
    try testing.expectEqual(@as(usize, 2), count(text_dfinish_idx, had128)); // dfinish_v's and the rows-1.. finish's
}

test "turboexl3 decode merges texts: the indexer epilogues are the host's hd-128 QK-norm+RoPE and pooled-key statements, names mapped" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("../../core/kernels/test/pins/decb_indexer.metal"));
    // QK_NORM_ROPE_SOURCE's arithmetic lines (transformer.zig), with lane -> L, HD -> IHD, eps -> qeps, w -> qn, angles -> arow,
    // T -> OUT_T (the host's CPU test checks these copies against its own text)
    const qrope = [_][]const u8{
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
    };
    // QSA_POOL_ROPE_SOURCE's arithmetic lines, names mapped as idx_pool's doc says
    const pool = [_][]const u8{
        pin[11],
        pin[12],
        pin[13],
        pin[14],
        pin[15],
        pin[16],
        pin[17],
        pin[18],
        pin[19],
        pin[20],
        pin[21],
        pin[22],
        pin[23],
        pin[24],
        pin[25],
    };
    for ([_][]const u8{ text_dec0_idx, text_dec1_idx, text_dfinish_idx }) |t| {
        for (qrope) |l| try testing.expect(has(t, l));
        for (pool) |l| try testing.expect(has(t, l));
        // the block sum folds rows 0..R-1 in order, each row's bf16 value read as f32
        try testing.expect(has(t, pin[26]));
    }
    try testing.expectEqual(@as(usize, 1), count(idx_qrope, pin[27]));
    try testing.expectEqual(@as(usize, 1), count(idx_pool, pin[27]));
}

test "turboexl3 decode merges texts: the signatures' names are used and no name is followed by MLX's shape / strides / ndim suffixes" {
    const pin = comptime @import("../../core/test_support.zig").pins(@embedFile("../../core/kernels/test/pins/decb_signatures.metal"));
    const Sig = struct { txt: []const u8, names: []const [*:0]const u8, unused: []const []const u8 = &.{} };
    const sigs = [_]Sig{
        .{ .txt = text_prep_gate_sc, .names = &(prep_gate_sc_in ++ prep_sc_out) },
        .{ .txt = text_merge_gate_sc, .names = &(merge_gate_in ++ prep_sc_out) },
        .{ .txt = text_dec0_planes, .names = &(dec0_planes_in ++ dec0_planes_out) },
        .{ .txt = text_dec1_planes, .names = &(dec1_planes_in ++ dec1_planes_out) },
        .{ .txt = text_dfinish_planes, .names = &(dfin_planes_in ++ dfin_planes_out) },
        .{ .txt = text_dec0_idx, .names = &(dec0_idx_in ++ dec0_idx_out) },
        .{ .txt = text_dec1_idx, .names = &(dec1_idx_in ++ dec1_idx_out) },
        .{ .txt = text_dfinish_idx, .names = &(dfin_idx_in ++ dfin_idx_out) },
    };
    for (sigs) |sg| for (sg.names) |n| {
        const name = std.mem.span(n);
        var used = false;
        // a use: the name followed by an index, a cast or an arithmetic operator (not a substring of a longer name)
        for ([_][]const u8{ "[", " +", ")", ";", " *", ",", " :" }) |after| {
            var buf: [48]u8 = undefined;
            if (has(sg.txt, std.fmt.bufPrint(&buf, "{s}{s}", .{ name, after }) catch unreachable)) used = true;
        }
        try testing.expect(used);
        var buf: [48]u8 = undefined;
        for ([_][]const u8{ "_shape", "_strides", "_ndim" }) |suf| {
            const needle = std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suf }) catch unreachable;
            // the merge reads pacc's shape on purpose (its NSPLIT and qL ride it, as the host's merge)
            if (std.mem.eql(u8, needle, "pacc_shape")) continue;
            try testing.expect(!has(sg.txt, needle));
        }
    };
    try testing.expect(has(text_merge_gate_sc, pin[0]) and has(text_merge_gate_sc, pin[1]));
}

test "turboexl3 decode merges texts: the q_proj planes address each head's query and gate halves, every column once" {
    // planes_store's map at Qwen3.8's q_proj: 24 heads x [query 256 | gate 256]
    const heads: u32 = 24;
    const hd: u32 = 256;
    var seen_q: [24 * 256]u8 = @splat(0);
    var seen_g: [24 * 256]u8 = @splat(0);
    var o: u32 = 0;
    while (o < 2 * heads * hd) : (o += 1) {
        const hh = o / (2 * hd);
        const c = o - hh * (2 * hd);
        const at = hh * hd + (c % hd);
        if (c < hd) seen_q[at] += 1 else seen_g[at] += 1;
        // the chain's split: queries are columns [0, 256) of each head's 512, the gate the rest
        if (c < hd) try testing.expectEqual(hh * 2 * hd + (c % hd), o) else try testing.expectEqual(hh * 2 * hd + hd + (c % hd), o);
    }
    for (seen_q) |n| try testing.expectEqual(@as(u8, 1), n);
    for (seen_g) |n| try testing.expectEqual(@as(u8, 1), n);
    // a 128-block lies inside one half (PH a multiple of 128), so a block's stores all go to one plane
    try testing.expect(hd % 128 == 0);
}

test "turboexl3 decode merges: the gated prepares zero exactly DEC-ONE's counters and cover every prepared element once" {
    // prep_gate's grid (NB * 32 lanes, rows): g = (row NB + b) 32 + L steps by NTH = NB 32 rows; the merge's (256, rows,
    // heads): g = (row heads + hq) 256 + d steps by 256 rows heads. Both cover [0, NCNT) exactly once for every count
    // DEC-ONE uses at o_proj (out 2560: 20 blocks x chunks).
    const a = testing.allocator;
    for ([_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16 }) |rows| {
        const rb = dense.Dense.rbFor(rows);
        const chunks = (rows + rb - 1) / rb;
        const ncnt: u32 = @max(20 * chunks, 8);
        const nb: u32 = 6144 / 128;
        var seen = try a.alloc(u8, ncnt);
        defer a.free(seen);
        @memset(seen, 0);
        const nth = nb * 32 * rows;
        for (0..rows) |row| for (0..nb) |b| for (0..32) |l| {
            var g: u32 = @intCast((row * nb + b) * 32 + l);
            while (g < ncnt) : (g += nth) seen[g] += 1;
        };
        for (seen) |n| try testing.expectEqual(@as(u8, 1), n);
        @memset(seen, 0);
        const heads: u32 = 24;
        const nth2 = 256 * rows * heads;
        for (0..rows) |row| for (0..heads) |hq| for (0..256) |d| {
            var g: u32 = @intCast((row * heads + hq) * 256 + d);
            while (g < ncnt) : (g += nth2) seen[g] += 1;
        };
        for (seen) |n| try testing.expectEqual(@as(u8, 1), n);
        // the merge's prepare: head hq's two simdgroups own o_proj input blocks 2 hq and 2 hq + 1, every block once
        var blocks: [48]u8 = @splat(0);
        for (0..heads) |hq| for (0..2) |sgi| {
            blocks[hq * 2 + sgi] += 1;
        };
        for (blocks) |n| try testing.expectEqual(@as(u8, 1), n);
    }
}

/// This file's module state (`Global`): what a host that shares its units' state registers.
pub const shared_state = .{ kernels_lock, kernels, cfg_cache, tiling_cache, dummy_f32 };
