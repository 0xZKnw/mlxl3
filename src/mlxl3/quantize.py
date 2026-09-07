"""Exact Metal search acceleration for the optional PonyExl3 converter.

Keeps upstream search arithmetic, tie-breaking and the two tail-biting passes.
PonyExl3 is only imported when converting, never by the inference engine.
"""

from functools import lru_cache


@lru_cache(maxsize=3)
def _decoded_codebook(cb):
    """Decode with the original Metal expression, including half rounding."""
    import mlx.core as mx
    from ponyexl3.convert import metal_search as ref

    make = mx.fast.metal_kernel(
        name=f"mlxl3_codebook_cb{int(cb)}",
        input_names=["dummy"],
        output_names=["values"],
        source="uint cw = thread_position_in_grid.x;\n"
        + ref._decode_expr(cb)
        + "values[cw] = dq_val;",
    )
    values = make(
        inputs=[mx.array(0)],
        grid=(65536, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(65536,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(values)
    return values


@lru_cache(maxsize=6)
def _ordered_kernel(k, cb):
    """Exact distance-ordered search for K=7/8, with a predecessor-cost bound."""
    import mlx.core as mx
    import numpy as np
    from ponyexl3.convert import metal_search as ref

    values = np.asarray(_decoded_codebook(cb)).reshape(1 << k, -1)
    order = np.argsort(values, axis=0, kind="stable")
    tables = (
        mx.array(np.take_along_axis(values, order, axis=0).reshape(-1)),
        mx.array(order.astype(np.uint8).reshape(-1)),
    )
    mx.eval(*tables)
    groups = 1 << (16 - 2 * k)
    source = ref._source(k, cb)
    begin, end = "            for (uint e = tid;", "\n            }"
    try:
        start = source.index(begin, source.index("for (uint step = 1u;"))
        stop = source.index(end, source.index("edge_ptr[ulong(ri)", start)) + len(end)
    except ValueError as exc:
        raise RuntimeError(
            "Unsupported PonyExl3 search source; use --reference-metal-search"
        ) from exc
    body = f"""
            for (uint g = tid; g < {groups}u; g += 256u) {{
                half lo = half(H_INF_F);
                for (uint fresh = 0; fresh < KK; fresh++)
                    lo = min(lo, costs[prev_buf][fresh * {groups}u + g]);
                minima[g] = lo;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = tid; e < EDGES; e += 256u) {{
                float target = in_tile[ri];
                int low = 0, high = int(KK);
                while (low < high) {{
                    int mid = (low + high) / 2;
                    if (sorted_values[uint(mid) * EDGES + e] < target) low = mid + 1;
                    else high = mid;
                }}
                int left = low - 1, right = low;
                float best = H_INF_F;
                uint best_pred = 0u;
                float minimum = float(minima[e >> K_BITS]);
                for (uint iteration = 0; iteration < KK; iteration++) {{
                    float dl = left >= 0 ? sorted_values[uint(left) * EDGES + e] - target : INFINITY;
                    float dr = right < int(KK) ? sorted_values[uint(right) * EDGES + e] - target : INFINITY;
                    bool take_left = right >= int(KK) || (left >= 0 && abs(dl) <= abs(dr));
                    float dh = take_left ? dl : dr;
                    // Strict bound: equal-cost candidates may still win the reference tie-break.
                    if (fma(dh, dh, minimum) > best) break;
                    uint pos = uint(take_left ? left-- : right++);
                    uint fresh = uint(sorted_fresh[pos * EDGES + e]);
                    uint pred = fresh * {groups}u + (e >> K_BITS);
                    float err = fma(dh, dh, float(costs[prev_buf][pred]));
                    if (err > H_INF_F) err = H_INF_F;
                    if (err < best || (err == best && pred < best_pred)) {{
                        best = err;
                        best_pred = pred;
                    }}
                }}
                costs[curr][e] = half(best);
                if (pass == 1u || ri < 128u) edge_ptr[ulong(ri) * EDGES + e] = ushort(best_pred);
            }}
"""
    source = (source[:start] + body + source[stop:]).replace(
        "threadgroup uint sh_pin;",
        f"threadgroup uint sh_pin;\nthreadgroup half minima[{groups}];",
    )
    kernel = mx.fast.metal_kernel(
        name=f"mlxl3_quantize_ordered_k{k}_cb{int(cb)}",
        input_names=["tiles", "sorted_values", "sorted_fresh"],
        output_names=["q_tiles", "indices", "temp_edges", "temp_costs"],
        source=source,
    )
    return kernel, tables


@lru_cache(maxsize=15)
def _paired_kernel(k, cb):
    """Share predecessor reads across aligned edges; retain float32 error math."""
    import mlx.core as mx
    from ponyexl3.convert import metal_search as ref

    if k == 4 and int(cb) == 2:
        return ref._kernel(k, cb), ()
    source = _shared_k3_source(cb) if k == 3 else ref._source(k, cb)
    if k < 4:
        # Normalize the two cost layouts so the transition generator is shared.
        source = source.replace(
            "device half* costs = temp_costs + ulong(tile) * 2u * EDGES;",
            "device half (*costs)[EDGES] = (device half (*)[EDGES])"
            "(temp_costs + ulong(tile) * 2u * EDGES);",
        ).replace("threadgroup half costs[2 * EDGES];", "threadgroup half costs[2][EDGES];")
        source = source.replace("costs[ulong(curr) * EDGES + e]", "costs[curr][e]")
        source = source.replace("costs[ulong(prev_buf) * EDGES + pred]", "costs[prev_buf][pred]")
        source = source.replace("costs[0]", "costs[0][0]")
    begin = "            for (uint e = tid;"
    end = "\n            }"
    try:
        start = source.index(begin, source.index("for (uint step = 1u;"))
        stop = source.index(end, source.index("edge_ptr[ulong(ri)", start)) + len(end)
    except ValueError as exc:
        raise RuntimeError(
            "Unsupported PonyExl3 search source; use --reference-metal-search"
        ) from exc
    width = {2: 4, 3: 8, 4: 16, 5: 4, 6: 4}[k]
    compact = k != 4
    cost_space = "device" if k == 2 else "threadgroup"
    body = f"for (uint e = tid * {width}u; e < EDGES; e += {256 * width}u) {{\n"
    for lane in range(width // 2):
        body += f"""
            float2 best{lane} = H_INF_F;
            ushort2 pred{lane} = 0u;
        """
    body += """
        float target = in_tile[ri];
        for (uint fresh = 0u; fresh < KK; fresh++) {
            uint pred = ((fresh << KR_BITS) | e) >> K_BITS;
            float previous = float(costs[prev_buf][pred]);
    """
    if k == 4:
        # Group the independent reads ahead of arithmetic to hide lookup latency.
        for lane in range(width // 2):
            body += f"half2 loaded{lane} = *((const device half2*)&codebook[(fresh << KR_BITS) + e + {2 * lane}u]);\n"
    for lane in range(width // 2):
        lookup = (
            f"loaded{lane}"
            if k == 4
            else f"*((const device half2*)&codebook[(fresh << KR_BITS) + e + {2 * lane}u])"
        )
        body += f"""
            {{
                float2 dq = float2({lookup});
                float2 dh = dq - target;
                float2 err = min(fma(dh, dh, float2(previous)), float2(H_INF_F));
                pred{lane} = select(pred{lane}, ushort2(pred), err < best{lane});
                best{lane} = min(best{lane}, err);
            }}
        """
    body += "}\n"
    for lane in range(width // 2):
        edge_type = "uchar2" if compact else "ushort2"
        edge_value = (
            f"select(uchar2(pred{lane} >> {16 - 2 * k}u), uchar2(128), pred{lane} == ushort2(0))"
            if compact
            else f"pred{lane}"
        )
        body += f"""
            *(({cost_space} half2*)&costs[curr][e + {2 * lane}u]) = half2(best{lane});
            if (pass == 1u || ri < 128u)
                ((device {edge_type}*)edge_ptr)[(ulong(ri) * EDGES + e) / 2u + {lane}u] = {edge_value};
        """
    body += "}\n"
    source = source[:start] + body + source[stop:]
    if compact:
        # Only the fresh label is needed: the other predecessor bits come from
        # the current edge. 128 marks the reference's default predecessor zero,
        # including saturated/NaN paths; it cannot collide with K<=6 labels.
        source = source.replace("device ushort* edge_ptr", "device uchar* edge_ptr")
        source = source.replace(
            "= ushort(best_pred);",
            f"= best_pred ? uchar(best_pred >> {16 - 2 * k}u) : uchar(128);",
        )
        source = source.replace(
            "uint prev = uint(edge_ptr[ulong(ri) * EDGES + edge]);",
            "uint fresh = uint(edge_ptr[ulong(ri) * EDGES + edge]);\n"
            f"uint prev = fresh == 128u ? 0u : (fresh << {16 - 2 * k}u) | (edge >> K_BITS);",
        )
    # Each codebook already rounds its decoded value to half. Exhaustively
    # checked: this lookup adds no rounding; error arithmetic stays float32.
    tables = (_decoded_codebook(cb).astype(mx.float16),)
    mx.eval(*tables)
    kernel = mx.fast.metal_kernel(
        name=f"mlxl3_quantize_group{width}_k{k}_cb{int(cb)}",
        input_names=["tiles", "codebook"],
        output_names=["q_tiles", "indices", "temp_edges", "temp_costs"],
        source=source,
    )
    return kernel, tables


@lru_cache(maxsize=3)
def _shared_k3_source(cb):
    from ponyexl3.convert import metal_search as ref

    # K=3's two half cost rows fit exactly in Metal's 32 KiB threadgroup budget.
    # Reuse a dead cost slot for the pin instead of allocating four more bytes.
    source = ref._source(3, cb)
    replacements = (
        (
            "device half* costs = temp_costs + ulong(tile) * 2u * EDGES;",
            "threadgroup half costs[2 * EDGES];",
        ),
        ("threadgroup uint sh_pin;", ""),
        ("sh_pin = edge;", "costs[0] = as_type<half>(ushort(edge));"),
        (
            "pin = sh_pin;",
            (
                "pin = uint(as_type<ushort>(costs[0]));\n"
                "threadgroup_barrier(mem_flags::mem_threadgroup);"
            ),
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Unsupported PonyExl3 search source; use --reference-metal-search")
        source = source.replace(old, new)
    source = source.replace(
        "mem_flags::mem_threadgroup | mem_flags::mem_device", "mem_flags::mem_threadgroup"
    )
    return source


def quantize_tiles_mlx(tiles, k, cb=0, *, max_scratch_bytes=256 * 1024 * 1024):
    """Same results as PonyExl3; bounded scratch and no unused device cost array."""
    import mlx.core as mx
    from ponyexl3.convert import cancel, timing
    from ponyexl3.convert import metal_search as ref
    from ponyexl3.ref.codebook import CodebookMode

    if not 2 <= k <= 8:
        raise ValueError("Metal trellis search currently supports K in [2, 8]")
    cb = CodebookMode(cb)
    cancel.raise_if_requested()
    arr = mx.array(tiles, dtype=mx.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 256)
    if arr.ndim != 2 or arr.shape[1] != 256:
        raise ValueError(f"expected tiles shape (N, 256), got {arr.shape}")
    n = arr.shape[0]
    if n == 0 or max_scratch_bytes <= 0:
        raise ValueError("expected at least one tile and a positive scratch budget")
    edges = 1 << (16 - k)
    device_costs = k == 2
    edge_dtype = mx.uint8 if k in (2, 3, 5, 6) else mx.uint16
    scratch = 256 * edges * edge_dtype.size + (4 * edges if device_costs else 0)
    batch = max(1, max_scratch_bytes // scratch)
    tables = ()
    if k >= 7:
        kernel, tables = _ordered_kernel(k, cb)
    else:
        kernel, tables = _paired_kernel(k, cb)
    threads = ref._threads_for_k(k)
    q_parts, idx_parts = [], []
    with timing.gpu("search"):
        for start in range(0, n, batch):
            cancel.raise_if_requested()
            count = min(batch, n - start)
            q, idx = kernel(
                inputs=[arr[start : start + count], *tables],
                template=[("T", mx.float32)],
                grid=(count * threads, 1, 1),
                threadgroup=(threads, 1, 1),
                output_shapes=[
                    (count, 256),
                    (count, 256),
                    (count, 256 * edges),
                    (count, 2, edges) if device_costs else (1,),
                ],
                output_dtypes=[mx.float32, mx.uint16, edge_dtype, mx.float16],
            )[:2]
            # At most two launches in flight: overlap CPU dispatch with Metal,
            # without retaining an entire large matrix's scratch allocations.
            if n > batch:
                mx.async_eval(q, idx)
                if q_parts:
                    mx.eval(q_parts[-1], idx_parts[-1])
            q_parts.append(q)
            idx_parts.append(idx)
        mx.eval(q_parts[-1], idx_parts[-1])
    if len(q_parts) == 1:
        return q_parts[0], idx_parts[0]
    return mx.concatenate(q_parts), mx.concatenate(idx_parts)
