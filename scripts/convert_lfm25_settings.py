"""Choose official per-projection Hessian options without changing the CUDA quantizer."""

from __future__ import annotations

from contextlib import contextmanager

from pilot_lfm25_settings import profile_options, projection_scaled


@contextmanager
def converter_settings(converter, profile):
    profile_options(profile)
    names = ("quantize_linears_single", "quantize_linears_parallel")
    originals = {name: getattr(converter, name) for name in names}

    def wrap(original):
        def quantize(
            args,
            linears,
            config,
            strategy,
            idx,
            devices,
            device_ratios,
            capture_H,
            state,
            out_sens=None,
        ):
            if len({l.key for l in linears}) != len(linears):
                raise ValueError("Expected unique projections")
            groups = {False: [], True: []}
            shared = {}
            for linear in linears:
                scaled = projection_scaled(profile, linear.key)
                if linear.qmap in shared and shared[linear.qmap] != scaled:
                    raise ValueError(
                        "A shared Hessian cannot have conflicting factorization settings"
                    )
                shared[linear.qmap] = scaled
                groups[scaled].append(linear)
            for scaled, group in groups.items():
                if group:
                    print(
                        f"Official settings: {profile}, scaled={scaled}, projections={len(group)}",
                        flush=True,
                    )
                    original(
                        {**args, "scaled_hessian": scaled},
                        group,
                        config,
                        strategy,
                        idx,
                        devices,
                        device_ratios,
                        capture_H,
                        state,
                        out_sens=out_sens,
                    )

        return quantize

    try:
        for name, original in originals.items():
            setattr(converter, name, wrap(original))
        yield
    finally:
        for name, original in originals.items():
            setattr(converter, name, original)


def run(converter, profile, arguments):
    profile_options(profile)
    parsed = converter.parser.parse_args(arguments)
    args, state, ok, error = converter.prepare(parsed)
    if not ok:
        raise ValueError(f"Official converter rejected its arguments: {error}")
    if args.get("codebook") != "mcg":
        raise ValueError("The registered profiles require the MCG codebook")
    with converter_settings(converter, profile):
        converter.main(args, state)


if __name__ == "__main__":
    import sys

    from exllamav3.conversion import convert_model

    if len(sys.argv) < 3:
        raise SystemExit("profile official-converter-arguments...")
    run(convert_model, sys.argv[1], sys.argv[2:])
