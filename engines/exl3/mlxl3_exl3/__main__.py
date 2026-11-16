"""Command line entry: ``python -m mlxl3_exl3 {bridge,probe,download}``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlxl3_exl3")
    commands = parser.add_subparsers(dest="command", required=True)
    bridge = commands.add_parser("bridge", help="resident JSON-lines bridge")
    bridge.add_argument("model")
    bridge.add_argument("--context-length", type=int, default=0)
    bridge.add_argument("--backend", choices=["exl3", "fake"], default="exl3")
    commands.add_parser("probe", help="report torch, GPU and ExLlamaV3 status")
    download = commands.add_parser("download", help="download a checkpoint from Hugging Face")
    download.add_argument("repo")
    download.add_argument("destination")
    download.add_argument("--folder", default="")
    download.add_argument("--revision")
    args = parser.parse_args(argv)

    if args.command == "probe":
        from .probe import main as probe_main
        return probe_main()
    if args.command == "download":
        from .download import download as run_download
        try:
            run_download(args.repo, args.destination, args.folder, args.revision)
        except Exception as error:
            sys.stdout.write(json.dumps({"type": "error", "message": f"{type(error).__name__}: {error}"}) + "\n")
            return 1
        return 0
    from . import backend, bridge as bridge_module
    factory = backend.FakeBackend if args.backend == "fake" else backend.Exl3Backend
    return bridge_module.run(args.model, args.context_length, factory)


if __name__ == "__main__":
    sys.exit(main())
