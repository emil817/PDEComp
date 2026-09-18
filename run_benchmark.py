"""Run one benchmark protocol across isolated framework containers."""

import argparse
import subprocess
import sys

from utils.protocols import FIXED_PROTOCOL, PROTOCOLS


FRAMEWORKS = ("pysindy", "deepmod", "epde", "discover", "edl", "vwsr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEComp metrics in each framework's Docker environment."
    )
    parser.add_argument("metric", choices=["clean", "noise"])
    parser.add_argument("--protocol", choices=PROTOCOLS, default=FIXED_PROTOCOL)
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--levels", nargs="*", type=float, default=None)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-max-iterations", type=int, default=None)
    parser.add_argument("--native-max-samples", type=int, default=None)
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--algorithm-seed", type=int, default=0)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def framework_command(framework, args):
    script = "clean_run_metrics.py" if args.metric == "clean" else "noise_test.py"
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        framework,
        "python",
        script,
        framework,
        "--protocol",
        args.protocol,
        "--device",
        args.device,
        "--algorithm-seed",
        str(getattr(args, "algorithm_seed", 0)),
    ]
    if args.datasets:
        command.extend(["--datasets", *args.datasets])
    if args.metric == "noise":
        command.extend(["--runs", str(args.runs)])
        if args.levels is not None:
            command.extend(["--levels", *map(str, args.levels)])
    if args.native_max_iterations is not None:
        command.extend(["--native-max-iterations", str(args.native_max_iterations)])
    if args.native_max_samples is not None:
        command.extend(["--native-max-samples", str(args.native_max_samples)])
    if args.allow_external_llm:
        command.append("--allow-external-llm")
    if getattr(args, "allow_empty", False):
        command.append("--allow-empty")
    return command


def main():
    args = parse_args()
    failed = []
    for framework in args.frameworks:
        print(f"\n=== {args.protocol} / {args.metric} / {framework} ===", flush=True)
        completed = subprocess.run(framework_command(framework, args), check=False)
        if completed.returncode:
            failed.append(framework)
    if failed:
        print(f"\nFailed containers: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
