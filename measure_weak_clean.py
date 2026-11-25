"""Reproduce clean WSINDy/WENDy measurements, models, logs and provenance."""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

from clean_run_metrics import (
    DEFAULT_DATASETS, ROOT, default_output, error_row, load_framework_module,
    normalize_result, run_framework, summarize_system, summarize_target, write_rows,
)
from utils.protocols import ProtocolUnavailableError
from utils.randomness import seed_everything
from utils.weak_sources import load_source, source_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    packages = {}
    for package in ("numpy", "scipy", "torch", "sympy", "symengine", "matplotlib", "ipython", "tqdm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "packages": packages,
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def measure(framework, protocol, datasets, seed):
    module = load_framework_module(framework)
    # Dependency imports are outside the measured data-to-equation interval.
    load_source(framework)
    output = default_output(framework, protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(protocol=protocol, algorithm_seed=seed, device="cpu",
                           solution_index=0, allow_external_llm=False)
    rows, details = [], []
    started = datetime.now(timezone.utc).isoformat()
    for dataset in datasets:
        seed_everything(seed)
        stdout, stderr = io.StringIO(), io.StringIO()
        print(f"{framework} / {protocol} / {dataset}", flush=True)
        result = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                start = time.perf_counter()
                result = normalize_result(run_framework(framework, module, dataset, args))
                elapsed = time.perf_counter() - start
            current = [summarize_target(framework, dataset, result, index, elapsed)
                       for index in range(len(result["targets"]))]
            if len(current) > 1:
                current.append(summarize_system(framework, dataset, current, elapsed, result=result))
            details.append({"dataset": dataset, "runtime_seconds": elapsed, "result": result})
        except ProtocolUnavailableError as error:
            current = [error_row(framework, dataset, error, protocol, status="unsupported")]
            details.append({"dataset": dataset, "status": "unsupported", "reason": str(error)})
        except Exception as error:
            import traceback
            traceback.print_exc(file=stderr)
            current = [error_row(framework, dataset, error, protocol)]
            details.append({"dataset": dataset, "status": "error", "reason": str(error)})
        rows.extend(current)
        log = stdout.getvalue() + stderr.getvalue()
        if log:
            (output.parent / (Path(dataset).stem + ".log")).write_text(log, encoding="utf-8")
        else:
            (output.parent / (Path(dataset).stem + ".log")).unlink(missing_ok=True)
        write_rows(rows, output)
        (output.parent / "clean_models.json").write_text(json.dumps(details, indent=2, allow_nan=False), encoding="utf-8")
        representative = current[-1]
        print(f"  {representative['status']}: HD={representative['structure_hamming']} RE={representative['coefficient_error']} time={representative['runtime_seconds']}", flush=True)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    inputs = [ROOT / "data" / dataset for dataset in datasets if (ROOT / "data" / dataset).is_file()]
    # Time vectors and generated inputs also influence the common loader.
    inputs.extend(path for path in (ROOT / "data").glob("*time*.npy") if path not in inputs)
    scripts = [ROOT / "clean_run_metrics.py", ROOT / "measure_weak_clean.py", ROOT / framework / "run.py",
               ROOT / "data" / "config.py", *sorted((ROOT / "utils").glob("*.py"))]
    manifest = {
        "framework": framework, "protocol": protocol, "noise_level": 0,
        "algorithm_seed": seed, "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets, "source": source_manifest(framework),
        "environment": environment(), "git_head": git_head,
        "working_tree": "Includes uncommitted changes; exact evaluated file hashes are recorded below.",
        "files_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in scripts + inputs},
        "timing": "One clean run per dataset; wall-clock data loading, library and fit; imports and metric calculation excluded. No hyperparameter or ground-truth search.",
    }
    (output.parent / "clean_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return any(row["status"] == "error" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frameworks", nargs="+", choices=["wsindy", "wendy"], default=["wsindy", "wendy"])
    parser.add_argument("--protocols", nargs="+", choices=["fixed", "native"], default=["fixed", "native"])
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--algorithm-seed", type=int, default=0)
    args = parser.parse_args()
    failed = False
    for framework in args.frameworks:
        for protocol in args.protocols:
            failed = measure(framework, protocol, args.datasets, args.algorithm_seed) or failed
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
