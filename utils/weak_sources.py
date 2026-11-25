"""Fetch and load pinned upstream implementations without vendoring their code."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "wsindy": {
        "repository": "SethMinor/PyWSINDy-for-PDEs",
        "revision": "4c5765bc7948ae3409bea6c40f320d95ac1de74d",
        "files": {
            "dev/wsindy_dev.py": "wsindy_dev.py",
            "dev/helper_fcns_dev.py": "helper_fcns_dev.py",
        },
    },
    "wendy": {
        "repository": "MathBioCU/pyWENDy",
        "revision": "e071fc88f161f3f4256aeb949be30deba55e80b4",
        "files": {"WENDy.py": "WENDy.py", "LICENSE": "LICENSE"},
    },
}


def source_directory(framework):
    return ROOT / framework / "_upstream"


def fetch_sources(framework):
    source = SOURCES[framework]
    directory = source_directory(framework)
    directory.mkdir(parents=True, exist_ok=True)
    for remote, local in source["files"].items():
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{remote}"
        with urlopen(url, timeout=60) as response:
            contents = response.read()
        (directory / local).write_bytes(contents)
    return source_manifest(framework)


def source_manifest(framework):
    source = SOURCES[framework]
    return {
        **source,
        "sha256": {
            local: hashlib.sha256((source_directory(framework) / local).read_bytes()).hexdigest()
            for local in source["files"].values()
        },
    }


def load_source(framework):
    name, filename = {
        "wsindy": ("pdecomp_upstream_wsindy", "wsindy_dev.py"),
        "wendy": ("pdecomp_upstream_wendy", "WENDy.py"),
    }[framework]
    if name in sys.modules:
        return sys.modules[name]
    path = source_directory(framework) / filename
    if not path.exists():
        raise ImportError(f"Missing {path}. Run python -m utils.weak_sources {framework}")
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    finally:
        sys.path.pop(0)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frameworks", nargs="+", choices=SOURCES)
    args = parser.parse_args()
    for framework in args.frameworks:
        print(json.dumps(fetch_sources(framework), indent=2))
