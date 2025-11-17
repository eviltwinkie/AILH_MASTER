#!/usr/bin/env python3
"""
collect_requirements.py

Fancy dependency scanner:

- Recursively scans a directory for Python files.
- Parses imports using AST (import X, import X.Y, from X import Y, etc.).
- Filters out:
    * Stdlib modules
    * Local project modules/packages under the root
- Maps common module names => pip package names (e.g. "cv2" -> "opencv-python").
- Checks installed status and versions via importlib.metadata + importlib.util.
- ALWAYS:
    * Prints a space-separated list of missing pip modules.
    * Writes requirements.txt at the project root.

Usage:
  python collect_requirements.py .
  python collect_requirements.py . --mode all
  python collect_requirements.py /path/to/project --mode missing

Modes (what goes into requirements.txt):
  - missing   : only packages that are NOT currently installed
  - installed : only packages that are installed (with pinned versions if possible)
  - all       : both (installed pinned + missing unpinned)  [default]
"""

import ast
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Iterable, Set, Dict, Optional, List

import importlib.util
import importlib.metadata


# ---------------------------------------------------------------------------
# Common module -> pip package mappings
# ---------------------------------------------------------------------------

MODULE_TO_PIP: Dict[str, str] = {
    # ML / data
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "bs4": "beautifulsoup4",
    "BeautifulSoup": "beautifulsoup4",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "tqdm": "tqdm",
    "requests": "requests",
    "aiohttp": "aiohttp",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
    "flask": "Flask",
    "django": "Django",
    "sqlalchemy": "SQLAlchemy",
    "psycopg2": "psycopg2-binary",  # often you want binary
    "pymongo": "pymongo",
    "redis": "redis",
    "celery": "celery",
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DepInfo:
    module: str                 # top-level module name e.g. "torch"
    pip_name: str               # guess of pip distribution name
    installed: bool
    version: Optional[str]      # distribution version if known


# ---------------------------------------------------------------------------
# File iteration / parsing
# ---------------------------------------------------------------------------

def iter_python_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip common junk/venv dirs
        dirnames[:] = [
            d for d in dirnames
            if d not in {
                ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
                ".mypy_cache", ".pytest_cache", ".tox", ".idea", ".vscode"
            }
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def get_stdlib_modules() -> Set[str]:
    """
    Try to get stdlib module names for this interpreter.
    """
    stdlib = set()
    if hasattr(sys, "stdlib_module_names"):
        stdlib.update(sys.stdlib_module_names)  # type: ignore[attr-defined]
    else:
        # Fallback minimal set
        stdlib.update({
            "sys", "os", "math", "json", "re", "itertools", "functools", "collections",
            "subprocess", "asyncio", "logging", "time", "datetime", "pathlib", "typing",
            "random", "unittest", "http", "urllib", "socket", "selectors", "threading",
            "multiprocessing", "statistics", "argparse", "shutil", "tempfile",
            "glob", "inspect", "dataclasses", "enum", "traceback", "contextlib",
            "csv", "hashlib", "hmac", "getpass", "base64", "heapq",
        })

    stdlib.update(sys.builtin_module_names)
    return stdlib


def parse_imports_from_file(path: str) -> Set[str]:
    """
    Return a set of top-level module names imported in a Python file.
    """
    modules: Set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return modules

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    top = alias.name.split(".")[0]
                    modules.add(top)

        elif isinstance(node, ast.ImportFrom):
            # relative imports (from .foo import bar) -> local
            if node.level and node.level > 0:
                continue
            if node.module:
                top = node.module.split(".")[0]
                modules.add(top)

    return modules


def find_local_packages(root: str) -> Set[str]:
    """
    Detect local packages/modules under the root directory so we can
    avoid mistakenly treating them as external pip deps.

    Heuristics:
      - Any directory with an __init__.py is a package name (directory name)
      - Any .py file is a module name
    """
    local: Set[str] = set()
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        if "__init__.py" in filenames:
            pkg_name = os.path.basename(dirpath)
            local.add(pkg_name)

        for fn in filenames:
            if fn.endswith(".py"):
                mod_name = os.path.splitext(fn)[0]
                local.add(mod_name)

    return local


# ---------------------------------------------------------------------------
# Dependency inspection
# ---------------------------------------------------------------------------

def guess_pip_name(module: str) -> str:
    """
    Map a module name to a pip distribution name, using a small heuristic table.
    Fallback: assume same as module name.
    """
    return MODULE_TO_PIP.get(module, module)


def is_module_installed(module: str) -> bool:
    """
    Check if a module is importable (not whether a specific pip package is installed).
    """
    spec = importlib.util.find_spec(module)
    return spec is not None


def get_distribution_version(dist_name: str) -> Optional[str]:
    """
    Try to obtain the version of a distribution by pip name.
    """
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def inspect_dependency(module: str) -> DepInfo:
    pip_name = guess_pip_name(module)
    installed = is_module_installed(module)
    version = get_distribution_version(pip_name) if installed else None
    return DepInfo(module=module, pip_name=pip_name, installed=installed, version=version)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_dep_row(dep: DepInfo) -> str:
    status = "INSTALLED" if dep.installed else "MISSING"
    version = dep.version if dep.version is not None else "-"
    return f"{dep.module:<20} {dep.pip_name:<25} {status:<10} {version:<15}"


def print_summary(deps: List[DepInfo]) -> None:
    print("=" * 80)
    print("Dependency summary")
    print("=" * 80)
    print(f"{'module':<20} {'pip_name':<25} {'status':<10} {'version':<15}")
    print("-" * 80)
    for dep in deps:
        print(format_dep_row(dep))

    installed = [d for d in deps if d.installed]
    missing = [d for d in deps if not d.installed]

    print("\nInstalled modules:")
    if installed:
        for d in installed:
            ver = d.version or "?"
            print(f"  {d.pip_name} (module={d.module}, version={ver})")
    else:
        print("  (none)")

    print("\nMissing modules:")
    if missing:
        for d in missing:
            print(f"  {d.pip_name} (module={d.module})")
    else:
        print("  (none)")


# ---------------------------------------------------------------------------
# Requirements output
# ---------------------------------------------------------------------------

def build_requirements_lines(deps: List[DepInfo], mode: str) -> List[str]:
    """
    mode:
      - 'missing'   -> only missing packages (unversioned)
      - 'installed' -> only installed packages, pinned to version if available
      - 'all'       -> installed pinned + missing unversioned
    """
    lines: List[str] = []
    # Use pip_name as key to avoid duplicates
    by_pip: Dict[str, DepInfo] = {}
    for d in deps:
        by_pip.setdefault(d.pip_name, d)

    for pip_name in sorted(by_pip.keys(), key=str.lower):
        d = by_pip[pip_name]

        if mode == "missing":
            if not d.installed:
                lines.append(pip_name)
        elif mode == "installed":
            if d.installed:
                if d.version:
                    lines.append(f"{pip_name}=={d.version}")
                else:
                    lines.append(pip_name)
        elif mode == "all":
            if d.installed:
                if d.version:
                    lines.append(f"{pip_name}=={d.version}")
                else:
                    lines.append(pip_name)
            else:
                lines.append(pip_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan a project for imports and generate a pip requirements.txt."
    )
    parser.add_argument(
        "root",
        help="Root directory of the project to scan (e.g. . or /path/to/src).",
    )
    parser.add_argument(
        "--mode",
        choices=["missing", "installed", "all"],
        default="all",
        help="Which dependencies to include in requirements.txt "
             "(missing | installed | all). Default: all.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    stdlib = get_stdlib_modules()
    local = find_local_packages(root)

    all_mods: Set[str] = set()
    for path in iter_python_files(root):
        mods = parse_imports_from_file(path)
        all_mods.update(mods)

    # Filter to external "interesting" modules
    candidates = sorted(
        m for m in all_mods
        if m
        and m not in stdlib
        and m not in local
        and not m.startswith("_")
    )

    deps: List[DepInfo] = [inspect_dependency(m) for m in candidates]

    print(f"# Project root: {root}")
    print(f"# Detected {len(candidates)} unique external module candidates.\n")
    print_summary(deps)

    # Always print a space-separated list of missing pip modules
    missing_pip_names = sorted(
        {d.pip_name for d in deps if not d.installed},
        key=str.lower,
    )
    if missing_pip_names:
        spaced = " ".join(missing_pip_names)
        print("\nMissing pip modules (space-separated):")
        print(spaced)
    else:
        print("\nMissing pip modules (space-separated):")
        print("(none)")

    # Always write requirements.txt at root
    requirements_path = os.path.join(root, "requirements.txt")
    lines = build_requirements_lines(deps, mode=args.mode)
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by collect_requirements.py\n")
        f.write(f"# Mode: {args.mode}\n")
        f.write("# NOTE: Verify mappings like cv2->opencv-python, PIL->Pillow, sklearn->scikit-learn, etc.\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"\nWrote {len(lines)} requirement lines to {requirements_path}")


if __name__ == "__main__":
    main()
