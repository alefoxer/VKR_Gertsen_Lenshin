from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


INCLUDE_DIRS = [
    "adapters",
    "analysis",
    "attacks",
    "audio",
    "docs",
    "experiments",
    "models",
    "scripts",
    "ui",
    "utils",
]

INCLUDE_FILES = [
    "app.py",
    "README.md",
    "requirements.txt",
    "run_local.bat",
    "run_lan.bat",
    "run_share.bat",
]

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".gradio",
}

EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}

EXCLUDED_FILE_NAMES = {
    "cloudflared_stderr.log",
    "cloudflared_stdout.log",
    "gradio_share_stderr.log",
    "gradio_share_stdout.log",
    "localtunnel_stderr.log",
    "localtunnel_stdout.log",
    "simple_share_stderr.log",
    "simple_share_stdout.log",
    "ssh_tunnel_stderr.log",
    "ssh_tunnel_stdout.log",
}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path_name = Path(name)
        if name in EXCLUDED_DIR_NAMES or name in EXCLUDED_FILE_NAMES:
            ignored.add(name)
        elif path_name.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
            ignored.add(name)
    return ignored


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def _ignore_demo(_directory: str, names: list[str]) -> set[str]:
    ignored = _ignore(_directory, names)
    for name in names:
        if name in {"runs", "studies"}:
            ignored.add(name)
    return ignored


def _copy_demo_export(src: Path, dst: Path) -> None:
    """Copy the portable demo without very deep transient run folders.

    Full run artifacts stay in the original outputs/demo_vkr folder. The release package
    keeps the human-facing protocol, manifest, assets, reports, summaries, and plots.
    This avoids Windows path length failures when the project itself already lives in a
    long OneDrive path.
    """
    if not src.exists():
        return
    shutil.copytree(src, dst, ignore=_ignore_demo, dirs_exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_release_notes(package_dir: Path, demo_dir: Path | None, zip_path: Path | None) -> None:
    lines = [
        "# GRADV Release Package",
        "",
        "This folder is a portable project snapshot for demonstration or transfer to another PC.",
        "",
        "## How to run",
        "",
        "1. Install Python 3.10+.",
        "2. Open a terminal in this folder.",
        "3. Install dependencies: `pip install -r requirements.txt`.",
        "4. Start locally: `python app.py` or `run_local.bat`.",
        "5. Open `http://127.0.0.1:7860`.",
        "",
        "For another PC on the same local network, run `run_lan.bat` and open the printed LAN address.",
        "For a temporary public demo, run `run_share.bat`; the external Gradio link works only while the app is running.",
        "",
        "## Included",
        "",
        "- source code and UI",
        "- compact `gradv_ru_kws_baseline` artifacts from `models/`",
        "- documentation and helper scripts",
    ]
    if demo_dir:
        lines.extend(["- copied VKR demo export in `demo_vkr/`"])
    if zip_path:
        lines.extend(["", f"ZIP archive: `{zip_path.name}`"])
    package_dir.joinpath("RELEASE_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def make_release_package(*, output_root: Path, demo_dir: Path | None, make_zip: bool) -> tuple[Path, Path | None]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = output_root / f"gradv_release_{timestamp}_{uuid.uuid4().hex[:6]}"
    if package_dir.exists():
        raise FileExistsError(package_dir)
    package_dir.mkdir(parents=True, exist_ok=False)

    for dir_name in INCLUDE_DIRS:
        _copy_tree(PROJECT_ROOT / dir_name, package_dir / dir_name)

    for file_name in INCLUDE_FILES:
        _copy_file(PROJECT_ROOT / file_name, package_dir / file_name)

    copied_demo_dir: Path | None = None
    if demo_dir:
        source_demo = demo_dir.resolve()
        if not source_demo.exists() or not source_demo.is_dir():
            raise FileNotFoundError(f"Demo directory was not found: {source_demo}")
        copied_demo_dir = package_dir / "demo_vkr" / source_demo.name
        _copy_demo_export(source_demo, copied_demo_dir)

    zip_path: Path | None = None
    if make_zip:
        archive_base = package_dir.with_suffix("")
        zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=package_dir))

    _write_release_notes(package_dir, copied_demo_dir, zip_path)
    if zip_path:
        # Refresh archive so RELEASE_NOTES.md is included.
        zip_path.unlink(missing_ok=True)
        zip_path = Path(shutil.make_archive(str(package_dir.with_suffix("")), "zip", root_dir=package_dir))
    return package_dir, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable GRADV release folder.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "release")
    parser.add_argument("--demo-dir", type=Path, default=None, help="Optional outputs/demo_vkr/<id> folder to include.")
    parser.add_argument("--zip", action="store_true", help="Also create a zip archive next to the release folder.")
    args = parser.parse_args()

    package_dir, zip_path = make_release_package(
        output_root=args.output_root,
        demo_dir=args.demo_dir,
        make_zip=bool(args.zip),
    )
    print("[OK] release package created")
    print(f"RELEASE_DIR={package_dir}")
    if zip_path:
        print(f"RELEASE_ZIP={zip_path}")


if __name__ == "__main__":
    main()
