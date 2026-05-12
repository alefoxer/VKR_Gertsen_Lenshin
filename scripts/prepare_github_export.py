from __future__ import annotations

import argparse
import shutil
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
    ".gitignore",
    "app.py",
    "INSTALL.md",
    "README.md",
    "requirements.txt",
    "run_local.bat",
    "run_lan.bat",
    "run_share.bat",
]

EXCLUDED_DIRS = {
    "__pycache__",
    ".gradio",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".tmp",
    ".bak",
}

EXCLUDED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

ALLOWED_MODEL_FILES = {
    "gradv_ru_kws_baseline.pt",
    "gradv_ru_kws_vocab.txt",
    "gradv_ru_kws_training_summary.json",
}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    current_dir = Path(directory)
    is_models_dir = current_dir.name == "models"

    for name in names:
        path = Path(name)
        if name in EXCLUDED_DIRS or name in EXCLUDED_NAMES:
            ignored.add(name)
        elif path.suffix.lower() in EXCLUDED_SUFFIXES:
            ignored.add(name)
        elif is_models_dir and name not in ALLOWED_MODEL_FILES:
            ignored.add(name)

    return ignored


def _copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_upload_steps(export_dir: Path, zip_path: Path | None) -> None:
    lines = [
        "# Загрузка проекта в GitHub",
        "",
        "Эта папка содержит чистую копию GRADV для загрузки в репозиторий.",
        "",
        "## Вариант 1: через сайт GitHub",
        "",
        "1. Создайте пустой репозиторий на GitHub.",
        "2. Загрузите содержимое этой папки, а не саму папку целиком.",
        "3. Нажмите `Commit changes` в интерфейсе GitHub.",
        "",
        "## Вариант 2: через git на компьютере",
        "",
        "```powershell",
        "git init",
        "git add .",
        "git commit -m \"Initial GRADV project\"",
        "git branch -M main",
        "git remote add origin https://github.com/<user>/<repo>.git",
        "git push -u origin main",
        "```",
        "",
        "## Что включено",
        "",
        "- исходный код",
        "- документация",
        "- скрипты запуска и проверки",
        "- компактные baseline-файлы из `models/`",
        "- `.gitignore`",
        "",
        "## Что не включено",
        "",
        "- результаты запусков из `outputs/`",
        "- DOCX, PNG-рендеры и временные материалы",
        "- runtime-логи",
        "- кэши Python и локальные виртуальные окружения",
    ]
    if zip_path is not None:
        lines.extend(["", f"ZIP-архив создан рядом с папкой экспорта: `{zip_path.name}`"])

    export_dir.joinpath("GITHUB_UPLOAD_STEPS.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_export(output_root: Path, make_zip: bool) -> tuple[Path, Path | None]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = output_root / f"gradv_github_export_{timestamp}"
    if export_dir.exists():
        raise FileExistsError(export_dir)
    export_dir.mkdir(parents=True, exist_ok=False)

    for dir_name in INCLUDE_DIRS:
        _copy_tree(PROJECT_ROOT / dir_name, export_dir / dir_name)

    for file_name in INCLUDE_FILES:
        _copy_file(PROJECT_ROOT / file_name, export_dir / file_name)

    zip_path: Path | None = None
    _write_upload_steps(export_dir, zip_path)

    if make_zip:
        zip_path = Path(shutil.make_archive(str(export_dir), "zip", root_dir=export_dir))
        _write_upload_steps(export_dir, zip_path)
        zip_path.unlink(missing_ok=True)
        zip_path = Path(shutil.make_archive(str(export_dir), "zip", root_dir=export_dir))

    return export_dir, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean GRADV folder for manual GitHub upload.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "github_export")
    parser.add_argument("--no-zip", action="store_true", help="Create only the folder, without a ZIP archive.")
    args = parser.parse_args()

    export_dir, zip_path = prepare_export(output_root=args.output_root, make_zip=not args.no_zip)
    print("[OK] GitHub export prepared")
    print(f"EXPORT_DIR={export_dir}")
    if zip_path is not None:
        print(f"EXPORT_ZIP={zip_path}")


if __name__ == "__main__":
    main()
