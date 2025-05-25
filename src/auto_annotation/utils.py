"""Generic utility helpers used across the project."""

import os
import zipfile
from pathlib import Path


def zip_directory(output_dir: Path, zip_filename: Path):
    """Compress the specified directory into a zip archive.

    Parameters
    ----------
    output_dir : Path
        The directory to be zipped.
    zip_filename : Path
        The target zip file path.
    """
    output_dir = Path(output_dir)
    zip_filename = Path(zip_filename)

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"指定されたディレクトリが存在しません: {output_dir}")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(output_dir.parent))
                print(f"追加: {file_path}")

    print(f"ディレクトリ '{output_dir}' を '{zip_filename}' に圧縮しました。")