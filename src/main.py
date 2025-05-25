# Rewritten orchestration script that leverages the modularized auto_annotation package.

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from auto_annotation.config import Config
from auto_annotation.cvat_tools import (
    convert_to_cvat_xml,
    process_image_for_xml,
    sort_xml_images_by_name,
)
from auto_annotation.utils import zip_directory
from auto_annotation.yolo_engine import YOLOEngine


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = YOLOEngine()

    images_annotations = []
    image_id = 0

    for image_path in sorted(input_dir.glob("*.jpg")):
        # YOLO inference
        result = engine.infer(image_path)
        print(f"{image_path} の推論結果: {result}")

        # Get actual image size
        with Image.open(image_path) as img:
            width, height = img.size

        image_info = {
            "file_name": image_path.name,
            "width": width,
            "height": height,
        }

        annotation = process_image_for_xml(image_id, image_info, result)
        images_annotations.append(annotation)
        image_id += 1

    task_info = {
        "id": 1,
        "name": args.task_name,
    }

    category_mapping = {
        idx: {"name": class_name} for idx, class_name in enumerate(Config.detection_target)
    }

    output_file = output_dir / "annotations.xml"

    convert_to_cvat_xml(images_annotations, task_info, category_mapping, output_file)
    sort_xml_images_by_name(output_file, output_file)

    if args.zip:
        zip_directory(output_dir, args.zip_name)
        print("処理が完了しました。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO-based auto annotation to CVAT XML format."
    )
    parser.add_argument(
        "--input-dir",
        default="20241210/images",
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--output-dir",
        default="20241210",
        help="Directory to store output XML and ZIP.",
    )
    parser.add_argument(
        "--task-name",
        default="CVAT ANNOTATION",
        help="Name of the generated CVAT task.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="If provided, the output directory will be zipped.",
    )
    parser.add_argument(
        "--zip-name",
        default="sample.zip",
        help="Name of the generated zip archive.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()