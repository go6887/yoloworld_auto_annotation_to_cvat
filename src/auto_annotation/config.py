"""Configuration constants used across the auto_annotation package."""

from typing import List


class Config:
    """Simple configuration holder (kept as a class to avoid accidental mutation)."""

    # Path to the YOLO model file.
    model_path: str = "yolov8s-world.pt"

    # Classes that should be detected by the model.
    detection_target: List[str] = [
        "road sign",
    ]