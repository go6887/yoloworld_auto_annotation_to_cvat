"""Light abstraction around Ultralytics YOLO model.

This wrapper makes it easier to swap the underlying model implementation
without having to touch the rest of the codebase.
"""

from pathlib import Path
from typing import List, Optional, Union

from ultralytics import YOLO

from auto_annotation.config import Config


class YOLOEngine:
    """Encapsulates a YOLO model and inference logic."""

    def __init__(
        self,
        model_path: Union[str, Path] = Config.model_path,
        detection_target: Optional[List[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

        # Dynamically set classes to be detected (defaults to Config).
        if detection_target is None:
            detection_target = Config.detection_target
        self.model.set_classes(detection_target)

    def infer(self, image_path: Union[str, Path]):
        """Run prediction on a single image and return the summary dict."""
        results = self.model.predict(source=str(image_path))
        # results[0] is an ultralytics.engine.results.Results
        return results[0].summary()