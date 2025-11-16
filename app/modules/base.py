from typing import Any, Dict, Optional
import numpy as np


class BaseModule:
    """Base interface for pluggable AI modules.

    Subclass this and implement load/infer/unload. draw is optional.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self.loaded: bool = False

    def load(self) -> None:
        """Load model/resources; set self.loaded=True when done."""
        self.loaded = True

    def unload(self) -> None:
        """Release resources; set self.loaded=False when done."""
        self.loaded = False

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single BGR frame. Return results dict."""
        if not self.loaded:
            raise RuntimeError(f"Module {self.name} not loaded")
        return {"module": self.name, "ok": True}

    def draw(self, frame_bgr: np.ndarray, results: Dict[str, Any]) -> None:
        """Optionally overlay results to the frame in place."""
        return

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        """Infer, draw and postprocess in one step."""
        return
