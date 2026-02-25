"""
gemini_client.py
Thin wrapper around the google-genai SDK for the Gemini Robotics-ER model.

Single-call design
------------------
One Gemini call receives:
  * The current RGB camera frame.
  * The natural-language instruction from the operator.

And returns a *complete* task payload that can be forwarded directly to
the motion_planner's /motion_task topic:

  {
    "action": "grasp" | "hand_over",   ← Gemini decides
    "point":  [x, y],                  ← absolute pixel, Gemini locates
    "label":  "red cup"                ← human-readable (for logging only)
  }

Coordinate convention
---------------------
Gemini returns points as [y_norm, x_norm] normalised to 0-1000.
This client converts to absolute pixel [x, y] (column, row).
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image as PILImage

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported actions (must match motion_planner/action_types.py registry)
# ---------------------------------------------------------------------------

SUPPORTED_ACTIONS = ["grasp", "hand_over"]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robotic task planner for a mobile manipulator. \
You receive a camera image and a natural-language instruction from the operator, \
and you must output a single JSON object that completely describes the task for \
the robot to execute.

You MUST:
1. Decide the correct ACTION from this fixed list: {actions}.
   - Use "grasp" when the robot should pick up / grab an object.
   - Use "hand_over" when the robot should deliver / pass an object to a person.
2. Visually locate the PRIMARY TARGET OBJECT mentioned in the instruction.
3. Estimate the SINGLE BEST PIXEL to use as the grasp/approach centre for that object.
   A good grasp centre is near the object's centroid on a stable, grippable part.

Output ONLY valid JSON (no markdown, no extra text) in exactly this schema:
{{
  "action": "<action_name>",
  "point": [<y_norm>, <x_norm>],
  "label": "<short object description>"
}}

Where y_norm and x_norm are integers in [0, 1000], normalized so that
(0, 0) = top-left corner and (1000, 1000) = bottom-right corner of the image.

If the target object is NOT visible in the image, return:
{{
  "action": null,
  "point": null,
  "label": "not found"
}}
"""

_USER_TEMPLATE = "Instruction: {instruction}"


@dataclass
class TaskDecision:
    """Structured result from GeminiRoboticsClient.decide_task()."""
    action: str                 # e.g. "grasp"
    point: list[float]          # [x_px, y_px] in absolute pixel coords
    label: str                  # e.g. "red cup"  (for logging / debugging)


class GeminiRoboticsClient:
    """
    Uses Gemini Robotics-ER to decide which action to perform AND to
    visually ground the target object — all in a single API call.

    Parameters
    ----------
    api_key : str
        Google AI API key (GOOGLE_API_KEY).
    model : str, optional
        Gemini model identifier. Defaults to ``gemini-robotics-er-1.5-preview``.
    """

    MODEL = "gemini-robotics-er-1.5-preview"

    def __init__(self, api_key: str, model: Optional[str] = None):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "The 'google-genai' package is required. "
                "Install it with: pip install google-genai"
            )
        self._model_name = model or self.MODEL
        self._client = genai.Client(api_key=api_key)
        self._system_prompt = _SYSTEM_PROMPT.format(
            actions=", ".join(f'"{a}"' for a in SUPPORTED_ACTIONS)
        )
        logger.info("GeminiRoboticsClient ready (model=%s)", self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_task(
        self,
        image_rgb: np.ndarray,
        instruction: str,
    ) -> Optional[TaskDecision]:
        """
        Given the camera frame and a natural-language instruction, ask Gemini
        to decide which action to perform and where the target object is.

        Parameters
        ----------
        image_rgb : np.ndarray
            H×W×3 uint8 RGB image (from cv_bridge).
        instruction : str
            Operator instruction, e.g. ``"grasp the red cup"``.

        Returns
        -------
        TaskDecision | None
            Populated TaskDecision on success, ``None`` on any failure
            (Gemini API error, object not found, invalid response, etc.).
        """
        h, w = image_rgb.shape[:2]
        pil_image = PILImage.fromarray(image_rgb)
        user_text = _USER_TEMPLATE.format(instruction=instruction)

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    pil_image,
                    user_text,
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.0,      # deterministic / no hallucination
                    max_output_tokens=256,
                ),
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            return None

        raw_text = response.text.strip()
        logger.debug("Gemini raw response: %s", raw_text)

        return self._parse_response(raw_text, image_width=w, image_height=h)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw_text: str,
        image_width: int,
        image_height: int,
    ) -> Optional[TaskDecision]:
        """
        Parse the model JSON output into a TaskDecision.

        * Strips optional markdown code fences the model sometimes emits.
        * De-normalises [y_norm, x_norm] ∈ [0,1000] → absolute pixel [x, y].
        """
        # Strip markdown code fences if present.
        text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

        # Attempt JSON parse, with a regex fallback for embedded objects.
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            logger.error("Could not parse JSON from Gemini response: %s", raw_text)
            return None

        action = data.get("action")
        point  = data.get("point")
        label  = data.get("label", "unknown")

        # Object not found case.
        if action is None or point is None:
            logger.info(
                "Gemini could not locate target (label=%s). Object not in view.", label
            )
            return None

        # Validate action.
        if action not in SUPPORTED_ACTIONS:
            logger.error(
                "Gemini returned unknown action '%s'. Supported: %s",
                action, SUPPORTED_ACTIONS,
            )
            return None

        # Validate point.
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            logger.error("Unexpected point format from Gemini: %s", point)
            return None

        y_norm, x_norm = float(point[0]), float(point[1])

        # De-normalise from [0, 1000] → absolute pixels, return as [x, y].
        x_px = x_norm / 1000.0 * image_width
        y_px = y_norm / 1000.0 * image_height

        logger.info(
            "Gemini decision → action='%s', target='%s', pixel=(%.1f, %.1f)",
            action, label, x_px, y_px,
        )
        return TaskDecision(action=action, point=[x_px, y_px], label=label)
