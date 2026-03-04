import json
import logging
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image as PILImage, ImageDraw
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SUPPORTED_ACTIONS = ["grasp", "hand_over"]

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
        self._model_name = model or self.MODEL
        self._client = genai.Client(api_key=api_key)
        self._system_prompt = _SYSTEM_PROMPT.format(
            actions=", ".join(f'"{a}"' for a in SUPPORTED_ACTIONS)
        )
        logger.info("GeminiRoboticsClient ready (model=%s)", self._model_name)

    def decide_task(
        self,
        image_rgb: np.ndarray,
        instruction: str,
    ) -> Optional[TaskDecision]:
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
                    response_mime_type="application/json",
                    max_output_tokens=1024,
                ),
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            return None

        raw_text = (getattr(response, "text", None) or "").strip()
        if not raw_text:
            # Some responses have no direct `.text`; try candidates/parts.
            try:
                candidates = getattr(response, "candidates", None) or []
                text_parts = []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if not parts:
                        continue
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            text_parts.append(part_text)
                raw_text = "\n".join(text_parts).strip()
            except Exception:
                raw_text = ""

        if not raw_text:
            logger.error(
                "Gemini returned no text content. response=%r",
                response,
            )
            return None

        logger.debug("Gemini raw response: %s", raw_text)

        return self._parse_response(raw_text, image_width=w, image_height=h)

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

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    pkg_dir = Path(__file__).resolve().parent
    ws_dir = pkg_dir.parents[2]
    load_dotenv(ws_dir / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set in environment.")
    else:
        client = GeminiRoboticsClient(api_key=api_key)
        image_path = pkg_dir / "water_bottle.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        instruction = "grasp the water bottle"
        image_rgb = np.array(PILImage.open(image_path).convert("RGB"))
        h, w = image_rgb.shape[:2]
        pil_image = PILImage.fromarray(image_rgb)

        decision = client.decide_task(image_rgb=image_rgb, instruction=instruction)

        print("instruction:", instruction)
        print("image:", image_path)
        if decision is None:
            print("parsed_decision: None")
        else:
            x_px, y_px = decision.point
            y_norm = int(round(y_px / h * 1000.0))
            x_norm = int(round(x_px / w * 1000.0))

            print("parsed_output:")
            print("  action:", decision.action)
            print("  label:", decision.label)
            print(f"  point_px [x, y]: [{x_px:.1f}, {y_px:.1f}]")
            print(f"  point_norm [y, x]: [{y_norm}, {x_norm}]")
            print(f"  image_size [w, h]: [{w}, {h}]")

            vis = pil_image.copy()
            draw = ImageDraw.Draw(vis)
            r = 10
            draw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), outline="red", width=3)
            draw.line((x_px - 20, y_px, x_px + 20, y_px), fill="red", width=2)
            draw.line((x_px, y_px - 20, x_px, y_px + 20), fill="red", width=2)
            draw.text((10, 10), f"{decision.action} | {decision.label}", fill="red")
            out_path = pkg_dir / "water_bottle_gemini_vis.jpg"
            vis.save(out_path)
            print("saved_visualization:", out_path)
