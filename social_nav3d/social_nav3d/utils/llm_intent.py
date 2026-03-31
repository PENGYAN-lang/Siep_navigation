"""
LLM Intent Engine for SIEP-based social navigation.

Infers pedestrian intent from structured scene descriptions and returns
force parameters that are wired into the SIEP planner as a 5th force channel
(``F_intent``).

Two backends are provided:

1. **Rule-based fallback** (``use_local=False`` or when ``transformers``/
   ``torch`` are unavailable) — fast, zero-GPU heuristics based on speed,
   relative orientation, and explicit behaviour tags.

2. **Local LLM** (``use_local=True``) — uses a causal language model loaded
   via ``transformers`` (e.g. Qwen2.5-7B-Instruct, LLaMA-3-8B-Instruct).
   Requires a GPU environment such as the H800 on the Qizhi platform.

Usage::

    engine = LLMIntentEngine(use_local=False)   # rule-based, no GPU needed
    result = engine.infer_intent(
        "Pedestrian A: pos=(5.2, 3.1), facing=toward_robot, speed=0.1, "
        "dist=2.5, behavior=waving"
    )
    # → {"intent": "calling", "force_direction": "attract", "force_scale": 0.8}
"""
from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Intent label catalogue
# ─────────────────────────────────────────────────────────────────────────────

# Supported intent labels and their human-readable descriptions.
INTENT_LABELS: Dict[str, str] = {
    "calling":   "Pedestrian is actively calling / waving at the robot → approach",
    "curious":   "Pedestrian shows mild interest (facing robot, slow) → gentle approach",
    "gathering": "Pedestrian cluster near an exhibit → guided approach",
    "neutral":   "No strong intent signal → no social force",
    "avoiding":  "Pedestrian trying to stay away → maintain distance",
    "rushing":   "Pedestrian moving fast, ignoring robot → clear the way",
}

# Default result when inference fails.
_DEFAULT_RESULT: Dict[str, object] = {
    "intent": "neutral",
    "force_direction": "none",
    "force_scale": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based inference
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_infer(scene_description: str) -> Dict[str, object]:
    """Heuristic intent inference — no model required.

    Parses the scene description string and applies simple rules:

    * ``behavior=waving``  → "calling" (strong attraction, scale 0.8)
    * facing robot + speed ≤ 0.3  → "curious"  (mild attraction, scale 0.45)
    * group / gathering keyword   → "gathering" (guidance, scale 0.55)
    * walking away + speed ≥ 0.8  → "rushing"  (repulsion, scale 0.7)
    * facing away                 → "avoiding"  (mild repulsion, scale 0.4)
    * otherwise                   → "neutral"

    Args:
        scene_description: Free-text string describing one pedestrian's state.

    Returns:
        dict with keys ``intent``, ``force_direction``, ``force_scale``.
    """
    desc = scene_description.lower()

    # Extract speed if present
    speed = 0.5  # default assumption
    m = re.search(r"speed\s*[=:]\s*([0-9.]+)", desc)
    if m:
        try:
            speed = float(m.group(1))
        except ValueError:
            pass

    # ── Rule 1: explicit waving / calling ──────────────────────────────
    if any(kw in desc for kw in ("waving", "beckoning", "calling", "gesturing")):
        return {"intent": "calling", "force_direction": "attract", "force_scale": 0.8}

    # ── Rule 2: group / gathering ──────────────────────────────────────
    if any(kw in desc for kw in ("gathering", "group", "cluster", "crowd")):
        return {"intent": "gathering", "force_direction": "attract", "force_scale": 0.55}

    # ── Rule 3: facing robot and slow ─────────────────────────────────
    if "toward_robot" in desc or "facing robot" in desc or "toward robot" in desc:
        if speed <= 0.3:
            return {"intent": "curious", "force_direction": "attract", "force_scale": 0.45}

    # ── Rule 4: walking away fast ─────────────────────────────────────
    if ("away" in desc or "leaving" in desc) and speed >= 0.8:
        return {"intent": "rushing", "force_direction": "repel", "force_scale": 0.7}

    # ── Rule 5: facing away ───────────────────────────────────────────
    if any(kw in desc for kw in ("away_from_robot", "facing away", "back_to_robot")):
        return {"intent": "avoiding", "force_direction": "repel", "force_scale": 0.4}

    # ── Default ───────────────────────────────────────────────────────
    return {"intent": "neutral", "force_direction": "none", "force_scale": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Local-LLM inference helper
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are an intent-recognition module for a social navigation robot operating in \
a museum/gallery.  Given a structured description of a nearby pedestrian, output \
a JSON object with exactly three fields:

  "intent"          : one of {calling, curious, gathering, neutral, avoiding, rushing}
  "force_direction" : one of {attract, repel, none}
  "force_scale"     : a float in [0.0, 1.0] indicating force strength

Respond with ONLY valid JSON, no explanation.
"""


def _llm_infer(pipeline, scene_description: str) -> Dict[str, object]:
    """Run inference using a loaded HuggingFace text-generation pipeline.

    Args:
        pipeline:          A ``transformers.pipeline`` object (text-generation).
        scene_description: Structured scene text.

    Returns:
        Parsed intent dict, or ``_DEFAULT_RESULT`` on any failure.
    """
    import json  # local import — only reached when transformers is available

    prompt = (
        f"{_LLM_SYSTEM_PROMPT}\n\nPedestrian description:\n{scene_description}\n\nJSON:"
    )

    try:
        outputs = pipeline(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            temperature=1.0,
            return_full_text=False,
        )
        raw = outputs[0]["generated_text"].strip()

        # Extract first JSON object
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if not m:
            logger.warning("[LLMIntentEngine] No JSON in LLM output: %s", raw)
            return dict(_DEFAULT_RESULT)

        data = json.loads(m.group())

        # Validate fields
        intent = str(data.get("intent", "neutral")).lower()
        if intent not in INTENT_LABELS:
            intent = "neutral"
        direction = str(data.get("force_direction", "none")).lower()
        scale = float(data.get("force_scale", 0.0))
        scale = max(0.0, min(1.0, scale))

        return {"intent": intent, "force_direction": direction, "force_scale": scale}

    except Exception as exc:  # noqa: BLE001
        logger.warning("[LLMIntentEngine] LLM inference failed: %s", exc)
        return dict(_DEFAULT_RESULT)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMIntentEngine:
    """Infers pedestrian intent from structured scene descriptions.

    Provides two backends:

    * ``use_local=True``  — loads a local causal LM via ``transformers``.
      Requires ``torch`` and ``transformers``; gracefully falls back to
      rule-based if either is unavailable.
    * ``use_local=False`` — pure rule-based inference (no GPU, instant).

    Args:
        model_name: HuggingFace model name or local path.
                    Default is ``"Qwen/Qwen2.5-7B-Instruct"``.
        use_local:  Whether to attempt loading a local/HF LLM.
        device:     PyTorch device string (``"cuda"``, ``"cpu"``, etc.).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_local: bool = True,
        device: str = "cuda",
    ) -> None:
        self._pipeline = None
        self._use_local = use_local
        self._model_name = model_name

        if use_local:
            self._pipeline = self._load_pipeline(model_name, device)

    # ------------------------------------------------------------------

    def _load_pipeline(self, model_name: str, device: str):
        """Attempt to load a transformers text-generation pipeline.

        Returns the pipeline on success, or ``None`` on failure (triggering
        automatic fallback to rule-based inference).
        """
        try:
            import torch  # noqa: F401 — imported to check availability
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            logger.info(
                "[LLMIntentEngine] Loading model '%s' on device '%s' …",
                model_name, device,
            )
            pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                device=device,
                torch_dtype="auto",
            )
            logger.info("[LLMIntentEngine] Model loaded successfully.")
            return pipe

        except ImportError:
            logger.warning(
                "[LLMIntentEngine] torch/transformers not available — "
                "falling back to rule-based inference."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[LLMIntentEngine] Failed to load model '%s': %s — "
                "falling back to rule-based inference.",
                model_name, exc,
            )
        return None

    # ------------------------------------------------------------------

    def infer_intent(self, scene_description: str) -> Dict[str, object]:
        """Infer pedestrian intent from a structured text description.

        Args:
            scene_description:
                A single-line or multi-line text describing one pedestrian,
                e.g.::

                    "Pedestrian A: pos=(5.2, 3.1), facing=toward_robot,
                     speed=0.1, dist=2.5, behavior=waving"

        Returns:
            A dict with keys:

            * ``intent`` (str): intent label from ``INTENT_LABELS``.
            * ``force_direction`` (str): ``"attract"``, ``"repel"``, or ``"none"``.
            * ``force_scale`` (float): magnitude in [0, 1].
        """
        if self._pipeline is not None:
            return _llm_infer(self._pipeline, scene_description)
        return _rule_based_infer(scene_description)

    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return a human-readable string identifying the active backend."""
        if self._pipeline is not None:
            return f"local-llm({self._model_name})"
        return "rule-based"
