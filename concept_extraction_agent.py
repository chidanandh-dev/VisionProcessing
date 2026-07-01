"""
Concept Extraction using Google Gemini Vision

Pipeline
--------
  Single step - Gemini Vision (gemini-2.0-flash)
           Reads the uploaded image directly and extracts key concepts.
           No separate OCR step needed.

Required .env variables
-----------------------
    GEMINI_API_KEY   Your Google Gemini API key (free tier works)
"""
import json
import base64
import os
from io import BytesIO

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Models tried in order; each has a separate free-tier quota pool
_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]

_PROMPT = (
    "You are an expert educational content analyzer. Look at this image of student notes "
    "and identify the key educational concepts.\n\n"
    "Extract 3-8 key concepts from these notes. For each concept provide:\n"
    '- "id": "concept_1", "concept_2", etc.\n'
    '- "name": short concept name (2-5 words)\n'
    '- "summary": one clear educational sentence explaining the concept\n'
    '- "category": the subject area (e.g., Biology, Physics, Chemistry, Mathematics, History, etc.)\n\n'
    "Respond with ONLY a valid JSON array. No markdown, no explanation, no code blocks.\n\n"
    'Example: [{"id": "concept_1", "name": "Natural Selection", "summary": "Process where organisms better adapted to their environment survive and reproduce.", "category": "Biology"}]'
)


class ConceptExtractor:
    """Concept extractor using Gemini Vision (free tier)."""

    @staticmethod
    def _get_demo_concepts():
        return {
            "concepts": [
                {
                    "id": "concept_1",
                    "name": "Photosynthesis",
                    "summary": "Process by which green plants convert sunlight into energy using chlorophyll.",
                    "category": "Biology",
                    "region": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.33},
                },
                {
                    "id": "concept_2",
                    "name": "Chlorophyll",
                    "summary": "Green pigment in plant cells that absorbs sunlight for photosynthesis.",
                    "category": "Biology",
                    "region": {"x1": 0.05, "y1": 0.35, "x2": 0.95, "y2": 0.63},
                },
                {
                    "id": "concept_3",
                    "name": "Glucose",
                    "summary": "A simple sugar produced as the output of photosynthesis, used for plant energy.",
                    "category": "Chemistry",
                    "region": {"x1": 0.05, "y1": 0.65, "x2": 0.95, "y2": 0.95},
                },
            ],
            "source": "demo",
        }

    @staticmethod
    def _extract_with_gemini(img_bytes: bytes) -> list:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set in .env")

        client = genai.Client(api_key=GEMINI_API_KEY)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

        last_error = None
        for model in _MODELS:
            try:
                print(f"[Gemini Vision] Trying {model}...")
                response = client.models.generate_content(
                    model=model,
                    contents=[image_part, _PROMPT],
                )
                raw = response.text.strip()
                print(f"[Gemini Vision] Response from {model}: {len(raw)} chars")

                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

                parsed = json.loads(raw)
                print(f"[Gemini Vision] ✓ {len(parsed)} concepts via {model}")
                return parsed

            except json.JSONDecodeError:
                raise  # Bad JSON is not a quota issue — don't retry
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[Gemini Vision] {model} quota exhausted, trying next...")
                    last_error = e
                    continue
                raise  # Re-raise anything that isn't a quota error

        raise RuntimeError(
            f"All Gemini models quota exhausted. Tried: {', '.join(_MODELS)}. "
            "Wait a minute and retry, or check https://ai.dev/rate-limit"
        ) from last_error

    @staticmethod
    def _assign_regions(concepts: list) -> list:
        n = max(len(concepts), 1)
        step = 1.0 / n
        for i, c in enumerate(concepts):
            if "region" not in c:
                c["region"] = {
                    "x1": 0.02,
                    "y1": round(i * step, 4),
                    "x2": 0.98,
                    "y2": round((i + 1) * step, 4),
                }
        return concepts

    @staticmethod
    def extract_concepts_from_highlighted_region(image_data, highlight_box=None, use_demo=False):
        if use_demo:
            print("[Concept Extraction] Demo mode")
            return ConceptExtractor._get_demo_concepts()

        try:
            print("[Concept Extraction] Starting Gemini Vision pipeline...")

            image_bytes = base64.b64decode(image_data)
            print(f"[Concept Extraction] Image decoded: {len(image_bytes)} bytes")

            image = Image.open(BytesIO(image_bytes))
            image.load()
            print(f"[Concept Extraction] Image size: {image.size}")

            if highlight_box:
                image = image.crop(highlight_box)
                print(f"[Concept Extraction] Cropped to: {image.size}")

            buf = BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            concepts_list = ConceptExtractor._extract_with_gemini(img_bytes)

            concepts_list = ConceptExtractor._assign_regions(concepts_list)
            for i, c in enumerate(concepts_list):
                c.setdefault("id", f"concept_{i + 1}")
                c.setdefault("category", "General")
                c.setdefault("summary", c.get("description", ""))

            print(f"[Concept Extraction] ✓ {len(concepts_list)} concepts via Gemini Vision")
            return {"concepts": concepts_list, "source": "gemini"}

        except Exception as e:
            print(f"[Concept Extraction] ✗ {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
