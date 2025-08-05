import base64
import io
import json
import re
import logging
from typing import Any, Dict, List

from PIL import Image

from .phi4 import Phi4MM

logger = logging.getLogger(__name__)


class LocalPhi4Adapter:
    def __init__(self, **kwargs):
        self.phi = Phi4MM()

    def generate(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Accepts a ready-made messages list, calls the underlying model,
        and parses the JSON response.
        """
        try:
            # The phi.generate method now takes the messages list directly
            response_str = self.phi.generate(messages)
            
            # --- Log the raw response for debugging ---
            logger.info(f"Raw response from local LLM:\n---\n{response_str}\n---")

            # Clean and parse the JSON output from the model
            code_to_parse = response_str.strip()
            if code_to_parse.startswith("```json"):
                code_to_parse = code_to_parse.split("```json", 1)[1].strip()
            if code_to_parse.endswith("```"):
                code_to_parse = code_to_parse.rsplit("```", 1)[0].strip()

            return json.loads(code_to_parse)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM: {e}\nFull Content: {response_str}")
            return {"error": f"JSONDecodeError: {e}"}
        except Exception as e:
            logger.error(f"LLM generation failed in adapter: {e}")
            return {"error": str(e)}

    def chapter_json_from_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        This method is now effectively replaced by the new generate() method,
        but is kept for compatibility in case it's called elsewhere.
        It now just wraps the new generate method.
        """
        return self.generate(messages) 