import base64
import io
import json
import re
from typing import Any, Dict, List

from PIL import Image

from .phi4 import Phi4MM


class LocalPhi4Adapter:
    def __init__(self, **kwargs):
        # kwargs is included to catch 'images_per_segment' which is no longer used
        self.phi = Phi4MM()

    def generate(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Accepts a ready-made messages list, converts it to the format
        expected by chapter_json, and calls the underlying model.
        """
        # The helper method unpacks the messages and calls the real model method.
        # The result from the helper is already a dictionary.
        return self.chapter_json_from_chat(messages)

    def chapter_json_from_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Convert a gpt-style messages list back into the arguments required by
        Phi4MM.chapter_json().
        Assumes the message format produced by ChapterGenerator._generate_segment_prompt.
        """
        # 1) Find the text part of the user message to extract context
        user_content = messages[1]["content"]
        text_part_content = ""
        images = []
        for part in user_content:
            if part["type"] == "text":
                text_part_content = part["text"]
            elif part["type"] == "image_url":
                b64_data = part["image_url"]["url"].split(",", 1)[1]
                image_data = base64.b64decode(b64_data)
                images.append(Image.open(io.BytesIO(image_data)).convert("RGB"))

        if not text_part_content:
            raise ValueError("No text part found in user message content")

        # 2) pull times out of the user prompt's text part. Note the en-dash (–).
        m = re.search(r"Context from ([\d.]+)s[–-]([\d.]+)s", text_part_content)
        if not m:
            raise ValueError(f"Could not parse start/end times from user prompt text: {text_part_content[:100]}")
            
        start, end = float(m.group(1)), float(m.group(2))

        # 3) Extract the transcription from the text part
        transcription = text_part_content.split("\n\n", 1)[-1].replace("Visual references:", "").strip()

        # 4) call the real function
        try:
            chapter = self.phi.chapter_json(
                images=images,
                transcription=transcription,
                start=start,
                end=end,
            )
            # The structure from chapter_json is just the chapter dict itself.
            # The caller in ChapterGenerator expects a {"chapters": [chapter_dict]} structure.
            return {"chapters": [chapter]}
        finally:
            # free PIL buffers
            for im in images:
                im.close() 