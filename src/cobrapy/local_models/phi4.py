#!/usr/bin/env python3
# -*- coding: cp1252 -*-
import logging
from typing import Any, Dict, List
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import backoff
from PIL import Image
import base64
from io import BytesIO
import json

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LocalPhi4:
    """
    Wrapper for Google Gemma-3-4B-IT multimodal model.
    """
    def __init__(self, model_name="google/gemma-3-4b-it", **kwargs):
        """
        Initializes the LocalPhi4 model and processor.
        """
        logger.info(f"Loading Gemma-3 4B weights …")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        try:
            # Match the working sanitygemma.py example: use device_map='auto'
            # and let the model determine the best dtype. We will cast inputs later.
            logger.info("Loading model with device_map='auto'...")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto", # More robust than hardcoding 'cuda'
                trust_remote_code=True,
                attn_implementation="sdpa",
                **kwargs
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load Gemma-3 model: {e}")
            raise e

        logger.info("Gemma-3 ready.")

    def _wrap_plain_text(self, messages: List[Dict]) -> List[Dict]:
        wrapped = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                fixed = [
                    {"type": "text", "text": p} if isinstance(p, str) else p
                    for p in content
                ]
                wrapped.append({**m, "content": fixed})
            elif isinstance(content, str):
                wrapped.append({**m,
                                "content": [{"type": "text", "text": content}]})
            else:
                wrapped.append(m)
        return wrapped

    def _normalize_images(self, messages: List[Dict]) -> List[Dict]:
        """
        Converts "image_url" format to the "image" format expected by the
        Gemma-3 processor, handling base64 data URIs and local file paths.
        """
        fixed = []
        for m in messages:
            if isinstance(m.get("content"), list):
                new_parts = []
                for part in m["content"]:
                    # Case 1: Handle local file paths provided with "image" type
                    if part.get("type") == "image" and isinstance(part.get("image"), str) and not part["image"].startswith("http"):
                        try:
                            image = Image.open(part["image"])
                            new_parts.append({"type": "image", "image": image})
                        except Exception as e:
                            logger.error(f"Failed to open image from path: {part['image']}. Error: {e}")
                            # Optionally skip this part or add a placeholder
                            continue
                    # Case 2: Handle "image_url" with base64 data or web URLs
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith('data:image/jpeg;base64,'):
                             _, b64_data = url.split(',', 1)
                             image_data = base64.b64decode(b64_data)
                             image = Image.open(BytesIO(image_data))
                             new_parts.append({"type": "image", "image": image})
                        else: # Assumes it's a web URL
                             new_parts.append({"type": "image", "image": url})
                    # Case 3: Pass other parts through unchanged
                    else:
                        new_parts.append(part)
                fixed.append({**m, "content": new_parts})
            else:
                fixed.append(m)
        return fixed

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, on_giveup=lambda e: print(f"LLM generation failed after multiple retries: {e}"))
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Main generation method. It now accepts the standard messages format.
        """
        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
            **kwargs
        }

        messages = self._wrap_plain_text(messages)
        normalized_messages = self._normalize_images(messages)

        loggable_messages = []
        for m in normalized_messages:
            loggable_content = []
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image":
                        loggable_content.append({"type": "image", "image": "<PIL.Image object>"})
                    else:
                        loggable_content.append(part)
            else:
                loggable_content = m.get("content")
            loggable_messages.append({**m, "content": loggable_content})
        logger.info(f"--- Raw Input to LLM Processor ---\n{json.dumps(loggable_messages, indent=2)}\n---------------------------------")

        inputs = self.processor.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # Ensure input tensors are on the correct device and dtype,
                # matching the model's expectations. This mirrors the working
                # sanitygemma.py script. We cast float tensors to bfloat16.
                if v.is_floating_point():
                    inputs[k] = v.to(self.model.device, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generation_args)
                if outputs.shape[1] <= input_len:
                    logger.warning("Model generated no new tokens.")
                    return ""
                new_tokens = outputs[0][input_len:]
                response = self.processor.decode(new_tokens, skip_special_tokens=True)

            return response.strip()
        except Exception as e:
            logger.error(f"Model generation failed during model.generate(): {e}", exc_info=True)
            raise e

# ---- keep old alias so `from .phi4 import Phi4MM` still works ----
Phi4MM = LocalPhi4