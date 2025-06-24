#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import random
import re
import time
import warnings
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          BitsAndBytesConfig)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0  # Set to 0.0 for deterministic output as per original project

LOAD_IN_4BIT = False
USE_FLASH_ATTENTION_2 = True

# ── Retry / prompt-size limits ────────────────────────────
MAX_SNIP_CHARS = 350
MAX_SNIP_CHARS_RETRY = 180
MAX_IMAGES_FIRST = 4
MAX_IMAGES_RETRY = 3
MAX_RETRIES = 3

# --- ADDED: Shot types list from ChapterGenerator ---
SHOT_TYPES: List[str] = [
    "Establishing Shot", "Extreme Wide Shot (EWS)", "Wide Shot (WS)", "Full Shot (FS)",
    "Medium Wide Shot (MWS)", "Medium Long Shot (MLS)", "Medium Shot (MS)", "Cowboy Shot",
    "Medium Close-Up (MCU)", "Close-Up (CU)", "Extreme Close-Up (ECU)", "Two-Shot",
    "Three-Shot", "Reverse Angle", "Over-the-Shoulder", "Point-of-View (POV) Shot",
    "Reaction Shot", "Insert Shot", "Cutaway", "Dutch Angle", "Tracking/Dolly Shot",
    "Crane/Jib Shot", "Handheld/Steadicam Shot", "Whip-Pan (Swish-Pan)", "Special / Other"
]

FORBIDDEN_WORDS = "video|clip|footage|camera|audio|transcription|segment|scene|frame|image|viewer"
_FORBIDDEN_RE = re.compile(rf"\b({FORBIDDEN_WORDS})s?\b", re.I)

# ---------------------------------------------------------------------
# Parsing & Validation Helpers
# ---------------------------------------------------------------------
def extract_last_json_block(text: str) -> str | None:
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i+1]
    return None

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def dbg(tag: str, txt: str) -> None:
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"── {tag} ─────────────────────────────────────────\n{txt}\n── end {tag} ──")


# ---------------------------------------------------------------------
# Phi-4 multimodal wrapper
# ---------------------------------------------------------------------
class Phi4MM:
    """Loads Phi-4 with 4-bit quantisation and handles generation with proper dtype management."""
    def __init__(self, fallback_to_float16: bool = True):
        self.fallback_to_float16 = fallback_to_float16
        self.model = None
        self.processor = None
        self.quantized = False
        self.compute_dtype = torch.float16  # Consistent compute dtype
        
        # Try loading with 4-bit first, fallback if needed
        self._load_model()

    def _load_model(self) -> None:
        """Load model with quantization, fallback to float16 if 4-bit fails."""
        logging.info("Loading Phi-4 weights …")

        # Load processor first
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True
        )

        # Suppress known warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        if LOAD_IN_4BIT and DEVICE == "cuda":
            try:
                self._load_quantized()
                self.quantized = True
                logging.info("Phi-4 loaded with 4-bit quantization.")
            except Exception as e:
                logging.warning("4-bit quantization failed: %s", str(e))
                if self.fallback_to_float16:
                    logging.info("Falling back to float16...")
                    self._load_float16()
                    self.quantized = False
                else:
                    raise
        else:
            self._load_float16()
            self.quantized = False

        # Configure model for inference
        self.model.eval()
        
        # Disable LoRA adapter switching for 4-bit models
        if self.quantized and hasattr(self.model, "set_lora_adapter"):
            logging.info("Disabling dynamic LoRA adapter switching (4-bit inference).")
            # Create a no-op function to replace the method
            def noop_lora(*args, **kwargs):
                pass
            self.model.set_lora_adapter = noop_lora

        logging.info("Phi-4 ready.")

    def _load_quantized(self) -> None:
        """Load model with 4-bit quantization using optimized settings."""
        # Optimized quantization config for dtype compatibility
        q_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,  # Consistent dtype
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8,
        )

        model_kwargs = {
            "torch_dtype": self.compute_dtype,  # Consistent with compute dtype
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "quantization_config": q_cfg,
        }

        # Add flash attention if available
        if USE_FLASH_ATTENTION_2:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logging.info("Using Flash Attention 2")
            except Exception:
                logging.warning("Flash Attention 2 not available, using default attention")

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    def _load_float16(self) -> None:
        """Load model in float16 without quantization."""
        model_kwargs = {
            "torch_dtype": self.compute_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # Add flash attention if available
        if USE_FLASH_ATTENTION_2 and DEVICE == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logging.info("Using Flash Attention 2")
            except Exception:
                logging.warning("Flash Attention 2 not available, using default attention")

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    def _clean_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub forbidden words from specified fields in the output JSON."""
        for fld in ("shotDescription", "summary"):
            if fld in d and isinstance(d[fld], str):
                # Replace forbidden word and any trailing 's' with an empty string, then clean up whitespace
                cleaned_text = _FORBIDDEN_RE.sub("", d[fld])
                d[fld] = ' '.join(cleaned_text.split())
        return d

    def chapter_json(
        self,
        images: List[Image.Image],
        transcription: str,
        start: float,
        end: float,
    ) -> Dict[str, Any]:
        attempt = 0
        last_error = None  # Initialize last_error
        while attempt <= MAX_RETRIES:
            is_retry = attempt > 0
            # On retry shrink images + audio snippet
            img_cap = MAX_IMAGES_RETRY if is_retry else MAX_IMAGES_FIRST
            images_subset = random.sample(images, k=min(img_cap, len(images)))

            try:
                result = self._generate_chapter_json(
                            images_subset, transcription, start, end,
                            retry_mode=is_retry, last_error=last_error)
                if result:
                    return result
            except Exception as e:
                logging.warning("Attempt %d failed: %s", attempt + 1, e)
                last_error = str(e)

            attempt += 1
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            time.sleep(1)

        logging.error("All attempts failed for chapter %.1f–%.1f", start, end)
        return self._create_fallback_json(start, end)

    def _generate_chapter_json(
        self,
        images: List[Image.Image],
        transcription: str,
        start: float,
        end: float,
        retry_mode: bool = False,
        last_error: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        
        header = "<|user|>\n" + "".join(f"<|image_{i+1}|>\n" for i in range(len(images)))
        
        snip_limit = MAX_SNIP_CHARS_RETRY if retry_mode else MAX_SNIP_CHARS
        audio_snip = (transcription[:snip_limit] + "…"
              if len(transcription) > snip_limit else transcription or "No transcription available.")

        shot_types_str = ", ".join([f'"{s}"' for s in SHOT_TYPES])

        # NEW, stricter prompt based on user feedback
        prompt = f"""{header}
You are a cinematic analyst. Your task is to write a single, detailed, and evocative chapter for a JSON object based on the provided visual and audible context for the time range {start:.3f}s to {end:.3f}s.

🚫 **ABSOLUTELY FORBIDDEN WORDS**
You must **NEVER** use any of the following terms or their plurals in your response: {FORBIDDEN_WORDS}.
Also avoid meta-phrases like "the chapter shows" or "the scene opens with".

✅ **STYLE GUIDELINES**
- Write in the **PRESENT tense**, third-person narrative.
- Begin your summary directly with the main subject or action (e.g., "A man in a sharp grey suit sits…").
- Blend what is visible and audible into a single, flowing cinematic paragraph.
- Infer a non-neutral **sentiment**, up to 3 nuanced **emotions**, and a concise **theme** (≤3 words). Do not default to "neutral".

✅ **JSON OUTPUT**
Return **only** a valid JSON object for a single chapter. Use the exact start/end times provided. The `summary` must be a rich, narrative paragraph.

```json
{{
  "chapters": [
    {{
      "start": "{start:.3f}s",
      "end": "{end:.3f}s",
      "shotType": ["Sample Shot Type"],
      "shotDescription": "A concise, present-tense description of the camera work and composition.",
      "sentiment": "inferred sentiment",
      "emotions": ["inferred", "list", "of emotions"],
      "theme": "inferred theme",
      "summary": "This is where the rich, narrative summary goes. It should be a flowing paragraph describing the action, setting, and dialogue as if it is happening now."
    }}
  ]
}}
```

---
**Audible context for this segment:**
"{audio_snip}"

**Your task:** Adhere strictly to all instructions and return only the JSON object.
<|end|>
<|assistant|>"""
        
        if retry_mode:
            retry_header = (
                "Your previous response was invalid. Please try again. "
                f"The error was: '{last_error}'.\n"
                "Review ALL instructions, especially about providing a complete, valid JSON with no empty fields. Follow all rules precisely.\n\n"
            )
            prompt = retry_header + prompt

        dbg("PROMPT", prompt)

        # Process inputs
        try:
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # Ensure all tensors have consistent dtype and device
            processed_inputs = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    # Move to device first
                    value = value.to(DEVICE)
                    
                    # Handle different tensor types appropriately
                    if key in ["input_ids", "attention_mask"]:
                        # These should remain as long tensors
                        if value.dtype != torch.long:
                            value = value.long()
                    elif key == "pixel_values":
                        # Pixel values should match compute dtype
                        if value.dtype != self.compute_dtype:
                            value = value.to(dtype=self.compute_dtype)
                    else:
                        # Other tensors should match compute dtype
                        if value.dtype != self.compute_dtype and value.dtype != torch.long:
                            value = value.to(dtype=self.compute_dtype)
                    
                    processed_inputs[key] = value
                else:
                    processed_inputs[key] = value

        except Exception as e:
            logging.error("Input processing failed: %s", str(e))
            raise

        # Generation
        try:
            with torch.inference_mode():
                gen_kwargs = {
                    **processed_inputs,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": False,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,
                    "return_dict_in_generate": False,
                }
                
                if TEMPERATURE > 0.0:
                    gen_kwargs["temperature"] = TEMPERATURE
                    gen_kwargs["do_sample"] = True

                out_ids = self.model.generate(**gen_kwargs)
        except Exception as e:
            logging.error("Model generation failed: %s", str(e))
            raise

        # Decode and Parse
        try:
            gen_tokens = out_ids[:, processed_inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(
                gen_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            logging.info(f"LLM Raw Response:\n{response}")

            payload = extract_last_json_block(response)
            if payload is None:
                raise ValueError("No JSON object found in response")
                
            parsed_json = json.loads(payload)

            # The prompt asks for a "chapters" key with one item.
            if "chapters" in parsed_json and isinstance(parsed_json["chapters"], list) and parsed_json["chapters"]:
                # Return the first chapter object after cleaning it
                chapter = parsed_json["chapters"][0]
                return self._clean_fields(chapter)
            else:
                raise ValueError("LLM response did not contain the expected chapter structure.")

        except (json.JSONDecodeError, Exception) as e:
            logging.error("JSON parsing/validation failed: %s. Raw response: %s", str(e), response[:200])
            raise

    def _create_fallback_json(self, start: float, end: float) -> Dict[str, Any]:
        """Create a basic fallback JSON when generation fails."""
        return {
            "start": f"{start:.3f}s",
            "end": f"{end:.3f}s",
            "shotType": ["fallback"],
            "shotDescription": "Model generation failed.",
            "sentiment": "unknown",
            "emotions": ["unknown"],
            "theme": "unknown",
            "summary": "Failed to generate summary."
        } 