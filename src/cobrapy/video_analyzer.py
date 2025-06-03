import os
import json
import time
import asyncio
import nest_asyncio
from typing import Union, Type, Optional, List, Dict, Set, Any, Coroutine, Tuple, ClassVar
from pydantic import PrivateAttr, BaseModel, Field
from openai import AzureOpenAI, AsyncAzureOpenAI, AsyncOpenAI
import openai # ADDED
import logging
from collections import defaultdict
import numpy as np
import math
import re
import tiktoken
from string import Formatter
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from .cobra_utils import seconds_to_iso8601_duration # Adjust import path if needed
import csv # Added for TSV writing
from urllib.parse import urlparse, urlunparse # Added for URL manipulation
from PIL import Image # ADDED: For image dimension reading
from math import ceil # ADDED: For tiles_needed helper
from pathlib import Path
import hashlib
from functools import lru_cache
import base64 # Added for new helper
import cv2 # Added for new helper
from azure.identity import DefaultAzureCredential, get_bearer_token_provider # Keep existing Azure identity imports

# --- ADDED: Single-tariff constants for GPT-4o/Vision ---
GPT4O_BASE      = 85      # tokens per image – low‑detail     (also the "base" for high‑detail)
GPT4O_PER_TILE  = 170     # tokens per 512 × 512 tile – high‑detail


def tiles_needed(w: int, h: int, tile: int = 512) -> int:
    return ceil(w / tile) * ceil(h / tile)

_TOKEN_USAGE_TEMPLATE = {
    "prompt_tokens": 0, # Will be populated by _finalize_token_usage
    "completion_tokens": 0, # Will be populated by _finalize_token_usage
    "image_tokens": 0, # Will be populated by _finalize_token_usage
    "total_tokens": 0, # Will be populated by _finalize_token_usage
    # Removed chapter and tag specific token buckets as they are not separately populated
    "summary_prompt_tokens": 0,
    "summary_completion_tokens": 0,
    "summary_image_tokens": 0, # Correctly 0 if summary prompt has no images
    "reid_linking_prompt_tokens": 0,
    "reid_linking_completion_tokens": 0,
    "reid_linking_image_tokens": 0,
    "segment_generic_analysis_prompt_tokens": 0,
    "segment_generic_analysis_completion_tokens": 0,
    "segment_generic_analysis_image_tokens": 0,
    "yolo_describe_prompt_tokens": 0,
    "yolo_describe_completion_tokens": 0,
    "yolo_describe_image_tokens": 0,
    "action_extraction_prompt_tokens": 0,
    "action_extraction_completion_tokens": 0,
    "action_extraction_image_tokens": 0,
    "report_input_text_tokens": 0,
    "report_output_text_tokens": 0,
    "report_input_image_tokens": 0,
    "report_total_tokens": 0,
}


def estimate_image_tokens(messages: list[dict]) -> int:
    """
    Reproduce Azure's billing formula for GPT‑4o / GPT‑4 Vision.

        • low‑detail  →  85 tokens flat
        • high‑detail →  85 + 170 × N_tiles   (N_tiles = ceil(w/512)*ceil(h/512))

    width / height are optional extra keys you attach when you build the
    prompt. If they are missing we assume one 512² tile.
    """
    total = 0
    for m in messages:
        content = m.get("content") or []
        # Ensure content is iterable and treat as list if it's a single dict
        if isinstance(content, dict): 
            content = [content]
        if not isinstance(content, list):
            continue 
            
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                img_url_data = part.get("image_url") or {}
                detail = (img_url_data.get("detail") or "low").lower()

                if detail == "high":
                    # Use provided width/height, fallback to 512 if missing
                    w = int(img_url_data.get("width",  512))
                    h = int(img_url_data.get("height", 512))
                    total += GPT4O_BASE + GPT4O_PER_TILE * tiles_needed(w, h)
                else:  # "low" or any other unspecified detail
                    total += GPT4O_BASE
    return total
# --- END UPDATED ---

from .models.video import VideoManifest, Segment
from .models.environment import CobraEnvironment
from .analysis import AnalysisConfig
from .analysis.base_analysis_config import SequentialAnalysisConfig
from .cobra_utils import (
    encode_image_base64,
    validate_video_manifest,
    write_video_manifest,
    ensure_person_group_exists,
    process_frame_with_faces,
    seconds_to_iso8601_duration,
    upload_blob, # Ensure upload_blob is imported
    get_frame_crop_base64, # IMPORTED
    estimate_image_tokens # ADDED: Import from cobra_utils
)

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    manifest: VideoManifest
    env: CobraEnvironment
    reprocess_segments: bool
    person_group_id: Optional[str] # Keep if used for other face features
    peoples_list_path: Optional[str]
    peoples_list: Optional[dict]
    emotions_list_path: Optional[str]
    emotions_list: Optional[dict]
    objects_list_path: Optional[str]
    objects_list: Optional[dict]
    themes_list_path: Optional[str]
    themes_list: Optional[dict]
    actions_list_path: Optional[str]
    actions_list: Optional[dict]
    video_blob_url: Optional[str] # ADDED: Store input video blob URL
    MAX_FRAMES_PER_PROMPT: int = 45  # Maximum number of frames to send in a single prompt
    token_usage: Dict[str, int]  # UPDATED structure
    TAG_CHUNK_FRAME_COUNT: int = 5
    image_detail_level: str = "low" # ADDED: Configurable image detail level
    active_yolo_tags_source: Optional[List[Dict[str, Any]]] # ADDED: To store active YOLO tags for current analysis
    using_refined_tags: bool # ADDED: Flag to indicate if refined tags are active
    ACTION_EXTRACTION_FRAME_CHUNK_SIZE: int = 10 # New constant for action extraction

    # Pricing info dictionary (based on user input, with assumptions)
    PRICING_INFO: Dict[str, Dict[str, float]] = {
        # NOTE: Image token pricing is NOT specified by the user.
        # Calculated cost will ONLY reflect text input/output tokens.
        "gpt-4.1": {"input_per_million": 2.00, "output_per_million": 8.00},
        "gpt-4.1-mini": {"input_per_million": 0.40, "output_per_million": 1.60},
        "gpt-4.1-nano": {"input_per_million": 0.10, "output_per_million": 0.40},
    }


    # Add instance variables to track known tags during analysis
    _current_known_persons: Set[str]
    _current_known_actions: Set[str]
    _current_known_objects: Set[str]

    # --- Blob Storage Attributes (NEW) ---
    # Moved from VideoPreProcessor to be accessible here for cleanup of segment frames
    # if they were uploaded to blob by VideoPreProcessor.
    _blob_service_client: Optional[BlobServiceClient] = PrivateAttr(default=None)
    _temp_frame_blob_names_by_segment: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    # REMOVED: _async_llm_client: Any = PrivateAttr(default=None) # Stores the async client instance


    def __init__(
        self,
        video_manifest: Union[str, VideoManifest],
        env: CobraEnvironment,
        # --- ADDED video_blob_url ---
        video_blob_url: Optional[str] = None,
        # --- END ADDED ---
        person_group_id: Optional[str] = None,
        peoples_list_path: Optional[str] = None,
        emotions_list_path: Optional[str] = None,
        objects_list_path: Optional[str] = None,
        themes_list_path: Optional[str] = None,
        actions_list_path: Optional[str] = None,
    ):
        # get and validate video manifest
        if isinstance(video_manifest, str):
            self.manifest = validate_video_manifest(video_manifest)
        elif isinstance(video_manifest, VideoManifest):
            self.manifest = video_manifest
        else:
            raise ValueError(
                "video_manifest must be a string path to a JSON file or a VideoManifest object"
            )

        self.env = env
        self.video_blob_url = video_blob_url # Store video blob URL
        self._internal_async_client: Optional[Any] = None # ADDED: Regular instance attribute for the async client

        # MODIFIED: Use _TOKEN_USAGE_TEMPLATE directly here for consistency
        self.token_usage = _TOKEN_USAGE_TEMPLATE.copy()
        
        self.person_group_id = person_group_id # Keep for potential future use

        # Initialize custom lists
        self.peoples_list_path = peoples_list_path

        # Load peoples list if provided
        self.peoples_list = self._load_json_list(peoples_list_path, "persons") # Corrected key to persons

        # Load emotions list if provided
        self.emotions_list = self._load_json_list(emotions_list_path, "emotions")

        # Load objects list if provided
        self.objects_list = self._load_json_list(objects_list_path, "objects")

        # Load themes list if provided
        self.themes_list = self._load_json_list(themes_list_path, "themes")

        # Load actions list if provided
        self.actions_list_path = actions_list_path
        self.actions_list = self._load_json_list(actions_list_path, "actions") if actions_list_path else None
        self.video_blob_url = video_blob_url # ADDED

        # Initialize current known tags (instance specific for prompting within one analysis run)
        self._current_known_persons = set()
        self._current_known_actions = set()
        self._current_known_objects = set()

        # Optionally pre-populate from lists if provided
        # --- FIX: Use correct key from loaded list data ---
        if self.peoples_list:
             people_items = self.peoples_list.get("persons", []) # Use \'persons\' key
             if isinstance(people_items, list):
                 self._current_known_persons.update(item.get("label") or item.get("name") for item in people_items if isinstance(item, dict)) # Use label or name
        if self.actions_list:
             action_items = self.actions_list.get("actions", [])
             if isinstance(action_items, list):
                 self._current_known_actions.update(item.get("label") or item.get("name") for item in action_items if isinstance(item, dict))
        if self.objects_list:
             object_items = self.objects_list.get("objects", [])
             if isinstance(object_items, list):
                 self._current_known_objects.update(item.get("label") or item.get("name") for item in object_items if isinstance(item, dict))
        # --- END FIX ---

        # --- HYBRID YOLO: Load yolo_tags from _yolo.json if present (fallback) ---
        # The primary source of YOLO tags will now be manifest.raw_yolo_tags or manifest.refined_yolo_tags
        # set by the main pipeline. This __init__ section is a fallback if those are not present.
        yolo_json = None
        try:
            video_path_obj = Path(self.manifest.source_video.path) 
            yolo_file_stem = f"{video_path_obj.stem}_yolo.json"
            
            path3 = None # Default to None
            if self.manifest.processing_params and self.manifest.processing_params.output_directory:
                path3 = Path(self.manifest.processing_params.output_directory) / yolo_file_stem

            potential_paths_to_check = []
            if path3: potential_paths_to_check.append(path3) # Most likely path from preprocessing
            # Add other potential paths if necessary, e.g., next to video file
            # potential_paths_to_check.append(video_path_obj.parent / yolo_file_stem)
            
            yolo_file_found_path = None
            for p_path in potential_paths_to_check:
                if p_path and p_path.exists():
                    yolo_file_found_path = p_path
                    break
            
            if yolo_file_found_path:
                with open(yolo_file_found_path, "r", encoding="utf-8") as f:
                    yolo_json = json.load(f)
                    # Prioritize direct 'yolo_tags' if present, else actionSummary.object (old format)
                    if isinstance(yolo_json, list): # Direct list of tags (newer format)
                        self.manifest.yolo_tags = yolo_json
                    elif isinstance(yolo_json, dict) and "objects" in yolo_json: # Flat structure
                         self.manifest.yolo_tags = yolo_json.get("objects", [])
                    elif isinstance(yolo_json, dict) and "actionSummary" in yolo_json and "object" in yolo_json.get("actionSummary", {}):
                        self.manifest.yolo_tags = yolo_json["actionSummary"]["object"]
                    else:
                        self.manifest.yolo_tags = []
                    logger.info(f"Loaded fallback YOLO tags from {yolo_file_found_path}. Count: {len(self.manifest.yolo_tags)}")
            else:
                self.manifest.yolo_tags = [] # Ensure it's an empty list if no file found

        except Exception as e:
            logger.warning(f"Could not load fallback YOLO tags. Error: {e}")
            self.manifest.yolo_tags = []
        # --- END HYBRID YOLO FALLBACK LOADING ---

        # --- ADDED: Initialize token_usage with Re-ID tokens from manifest ---
        if hasattr(self.manifest, 'initial_prompt_tokens') and self.manifest.initial_prompt_tokens is not None:
            self.token_usage["reid_linking_prompt_tokens"] += self.manifest.initial_prompt_tokens
            self.token_usage["report_input_text_tokens"] += self.manifest.initial_prompt_tokens
        if hasattr(self.manifest, 'initial_completion_tokens') and self.manifest.initial_completion_tokens is not None:
            self.token_usage["reid_linking_completion_tokens"] += self.manifest.initial_completion_tokens
            self.token_usage["report_output_text_tokens"] += self.manifest.initial_completion_tokens
        if hasattr(self.manifest, 'initial_image_tokens') and self.manifest.initial_image_tokens is not None:
            # Note: initial_image_tokens from manifest are already *estimated* image tokens.
            # The reid_linking_image_tokens bucket and report_input_image_tokens should store these.
            self.token_usage["reid_linking_image_tokens"] += self.manifest.initial_image_tokens
            self.token_usage["report_input_image_tokens"] += self.manifest.initial_image_tokens
        
        # Update total_tokens and report_total_tokens based on these additions
        self.token_usage["total_tokens"] = (
            self.token_usage.get("prompt_tokens", 0) +
            self.token_usage.get("completion_tokens", 0) # Base total is text only for now in this template
        )
        # report_total_tokens should reflect everything, including Re-ID.
        # The _call_llm methods update report_total_tokens with API's total_tokens.
        # Here, we are adding Re-ID tokens which were not part of an API call made by *this* class.
        # Let's sum them up into report_total_tokens.
        self.token_usage["report_total_tokens"] = (
            self.token_usage.get("report_input_text_tokens", 0) +
            self.token_usage.get("report_output_text_tokens", 0) +
            self.token_usage.get("report_input_image_tokens", 0)
        )
        if self.manifest.initial_prompt_tokens or self.manifest.initial_completion_tokens or self.manifest.initial_image_tokens:
            logger.info(f"Initialized VideoAnalyzer.token_usage with Re-ID tokens from manifest: "
                        f"P={self.manifest.initial_prompt_tokens or 0}, "
                        f"C={self.manifest.initial_completion_tokens or 0}, "
                        f"I={self.manifest.initial_image_tokens or 0}.")
            logger.debug(f"Updated report totals after Re-ID init: InputText={self.token_usage['report_input_text_tokens']}, OutputText={self.token_usage['report_output_text_tokens']}, InputImage={self.token_usage['report_input_image_tokens']}, Total={self.token_usage['report_total_tokens']}")
        # --- END ADDED ---

    def _load_json_list(self, file_path, expected_key):
        """Helper method to load and validate JSON list files."""
        if not file_path or not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if the file has the expected structure
                if isinstance(data, dict) and expected_key in data:
                    return data
                else:
                    print(f"Warning: {expected_key} list file doesn't contain expected format. Expected '{expected_key}' key.")
                    return None
        except Exception as e:
            print(f"Error loading {expected_key} list: {str(e)}")
            return None

    # Primary method to analyze the video
    async def analyze_video(
        self,
        analysis_config: Type[AnalysisConfig],
        run_async=False,
        max_concurrent_tasks=None,
        reprocess_segments=False,
        person_group_id=None,
        copyright_json_str: Optional[str] = None,
        **kwargs,
    ):
        # Reset known tags at the beginning of each analysis run
        self._current_known_persons = set()
        self._current_known_actions = set()
        self._current_known_objects = set()
        # Re-populate from lists if they exist
        # --- FIX: Use correct key from loaded list data ---
        if self.peoples_list:
             people_items = self.peoples_list.get("persons", []) # Use \'persons\' key
             if isinstance(people_items, list):
                 self._current_known_persons.update(item.get("label") or item.get("name") for item in people_items if isinstance(item, dict)) # Use label or name
        if self.actions_list:
             action_items = self.actions_list.get("actions", [])
             if isinstance(action_items, list):
                 self._current_known_actions.update(item.get("label") or item.get("name") for item in action_items if isinstance(item, dict))
        if self.objects_list:
             object_items = self.objects_list.get("objects", [])
             if isinstance(object_items, list):
                 self._current_known_objects.update(item.get("label") or item.get("name") for item in object_items if isinstance(item, dict))
        # --- END FIX ---


        self.reprocess_segments = reprocess_segments
        self.person_group_id = person_group_id

        # --- SET ACTIVE YOLO TAGS SOURCE --- 
        self.active_yolo_tags_source = None # Reset before check
        self.using_refined_tags = False # Reset before check

        if hasattr(self.manifest, 'refined_yolo_tags') and self.manifest.refined_yolo_tags:
            self.active_yolo_tags_source = self.manifest.refined_yolo_tags
            self.using_refined_tags = True
            logger.info(f"Using refined YOLO tags for analysis (count: {len(self.active_yolo_tags_source)}).")
        elif hasattr(self.manifest, 'raw_yolo_tags') and self.manifest.raw_yolo_tags:
            self.active_yolo_tags_source = self.manifest.raw_yolo_tags
            self.using_refined_tags = False # Explicitly false
            logger.info(f"Using raw YOLO tags for analysis (count: {len(self.active_yolo_tags_source)}).")
        elif self.manifest.yolo_tags: # Fallback to yolo_tags loaded in __init__
            self.active_yolo_tags_source = self.manifest.yolo_tags
            self.using_refined_tags = False # Explicitly false for this fallback
            logger.info(f"Using fallback YOLO tags (from _yolo.json) for analysis (count: {len(self.active_yolo_tags_source)}).")
        else:
            logger.info("No YOLO tags (refined, raw, or fallback) available for this analysis run.")
            self.active_yolo_tags_source = [] # Ensure it\'s an empty list if no source
        # --- END SET ACTIVE YOLO TAGS SOURCE ---

        # --- MODIFIED: Reset token usage and re-apply initial (Re-ID) tokens ---
        self.token_usage = _TOKEN_USAGE_TEMPLATE.copy() # Reset all to 0 initially for this run

        # Re-apply initial tokens from manifest (e.g., Re-ID tokens loaded in __init__ or passed)
        # This ensures they are not lost after the reset above.
        if hasattr(self.manifest, 'initial_prompt_tokens') and self.manifest.initial_prompt_tokens is not None:
            self.token_usage["reid_linking_prompt_tokens"] += self.manifest.initial_prompt_tokens
            # Add to report_input_text_tokens as well, assuming initial_prompt_tokens are text-based
            self.token_usage["report_input_text_tokens"] += self.manifest.initial_prompt_tokens
        if hasattr(self.manifest, 'initial_completion_tokens') and self.manifest.initial_completion_tokens is not None:
            self.token_usage["reid_linking_completion_tokens"] += self.manifest.initial_completion_tokens
            self.token_usage["report_output_text_tokens"] += self.manifest.initial_completion_tokens
        if hasattr(self.manifest, 'initial_image_tokens') and self.manifest.initial_image_tokens is not None:
            self.token_usage["reid_linking_image_tokens"] += self.manifest.initial_image_tokens
            self.token_usage["report_input_image_tokens"] += self.manifest.initial_image_tokens
        
        # Recalculate report_total_tokens based on these re-additions if they occurred
        # Note: _call_llm also adds to report_total_tokens from API responses.
        if (hasattr(self.manifest, 'initial_prompt_tokens') and self.manifest.initial_prompt_tokens) or \
           (hasattr(self.manifest, 'initial_completion_tokens') and self.manifest.initial_completion_tokens) or \
           (hasattr(self.manifest, 'initial_image_tokens') and self.manifest.initial_image_tokens):
            
            self.token_usage["report_total_tokens"] = (
                self.token_usage.get("report_input_text_tokens", 0) +
                self.token_usage.get("report_output_text_tokens", 0) +
                self.token_usage.get("report_input_image_tokens", 0)
            )
            logger.info(f"Applied initial/Re-ID tokens in analyze_video: "
                        f"P={self.manifest.initial_prompt_tokens or 0}, "
                        f"C={self.manifest.initial_completion_tokens or 0}, "
                        f"I={self.manifest.initial_image_tokens or 0}. "
                        f"Updated report_total_tokens: {self.token_usage['report_total_tokens']}")
        # --- END MODIFIED ---

        stopwatch_start_time = time.time()

        logger.info(f'Starting video analysis: "{analysis_config.name}" for {self.manifest.name}')

        # If person_group_id is provided, verify it exists
        if self.person_group_id:
            if ensure_person_group_exists(self.person_group_id, self.env):
                logger.info(f"Using face recognition with person group: {self.person_group_id}")
            else:
                logger.warning(f"Person group {self.person_group_id} not found or not accessible")
                self.person_group_id = None

        # --- Determine if custom aggregation exists ---
        has_custom_aggregation = hasattr(analysis_config, 'process_segment_results')
        # --- EDIT: Rename to reflect potential structure ---
        processed_results = None # Will hold the dict from process_segment_results

        # Analyze videos using the mapreduce sequence
        if analysis_config.analysis_sequence == "mapreduce":
            print(f"Populating prompts for each segment")

            raw_processed_results = None
            processed_results = None

            try:
                if run_async:
                    logger.info(f"Running analysis asynchronously with max {max_concurrent_tasks} tasks...")
                    # Ensure _async_llm_client is initialized.
                    await self._get_async_llm_client() # MODIFIED: Ensure client is ready

                    raw_processed_results = await self._analyze_segment_list_async( # MODIFIED: await instead of asyncio.run
                        analysis_config, max_concurrent_tasks, copyright_json_str
                    )
                else:
                    logger.info("Running analysis sequentially...")
                    raw_processed_results = self._analyze_segment_list(
                        analysis_config, copyright_json_str
                    )

                # --- PATCH START ---
                if has_custom_aggregation:
                    logger.info("Running custom aggregation method…")
                    # if the async branch already returned a dict, just use it
                    if isinstance(raw_processed_results, dict):
                        processed_results = raw_processed_results
                    else:
                        # sync path: raw_processed_results is a list of segment results
                        runtime_so_far = time.time() - stopwatch_start_time # Moved runtime calculation here for sync path
                        processed_results = analysis_config.process_segment_results(
                            enriched_segment_results=raw_processed_results, # Directly pass the list
                            manifest=self.manifest,
                            env=self.env,
                            parsed_copyright_info=(
                                json.loads(copyright_json_str)
                                if copyright_json_str else None
                            ),
                            runtime_seconds=runtime_so_far,
                            tokens=self.token_usage,
                        )
                else:
                    processed_results = raw_processed_results
                # --- PATCH END ---

            finally:
                # --- REMOVED: Call cleanup --- The call after the main analysis loop is sufficient.
                # self._cleanup_temp_blobs()
                pass # Finally block still useful if other cleanup is needed later
                # --- END REMOVED ---

            # --- Analysis complete, now process results ---
            # (Aggregation and summary logic remains here, using the 'processed_results' variable)
            # ... (rest of the logic starting from `final_results = {}`) ...

            # --- EDIT: Standardize final_results structure & Update Manifest ---
            final_results = {} # Initialize the dictionary to be saved as JSON

            # --- PATCH START ---
            if has_custom_aggregation:
                logger.info("Using results from custom aggregation method.")
                # Assign the potentially processed results to final_results
                final_results = processed_results # 'processed_results' IS the action summary dict

                # --- Update manifest.global_tags from the 'globalTags' within final_results ---
                if "globalTags" in final_results and isinstance(final_results["globalTags"], dict):
                    manifest_global_tags_data = final_results["globalTags"]
                    
                    if self.manifest.global_tags is None:
                        self.manifest.global_tags = {"persons": [], "actions": [], "objects": []}

                    # Ensure that the manifest.global_tags are updated with lists
                    self.manifest.global_tags["persons"] = manifest_global_tags_data.get("persons", [])
                    self.manifest.global_tags["actions"] = manifest_global_tags_data.get("actions", [])
                    self.manifest.global_tags["objects"] = manifest_global_tags_data.get("objects", [])
                    logger.debug("Updated self.manifest.global_tags using data from custom aggregation's globalTags.")
                else:
                    # This warning now means the custom aggregation (ActionSummary) itself didn't produce globalTags
                    logger.warning("Custom aggregation result (e.g., ActionSummary output) missing 'globalTags' key or it's not a dict. Manifest global_tags not updated.")
            
            elif isinstance(processed_results, list): # This branch now handles non-custom agg OR refine
                # --- Apply Generic Aggregation to the list of segment results ---
                # This path handles non-custom aggregation and 'refine' sequence outputs.
                logger.info("Running generic aggregation (produces standard structure).")
                try:
                    all_chapters_agg = []
                    global_tags_agg_dict = { "persons": {}, "actions": {}, "objects": {} }

                    for result_container in processed_results:
                         if not isinstance(result_container, dict):
                             logger.warning(f"Skipping non-dict item in results list during generic agg: {type(result_container)}")
                             continue
                         # --- EDIT: Check camelCase 'analysisResult' ---
                         segment_response = result_container.get("analysisResult", {})
                         if not isinstance(segment_response, dict):
                              logger.warning(f"Skipping item with non-dict 'analysisResult' during generic agg: {type(segment_response)}")
                              continue

                         # Add chapters ('chapters' key is fine)
                         chapters_data = segment_response.get("chapters", [])
                         if isinstance(chapters_data, dict): all_chapters_agg.append(chapters_data)
                         elif isinstance(chapters_data, list): all_chapters_agg.extend(chap for chap in chapters_data if isinstance(chap, dict))
                         else: logger.warning(f"Unexpected data type for 'chapters' during generic agg: {type(chapters_data)}")

                         # Merge global tags ('globalTags' key is camelCase)
                         # --- EDIT: Check camelCase 'globalTags' ---
                         tags_data = segment_response.get("globalTags", {})
                         if not isinstance(tags_data, dict):
                             logger.warning(f"Unexpected data type for 'globalTags' during generic agg: {type(tags_data)}")
                             continue

                         for category in ["persons", "actions", "objects"]: # Internal keys 'persons', etc. are fine
                             if category not in tags_data or not isinstance(tags_data.get(category), list): continue

                             for tag_obj in tags_data[category]:
                                 if not isinstance(tag_obj, dict):
                                     logger.warning(f"Skipping non-dictionary tag in '{category}' during generic agg: {type(tag_obj)}")
                                     continue
                                 name = tag_obj.get("name") # 'name' fine
                                 if not name or not isinstance(name, str) or not name.strip():
                                     logger.warning(f"Skipping tag in '{category}' with missing/invalid name during generic agg: {tag_obj}")
                                     continue

                                 cleaned_name = name.strip()
                                 if cleaned_name not in global_tags_agg_dict[category]:
                                     global_tags_agg_dict[category][cleaned_name] = {"name": cleaned_name, "timecodes": []} # 'name', 'timecodes' fine

                                 timecodes = tag_obj.get("timecodes", []) # 'timecodes' fine
                                 if isinstance(timecodes, list):
                                     valid_timecodes = [tc for tc in timecodes if isinstance(tc, dict) and "start" in tc and "end" in tc] # 'start', 'end' fine
                                     global_tags_agg_dict[category][cleaned_name]["timecodes"].extend(valid_timecodes)
                                 else: logger.warning(f"Unexpected timecode format for tag '{cleaned_name}' during generic agg: {timecodes}")

                    # Convert aggregated dict back to list structure and clean up timecodes
                    final_global_tags_agg_list = {} # Use plural keys for final structure
                    for category, tags_dict in global_tags_agg_dict.items():
                         tag_list_agg = []
                         for tag_name, tag_object in tags_dict.items():
                             unique_timecodes_set = set(tuple(sorted(d.items())) for d in tag_object["timecodes"])
                             unique_timecodes_list = sorted(
                                 [dict(t) for t in unique_timecodes_set],
                                 key=lambda x: float(str(x.get("start", "inf")).rstrip("s")) if str(x.get("start", "inf")).rstrip("s").replace('.', '', 1).isdigit() else float('inf')
                             )
                             tag_object["timecodes"] = unique_timecodes_list
                             tag_list_agg.append(tag_object)
                         final_global_tags_agg_list[category] = tag_list_agg # Use plural key

                    # --- Assemble the standard structure for generic aggregation ---
                    # This structure will be used for summary generation if needed,
                    # but the final saved file will likely be the ActionSummary structure
                    # if the ActionSummary config was used.
                    final_results_standard = {
                        "chapters": all_chapters_agg, # 'chapters' fine
                        "globalTags": final_global_tags_agg_list # Use camelCase 'globalTags' containing plural keys internally
                    }
                    # --- Store this standard structure for potential summary use ---
                    # The final file saved will depend on whether ActionSummary processing ran.
                    # For now, let's assume ActionSummary always runs for this config.
                    # We'll set final_results later. If ActionSummary fails, we might need this.
                    _intermediate_generic_results = final_results_standard

                    # Update manifest global tags from generic aggregation
                    self.manifest.global_tags = final_global_tags_agg_list
                    logger.debug("Updated self.manifest.global_tags from generic aggregation.")

                except Exception as e:
                    logger.error(f"Error during generic aggregation: {e}")
                    # ... (existing error handling for generic aggregation failure) ...
                    raise ValueError(f"Error during generic aggregation. Error: {e}")

                # --- EDIT: If generic aggregation ran BUT ActionSummary is the config,
                # we still expect the final result to be from ActionSummary's processing.
                # The generic aggregation path primarily serves non-ActionSummary mapreduce
                # or refine sequences. Let's check if ActionSummary processing needs to run
                # on the `processed_results` list. This happens automatically if `has_custom_aggregation` is true.
                # If `has_custom_aggregation` was false (e.g., refine), we need to handle it.
                # Let's assume `ActionSummary` processing IS the custom aggregation and already ran if `has_custom_aggregation` is true.
                # If it was false (refine), `final_results` is still empty.
                # We need a way to potentially save the generic results if no ActionSummary exists.
                # Given the current structure, let's prioritize the `ActionSummary` output.
                # If `has_custom_aggregation` was true, `final_results` already holds the ActionSummary dict.
                # If it was false (refine), we need to decide what to save. The prompt asks for ActionSummary format.
                # Let's assume for 'refine' we might save the `_intermediate_generic_results` if needed.
                # But for the user's request (ActionSummary config), `has_custom_aggregation` should be true.
                if not final_results: # If custom aggregation didn't run (e.g., refine)
                    logger.warning("No custom aggregation ran. Final results might not match ActionSummary format.")
                    final_results = _intermediate_generic_results # Use generic result as fallback


            # -- Summary Generation & Saving --
            try:
                if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                    # --- EDIT: Target the correct dictionary based on structure ---
                    content_to_summarize = final_results # Default: summarize the whole thing
                    summary_target_dict = final_results # Default target
                    summary_text = "Analysis resulted in empty content."
                    description_text = "No description generated."
                    should_generate_summary = False
                    video_duration = self.manifest.source_video.duration if self.manifest.source_video.duration else 0

                    if "actionSummary" in final_results:
                         action_summary_content = final_results["actionSummary"]
                         content_to_summarize = action_summary_content # Summarize content inside actionSummary
                         summary_target_dict = action_summary_content # Put summary inside actionSummary
                         # Check if there's content inside actionSummary to summarize
                         if action_summary_content and \
                            (action_summary_content.get("chapter") or \
                             action_summary_content.get("person") or \
                             action_summary_content.get("action") or \
                             action_summary_content.get("object")):
                              should_generate_summary = True
                         else:
                              logger.warning("Skipping final summary generation due to empty actionSummary content.")
                              # Add default summary/desc to actionSummary
                              summary_target_dict["description"] = description_text
                              summary_target_dict["summary"] = summary_text

                    # Check standard structure (if not actionSummary - e.g., generic fallback)
                    elif final_results and (final_results.get("chapters") or final_results.get("globalTags")):
                        should_generate_summary = True
                    else: # Standard structure but empty
                        logger.warning("Skipping final summary generation due to empty analysis results.")
                        # Add default summary/desc to top level
                        final_results["description"] = description_text
                        final_results["summary"] = summary_text


                    if should_generate_summary:
                        logger.info(f"Generating summary and description for {self.manifest.name}")

                        # --- Determine summary length instruction ---
                        if video_duration < 30:
                            summary_length_instruction = "1-3 sentences long."
                        elif 30 <= video_duration <= 300: # 30 seconds to 5 minutes
                            summary_length_instruction = "1-2 paragraphs long."
                        else: # Over 5 minutes
                            summary_length_instruction = "at least 3 paragraphs long."
                        logger.info(f"Video duration {video_duration:.2f}s. Requesting summary length: {summary_length_instruction}")

                        # --- Format the selected instruction into the prompt ---
                        # --- ADDED: Format asset categories list --- +
                        asset_cats_str = "Unknown"
                        if hasattr(analysis_config, 'ASSET_CATEGORIES') and isinstance(analysis_config.ASSET_CATEGORIES, list):
                             asset_cats_str = ", ".join([f'"{cat}"' for cat in analysis_config.ASSET_CATEGORIES])
                        else:
                             logger.warning("Could not find ASSET_CATEGORIES list in AnalysisConfig. Prompt will be incomplete.")
                        # --- END ADDED ---

                        formatted_summary_prompt = analysis_config.summary_prompt.format(
                            summary_length_instruction=summary_length_instruction,
                            asset_categories_list=asset_cats_str # Inject formatted categories
                        )

                        summary_prompt_messages = [
                            {"role": "system", "content": formatted_summary_prompt},
                            {"role": "user", "content": json.dumps(content_to_summarize)},
                        ]
                        # --- End prompt formatting ---

                        # Defaults reset here
                        description_text = "Failed to generate description."
                        summary_text = "Failed to generate summary."
                        try:
                            # --- Use the formatted prompt messages ---
                            summary_results_llm = self._call_llm(summary_prompt_messages, log_token_category="summary") # Renamed variable
                            raw_response_content = summary_results_llm.choices[0].message.content.strip()

                            try:
                                # ... (rest of the JSON parsing for description/summary) ...
                                if raw_response_content.startswith("```json"):
                                    raw_response_content = raw_response_content.split("```json", 1)[1]
                                if raw_response_content.endswith("```"):
                                    raw_response_content = raw_response_content.rsplit("```", 1)[0]

                                parsed_summaries = json.loads(raw_response_content.strip())
                                if isinstance(parsed_summaries, dict):
                                    # --- EDIT: Use camelCase keys 'description', 'summary' ---
                                    description_text = parsed_summaries.get("description", description_text).strip()
                                    summary_text = parsed_summaries.get("summary", summary_text).strip()
                                    # --- ADDED: Parse category and subCategory --- +
                                    category_text = parsed_summaries.get("category") # Allow None
                                    sub_category_text = parsed_summaries.get("subCategory") # Allow None
                                    # Basic validation/cleaning (ensure category is string, subCategory is string or None)
                                    if category_text and isinstance(category_text, str):
                                         summary_target_dict["category"] = category_text.strip()
                                    else:
                                         summary_target_dict["category"] = "Other" # Default if missing/invalid
                                    if sub_category_text and isinstance(sub_category_text, str):
                                         summary_target_dict["subCategory"] = sub_category_text.strip()
                                    else:
                                         summary_target_dict["subCategory"] = None # Ensure null if missing/invalid
                                    logger.info(f"Successfully parsed category ('{summary_target_dict['category']}') and subCategory ('{summary_target_dict['subCategory']}') from LLM response.")
                                    logger.info("Successfully parsed description and summary from LLM response.")
                                else:
                                    logger.warning("LLM response for summaries was not a JSON dictionary.")
                                    summary_text = raw_response_content

                            except json.JSONDecodeError as json_e:
                                logger.warning(f"Failed to parse JSON summary/description response: {json_e}")
                                logger.warning(f"Raw response: {raw_response_content}")
                                summary_text = raw_response_content

                        except Exception as llm_e:
                            logger.warning(f"LLM call for summary/description failed: {llm_e}")

                        # --- ADD keys to the target dictionary ---
                        # --- EDIT: Use camelCase keys 'description', 'summary' ---
                        summary_target_dict["description"] = description_text
                        summary_target_dict["summary"] = summary_text
                        logger.debug(f"Added description and summary (camelCase) to {'actionSummary object' if 'actionSummary' in final_results else 'top level'}.")

                    # Update the manifest's final summary field (use the main summary text)
                    # The manifest field itself is final_summary (snake_case)
                    self.manifest.final_summary = summary_text


                # Save the results
                final_results_output_path = os.path.join(
                    self.manifest.processing_params.output_directory,
                    f"_{analysis_config.name}.json",
                )
                logger.info(f"Writing final results structure to {final_results_output_path}")
                os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)
                with open(final_results_output_path, "w", encoding="utf-8") as f:
                     json_obj = json.loads(json.dumps(final_results))
                     f.write(json.dumps(json_obj, indent=4, ensure_ascii=False))

            except Exception as e:
                logger.error(f"Error during final summary generation or saving results: {e}")
                # ... (rest of the error handling for final step) ...
                raise ValueError(f"Failed during summary/saving. Original error: {e}")

            stopwatch_end_time = time.time()
            elapsed_time = stopwatch_end_time - stopwatch_start_time

            logger.info(f'Video analysis completed in {round(elapsed_time, 3)}: "{analysis_config.name}" for {self.manifest.name}')

            # --- Cleanup temp blobs ---
            self._cleanup_temp_blobs()

        # Save final manifest (contains URLs now)
        write_video_manifest(self.manifest)

        # --- Inject runtime into action summary if ActionSummary config is used ---
        # --- FIX: Don't re-call process_segment_results. Update existing dict. ---
        # If we already ran custom aggregation (which populates final_results as a dict)
        # and it's the ActionSummary config, just update its runtime fields.
        if (hasattr(analysis_config, "process_segment_results") and
            analysis_config.name == "ActionSummary" and
            isinstance(final_results, dict)): # Check if final_results is a dict (it should be if ActionSummary ran)

            logger.info("Injecting final runtime into existing actionSummary output.")
            
            # The final_results *is* the action summary dictionary here.
            # No need for .get("actionSummary") if final_results is already it.
            target_dict_for_runtime = final_results 

            if isinstance(target_dict_for_runtime, dict):
                 if "processing_info" not in target_dict_for_runtime:
                      target_dict_for_runtime["processing_info"] = {}
                 
                 # Ensure elapsed_time exists, calculate if not (e.g. if error before its calculation)
                 if 'elapsed_time' not in locals():
                     elapsed_time = time.time() - stopwatch_start_time

                 try:
                     from .cobra_utils import seconds_to_iso8601_duration
                     target_dict_for_runtime["processing_info"]["runtime_iso8601"] = seconds_to_iso8601_duration(elapsed_time)
                 except ImportError:
                     logger.warning("Warning: cobra_utils not found for runtime formatting. Using basic format for ISO.")
                     target_dict_for_runtime["processing_info"]["runtime_iso8601"] = f"PT{elapsed_time:.3f}S"

                 target_dict_for_runtime["processing_info"]["runtime_seconds"] = round(elapsed_time, 3)
                 # Also update the tokens in processing_info if ActionSummary is the direct output
                 target_dict_for_runtime["processing_info"]["tokens_used"] = self.token_usage

            else:
                 logger.warning(f"Warning: Expected 'final_results' to be a dict for ActionSummary, but got {type(target_dict_for_runtime)}. Cannot inject runtime.")


        # --- END FIX ---

        # --- ADDED: Finalize top-level token counts before returning ---
        self._finalize_token_usage()
        # --- END ADDED ---


        # --- ADDED: Generate Summary Report ---
        try:
            logger.info("Generating Analysis Summary Report...")
            # Ensure final_results is defined; it might be set in different branches
            final_results_for_report = final_results if 'final_results' in locals() else {}
            elapsed_time_for_report = elapsed_time if 'elapsed_time' in locals() else (time.time() - stopwatch_start_time)
            output_path_for_report = final_results_output_path if 'final_results_output_path' in locals() else "N/A"
            
            report_data = self._gather_report_data(
                final_results_for_report, 
                analysis_config.name, 
                elapsed_time_for_report, 
                output_path_for_report
            )
            self._write_summary_report(report_data)
            logger.info("Analysis Summary Report generated.")
        except Exception as report_e:
            logger.warning(f"Failed to generate analysis summary report: {report_e}")
        # --- END ADDED ---

        return final_results

    def generate_segment_prompts(self, analysis_config: Type[AnalysisConfig]):
        for segment in self.manifest.segments:
            self._generate_segment_prompt(segment, analysis_config)

    def generate_summary_prompt(
        self, analysis_config: Type[AnalysisConfig], final_results
    ):
        messages = [
            {"role": "system", "content": analysis_config.summary_prompt},
            {"role": "user", "content": json.dumps(final_results)},
        ]
        return messages

    def _analyze_segment_list(
        self,
        analysis_config: Type[AnalysisConfig],
        copyright_json_str: Optional[str] = None # Added copyright string param
    ):
        results_list = []
        for segment in self.manifest.segments:
            parsed_response = self._analyze_segment(
                segment=segment, analysis_config=analysis_config
            )
            results_list.append({
                "analysisResult": parsed_response,
                "segmentName": segment.segment_name,
                "startTime": segment.start_time,
                "endTime": segment.end_time,
                "framePaths": segment.segment_frames_file_path,
                "fullTranscriptionObject": self.manifest.audio_transcription if not results_list else None
             })
        if hasattr(analysis_config, 'process_segment_results'):
            parsed_copyright = None
            if copyright_json_str:
                try:
                    parsed_copyright = json.loads(copyright_json_str)
                    logger.debug("Successfully parsed copyright JSON string.")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse copyright JSON string: {e}. Content: {copyright_json_str[:100]}...")
            final_results = analysis_config.process_segment_results(
                results_list,
                self.manifest,
                self.env,
                parsed_copyright_info=parsed_copyright
            )
            logger.debug("Returning results from custom process_segment_results.")
            return final_results
        else:
             return results_list

    def _analyze_segment_list_sequentially(
        self, analysis_config: Type[SequentialAnalysisConfig]
    ):
        logger.info(f"Analyzing {len(self.manifest.segments)} segments sequentially (stateful)...")
        previous_segment_summary = None
        for i, segment in enumerate(self.manifest.segments):
            logger.info(f"Analyzing segment {i + 1}/{len(self.manifest.segments)} ({segment.id}) sequentially.")
            # Sequential analysis might need to pass the async_llm_client too if it makes direct calls.
            # For now, assuming _analyze_segment is synchronous and uses self._get_llm_client()
            segment_result = self._analyze_segment(
                segment, 
                analysis_config,
                copyright_json_str=None, # Assuming copyright is handled at a higher level or not per-seq-segment
                previous_segment_summary=previous_segment_summary # Pass previous summary
            )
            if segment_result:
                if segment.analyzed_result is None: segment.analyzed_result = {}
                segment.analyzed_result[analysis_config.name] = segment_result
                previous_segment_summary = segment_result.get("chapters", [{}])[0].get("summary") # Update for next iteration

    async def _analyze_segment_list_async(
        self,
        analysis_config: Type[AnalysisConfig],
        max_concurrent_tasks=None,
        copyright_json_str: Optional[str] = None
    ):
        if not self.manifest.segments:
            logger.warning("No segments to analyze.")
            return []

        # Ensure the internal async LLM client is initialized
        if not self._internal_async_client: # MODIFIED
            self._internal_async_client = await self._get_async_llm_client() # MODIFIED
            if not self._internal_async_client: # MODIFIED
                logger.error("VideoAnalyzer: Failed to initialize async LLM client for async segment processing. Aborting.")
                return None

        # Max concurrent tasks for segment processing
        # Ensure max_concurrent_tasks is positive, default if not specified or invalid
        if max_concurrent_tasks is None:
            max_concurrent_tasks = os.cpu_count() or 1 # Default to CPU count
        
        logger.info(f"Analyzing {len(self.manifest.segments)} segments asynchronously with {max_concurrent_tasks} workers.")
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks: List[Coroutine[Any, Any, Optional[Dict[str, Any]]]] = []

        async def sem_task(segment_to_analyze: Segment) -> Optional[Dict[str, Any]]:
            async with semaphore:
                logger.info(f"Starting analysis for segment {segment_to_analyze.segment_name}") # MODIFIED: Use .segment_name
                result = await self._analyze_segment_async(segment_to_analyze, analysis_config, copyright_json_str)
                logger.info(f"Completed analysis for segment {segment_to_analyze.segment_name}") # MODIFIED: Use .segment_name
                return result

        for i, segment_data in enumerate(self.manifest.segments):
            if self.reprocess_segments or segment_data.analyzed_result is None or not segment_data.analyzed_result.get(analysis_config.name):
                tasks.append(sem_task(segment_data))
            else:
                logger.info(f"Skipping segment {i + 1} ({segment_data.segment_name}) - already analyzed with {analysis_config.name}.") # MODIFIED: Use .segment_name for consistency

        if not tasks:
            logger.info("No segments to analyze asynchronously.")
            return

        results = await asyncio.gather(*tasks)

        # After gathering results, we need to map them back correctly and structure them
        # similar to what _analyze_segment_list (synchronous) would produce for process_segment_results.

        # Check if results is None or if any task failed critically and returned None in its place in the list.
        # It's assumed asyncio.gather will return a list of results, where individual results might be None
        # if a specific task failed but was caught and returned None.
        # If 'results' itself is None, it implies a more fundamental issue with asyncio.gather or client init.
        if results is None:
            logger.error("_analyze_segment_list_async: asyncio.gather returned None. This is unexpected. Returning empty list.")
            return []

        # Repackage results to match the expected structure for process_segment_results
        # The 'results' from asyncio.gather should be in the same order as 'tasks' were created,
        # and tasks were created based on self.manifest.segments.
        # Each item in 'results' is the output of _analyze_segment_async, which should be
        # a dictionary (final_segment_result) or None if that specific segment analysis failed.

        packaged_results = []
        for i, segment_obj in enumerate(self.manifest.segments):
            # The result for the i-th segment should be at results[i]
            segment_analysis_output = results[i] if i < len(results) else None

            # MODIFIED: Check for the "error" key in the result
            if segment_analysis_output and isinstance(segment_analysis_output, dict) and not segment_analysis_output.get("error"):
                # Successfully processed segment
                # Ensure segment.analyzed_result is updated if _analyze_segment_async updated it.
                # _analyze_segment_async is designed to return the analysis result,
                # but the manifest's segment.analyzed_result might not be directly updated by it if it's a copy.
                # For safety, let's use the returned 'segment_analysis_output' directly.
                
                # Determine frame paths: prefer URLs if available
                frame_paths_for_container: List[str] = []
                if segment_obj.frame_urls:
                    frame_paths_for_container = segment_obj.frame_urls
                elif segment_obj.segment_frames_file_path:
                    frame_paths_for_container = segment_obj.segment_frames_file_path

                packaged_results.append({
                    "analysisResult": segment_analysis_output, # This is the dict returned by _analyze_segment_async
                    "segment_object": segment_obj,  # <<< ADDED THIS LINE
                    "segmentName": segment_obj.segment_name,
                    "startTime": segment_obj.start_time,
                    "endTime": segment_obj.end_time,
                    "framePaths": frame_paths_for_container,
                    "fullTranscriptionObject": self.manifest.audio_transcription,
                    # Add any other fields expected by process_segment_results if necessary
                })
            elif segment_analysis_output and isinstance(segment_analysis_output, dict) and segment_analysis_output.get("error"): # Check for error key
                # MODIFIED: Log the specific error message
                error_message = segment_analysis_output.get('error', 'Unknown error')
                raw_response_details = segment_analysis_output.get('raw_response', 'N/A') # Get raw response if available
                logger.error(f"Segment {segment_obj.segment_name} (async) failed with error: {error_message}. Raw response hint: {str(raw_response_details)[:200]}...")
            else:
                logger.warning(f"No valid analysis result (or unexpected format) for segment {segment_obj.segment_name} (async) at index {i}. Result: {segment_analysis_output}. It will be excluded.")
        
        logger.info(f"Asynchronous segment analysis processing finished. Packaged {len(packaged_results)} successful segment results for aggregation.")
        return packaged_results

    def _analyze_segment(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig],
        copyright_json_str: Optional[str] = None, 
        previous_segment_summary: Optional[str] = None 
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes a single segment of the video.
        This is the synchronous version.
        """
        logger.info(f"Starting sync analysis for segment ID: {segment.id}, Time: {segment.start_time}-{segment.end_time}")
        llm_client = self._get_llm_client()

        if not segment.frames:
            logger.warning(f"Segment {segment.id} has no frames. Skipping analysis.")
            return None

        segment_frame_paths = [frame.frame_path for frame in segment.frames if frame.frame_path]
        if not segment_frame_paths:
             logger.warning(f"Segment {segment.id} has no valid frame paths. Skipping.")
             return None

        # --- Filter YOLO tags from self.active_yolo_tags_source for the current segment --- 
        yolo_tags_for_segment: List[Dict[str, Any]] = []
        if self.active_yolo_tags_source:
            for tag in self.active_yolo_tags_source:
                tag_overlaps = False
                # Each tag in active_yolo_tags_source should have a "timecodes" list
                # where each item is a dict {"start": float, "end": float}
                for tc in tag.get("timecodes", []):
                    if max(segment.start_time, tc["start"]) < min(segment.end_time, tc["end"]):
                        tag_overlaps = True
                        break
                if tag_overlaps:
                    yolo_tags_for_segment.append(tag)
        
        logger.debug(f"Segment {segment.id}: Found {len(yolo_tags_for_segment)} active YOLO tags for this segment (using_refined: {self.using_refined_tags}).")
        # --- End YOLO Tag Filtering ---

        prompt_messages = self._generate_segment_prompt(
            segment=segment,
            analysis_config=analysis_config,
            frames_subset=segment_frame_paths, 
            times_subset=[f.timestamp for f in segment.frames if f.timestamp is not None],
            copyright_json_str=copyright_json_str,
            yolo_tags_for_segment=yolo_tags_for_segment, # Pass the filtered tags
            previous_segment_summary=previous_segment_summary
        )

        if not prompt_messages:
            logger.error(f"Failed to generate prompt messages for segment {segment.id}.")
            return None

        logger.info(f"Calling LLM for segment {segment.id} ({analysis_config.name}). Image detail: {self.image_detail_level}")
        
        try:
            llm_response = self._call_llm(
                messages=prompt_messages,
                model=self.env.vision.deployment or getattr(analysis_config, 'DEFAULT_MODEL', 'gpt-4o'), # Pass model from env/config
                log_token_category="segment_generic_analysis" # Use the new category
            )
            # Ensure llm_response is not None and has the expected structure
            if not llm_response or not hasattr(llm_response, 'choices') or not llm_response.choices or not hasattr(llm_response.choices[0], 'message') or not hasattr(llm_response.choices[0].message, 'content'):
                logger.error(f"LLM call for segment {segment.id} returned an unexpected response structure or was None.")
                return {"error": "LLM call returned unexpected response structure", "raw_response": str(llm_response)}
            
            response_content = llm_response.choices[0].message.content
            
            # Token accumulation is now handled by _call_llm, so manual accumulation below is removed.
            # prompt_tokens = llm_response.usage.prompt_tokens
            # completion_tokens = llm_response.usage.completion_tokens
            # image_tokens = estimate_image_tokens(prompt_messages) 
            # self.token_usage["prompt_tokens"] += prompt_tokens
            # self.token_usage["completion_tokens"] += completion_tokens
            # self.token_usage["image_tokens"] += image_tokens
            # self.token_usage["total_tokens"] += prompt_tokens + completion_tokens 

            # logger.debug(f"Segment {segment.id} LLM response received. Tokens: P={prompt_tokens}, C={completion_tokens}, I(est)={image_tokens}")
            # The _call_llm wrapper now handles detailed logging, including the category.

        except Exception as e:
            logger.error(f"LLM call failed for segment {segment.id}: {e}", exc_info=True)
            return {"error": f"LLM call failed: {e}", "raw_response": None}

        parsed_llm_output = self._parse_llm_json_response(
            response_content,
            expecting_chapters="chapters" in getattr(analysis_config, 'results_template', {}),
            expecting_tags="globalTags" in getattr(analysis_config, 'results_template', {})
        )

        if not parsed_llm_output:
            logger.error(f"Failed to parse LLM JSON response for segment {segment.id}. Response: {response_content[:500]}")
            return {"error": "Failed to parse LLM response", "raw_response": response_content}
        
        final_segment_result = parsed_llm_output

        # --- Name-to-TrackID Linking (SYNC Version - Simplified) ---
        if "globalTags" in final_segment_result: # Ensure globalTags key exists
            llm_persons = final_segment_result.get("globalTags", {}).get("persons", [])
            if llm_persons:
                logger.info(f"Segment {segment.id} (sync): Processing {len(llm_persons)} persons from LLM output to ensure ID and yoloClass preservation.")
            
            updated_persons_sync = []
            for p_entry in llm_persons: 
                # p_entry is a dict that should already have name, id, yoloClass, timecodes from _describe_yolo_tags_with_gpt or main LLM
                new_person_entry = {
                    "classDescription": p_entry.get("classDescription"), # MODIFIED: Use classDescription
                    "id": p_entry.get("id"), # Directly use id from p_entry
                    "yoloClass": p_entry.get("yoloClass", "person"), # Default to person if not present
                    "timecodes": p_entry.get("timecodes", [{"start": segment.start_time, "end": segment.end_time}]),
                }

                # Carry over other relevant fields if they exist in p_entry
                for key in ["thumb", "refined_track_id_str", "original_yolo_ids_ref"]:
                    if key in p_entry and p_entry[key] is not None:
                        new_person_entry[key] = p_entry[key]
                
                if new_person_entry.get("id") is None:
                     logger.warning(f"Person '{new_person_entry.get('classDescription')}' in segment {segment.id} (sync) still has None ID after processing p_entry: {p_entry}. This should have been resolved earlier.") # MODIFIED: Use classDescription

                updated_persons_sync.append(new_person_entry)
            
            # Only update if the persons key actually existed in globalTags
            if "persons" in final_segment_result.get("globalTags", {}):
                final_segment_result["globalTags"]["persons"] = updated_persons_sync
            elif updated_persons_sync: # If persons key didn't exist but we created entries
                if "globalTags" not in final_segment_result or not isinstance(final_segment_result.get("globalTags"), dict):
                    final_segment_result["globalTags"] = {} # Ensure globalTags dict exists
                final_segment_result["globalTags"]["persons"] = updated_persons_sync
        # --- End Name-to-TrackID Linking (SYNC Version) ---

        # Integrate non-person YOLO tags if not already covered by LLM
        if "globalTags" not in final_segment_result: final_segment_result["globalTags"] = {}
        if "objects" not in final_segment_result["globalTags"]: final_segment_result["globalTags"]["objects"] = []
        
        yolo_object_names_from_llm = {obj.get("classDescription") for obj in final_segment_result["globalTags"].get("objects", [])} # MODIFIED: Use classDescription

        for y_tag in yolo_tags_for_segment:
            if y_tag["class"] != "person": # Persons are handled by the Name-to-TrackID linking or described path
                # Determine the numeric ID to use for this y_tag
                tag_numeric_id = None
                if self.using_refined_tags: # Check if the source was refined tags
                    original_ids = y_tag.get("original_yolo_ids")
                    if original_ids and isinstance(original_ids, list) and original_ids:
                        tag_numeric_id = original_ids[0]
                    if tag_numeric_id is None:
                        refined_id_str = y_tag.get("refined_track_id")
                        if refined_id_str and isinstance(refined_id_str, str):
                            try:
                                tag_numeric_id = int(refined_id_str.split('_')[-1])
                            except (ValueError, IndexError):
                                pass # Keep tag_numeric_id as None
                else: # Using raw tags
                    tag_numeric_id = y_tag.get("id")

                # Use refined_track_id or original yolo id as name if not described by LLM, else class
                yolo_tag_description = y_tag.get("gpt_description") # MODIFIED: Check gpt_description for classDescription
                if not yolo_tag_description:
                    if self.using_refined_tags and y_tag.get("refined_track_id"):
                         yolo_tag_description = y_tag.get("refined_track_id") # Fallback to refined_track_id string if not described
                    elif y_tag.get("id") is not None:
                         yolo_tag_description = f'{y_tag["class"]}_{y_tag.get("id")}' # Fallback to class_id for raw tags
                    else:
                         yolo_tag_description = y_tag["class"] # Ultimate fallback to class

                if yolo_tag_description not in yolo_object_names_from_llm: 
                    object_entry = {
                        "classDescription": yolo_tag_description,  # MODIFIED: Use classDescription
                        "yoloClass": y_tag["class"],  # Corrected to yoloClass
                        "id": tag_numeric_id, # Use the determined numeric ID
                        "timecodes": y_tag["timecodes"],
                        "thumb": y_tag.get("thumb") or y_tag.get("representative_thumb_path"),
                    }
                    # Add refined_track_id_str if it exists and we used refined tags, for traceability
                    if self.using_refined_tags and y_tag.get("refined_track_id"):
                        object_entry["refined_track_id_str"] = y_tag.get("refined_track_id")
                    # Add original_yolo_ids list if present, or the single original ID
                    if y_tag.get("original_yolo_ids"):
                        object_entry["original_yolo_ids_ref"] = y_tag.get("original_yolo_ids")
                    elif y_tag.get("id") is not None and not self.using_refined_tags: # for raw tags store the id under original_yolo_id_ref
                        object_entry["original_yolo_id_ref"] = y_tag.get("id")

                    final_segment_result["globalTags"]["objects"].append(object_entry)

        logger.info(f"Sync analysis for segment {segment.segment_name} completed.")
        return final_segment_result

    async def _analyze_segment_async(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig],
        copyright_json_str: Optional[str] = None,
        previous_segment_summary: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyzes a single segment asynchronously using the specified analysis_config."""
        logger.info(f"Starting ASYNC analysis for segment: {segment.segment_name}")
        if not segment.segment_frames_file_path and not segment.frame_urls:
            logger.warning(f"Segment {segment.segment_name} has no frames or frame URLs. Skipping analysis.")
            return None

        # --- BEGIN: Granular Action Extraction ---
        # Ensure this runs before the main segment analysis if its results might be used in prompts
        # or if it's a standalone tagging effort for actions.
        try:
            extracted_actions = await self._extract_actions_per_segment_async(segment)
            segment.extracted_actions = extracted_actions # Store raw extracted actions
            logger.info(f"Stored {len(extracted_actions)} raw actions for segment {segment.segment_name}")
        except Exception as e:
            logger.error(f"Error during dedicated action extraction for segment {segment.segment_name}: {e}")
            segment.extracted_actions = [] # Ensure it's an empty list on error
        # --- END: Granular Action Extraction ---

        # Determine active YOLO tags for this segment
        active_yolo_tags_for_segment = self._get_active_yolo_tags_for_segment(segment)

        start_time_seg = time.time()

        if not self._internal_async_client:
            self._internal_async_client = await self._get_async_llm_client()
            if not self._internal_async_client:
                logger.error(f"Failed to initialize async LLM client for segment {segment.segment_name}. Aborting analysis for this segment.")
                return {"error": f"Async LLM client initialization failed for segment {segment.segment_name}", "raw_response": None}

        logger.info(f"Starting async analysis for segment ID: {segment.segment_name}, Time: {segment.start_time}-{segment.end_time}")

        yolo_tags_for_segment = []
        if self.active_yolo_tags_source:
            for tag in self.active_yolo_tags_source:
                tag_overlaps = False
                for tc in tag.get("timecodes", []):
                    if max(segment.start_time, tc["start"]) < min(segment.end_time, tc["end"]):
                        tag_overlaps = True
                        break
                if tag_overlaps:
                    yolo_tags_for_segment.append(tag)
        logger.debug(f"Segment {segment.segment_name} (async): Found {len(yolo_tags_for_segment)} active YOLO tags (using_refined: {self.using_refined_tags}).")

        frames_for_prompt: List[str] = []
        if segment.frame_urls:
            frames_for_prompt = segment.frame_urls
            logger.debug(f"Segment {segment.segment_name} (async): Using frame_urls for prompt ({len(frames_for_prompt)} URLs).")
        elif segment.segment_frames_file_path:
            frames_for_prompt = segment.segment_frames_file_path
            logger.debug(f"Segment {segment.segment_name} (async): Using segment_frames_file_path for prompt ({len(frames_for_prompt)} paths).")
        
        times_for_prompt: List[float] = segment.segment_frame_time_intervals if segment.segment_frame_time_intervals is not None else []

        prompt_messages = self._generate_segment_prompt(
            segment=segment,
            analysis_config=analysis_config,
            frames_subset=frames_for_prompt,
            times_subset=times_for_prompt,
            copyright_json_str=copyright_json_str,
            yolo_tags_for_segment=yolo_tags_for_segment,
            previous_segment_summary=previous_segment_summary
        )

        if not prompt_messages:
            logger.error(f"Failed to generate prompt messages for segment {segment.segment_name} (async).")
            return {"error": f"Prompt generation failed for segment {segment.segment_name}", "raw_response": None}

        logger.info(f"Calling LLM async for segment {segment.segment_name} ({analysis_config.name}). Image detail: {self.image_detail_level}")
        
        response_content = None
        llm_response_obj = None

        try:
            llm_response_obj = await self._call_llm_async(
                messages=prompt_messages,
                model=self.env.vision.deployment or getattr(analysis_config, 'DEFAULT_MODEL', 'gpt-4o'),
                log_token_category="segment_generic_analysis" # Use the new category
            )

            if not llm_response_obj or not hasattr(llm_response_obj, 'choices') or not llm_response_obj.choices or not hasattr(llm_response_obj.choices[0], 'message') or not hasattr(llm_response_obj.choices[0].message, 'content'):
                logger.error(f"LLM async call for segment {segment.segment_name} returned an invalid/empty response object or no choices. Response object: {llm_response_obj}")
                raw_error_details = str(llm_response_obj) if llm_response_obj else "No response object"
                return {"error": f"LLM returned invalid/empty response object or no choices. Details: {raw_error_details}", "raw_response": None}

            response_content = llm_response_obj.choices[0].message.content

            # Token accumulation is now handled by _call_llm_async, so manual accumulation below is removed.
            # prompt_tokens = getattr(getattr(llm_response_obj, 'usage', None), 'prompt_tokens', 0)
            # completion_tokens = getattr(getattr(llm_response_obj, 'usage', None), 'completion_tokens', 0)
            # api_total_tokens = getattr(getattr(llm_response_obj, 'usage', None), 'total_tokens', 0)
            # image_tokens = estimate_image_tokens(prompt_messages)
            # self.token_usage["prompt_tokens"] += prompt_tokens
            # self.token_usage["completion_tokens"] += completion_tokens
            # self.token_usage["image_tokens"] += image_tokens
            # self.token_usage["total_tokens"] += api_total_tokens if api_total_tokens > 0 else (prompt_tokens + completion_tokens + image_tokens)
            # logger.debug(f"Segment {segment.segment_name} (async) LLM response. API Tokens: P={prompt_tokens}, C={completion_tokens}, TotalFromAPI={api_total_tokens}. Estimated ImgAdd: {image_tokens}")
            # The _call_llm_async wrapper now handles detailed logging, including the category.

        except openai.APIStatusError as e:
            logger.error(f"LLM async call FAILED for segment {segment.segment_name} with APIStatusError: {e.status_code} - {e.response.text if e.response else e.message}", exc_info=True)
            return {"error": f"LLM APIStatusError {e.status_code}: {e.message}", "raw_response": e.response.text if e.response else None}
        except openai.APITimeoutError as e:
            logger.error(f"LLM async call TIMEOUT for segment {segment.segment_name}: {e}", exc_info=True)
            return {"error": f"LLM APITimeoutError: {e}", "raw_response": None}
        except openai.APIConnectionError as e:
            logger.error(f"LLM async call CONNECTION FAILED for segment {segment.segment_name}: {e}", exc_info=True)
            return {"error": f"LLM APIConnectionError: {e}", "raw_response": None}
        except openai.APIError as e:
            logger.error(f"LLM async call FAILED for segment {segment.segment_name} with APIError: {e}", exc_info=True)
            return {"error": f"LLM APIError: {e}", "raw_response": str(e)}
        except Exception as e:
            logger.error(f"LLM async call FAILED for segment {segment.segment_name} with generic Exception: {e}", exc_info=True)
            raw_details = str(llm_response_obj) if llm_response_obj else "No response object before exception"
            return {"error": f"LLM call failed: {e}. Details: {raw_details}", "raw_response": None}

        parsed_llm_output = self._parse_llm_json_response(
            response_content,
            expecting_chapters="chapters" in getattr(analysis_config, 'results_template', {}),
            expecting_tags="globalTags" in getattr(analysis_config, 'results_template', {})
        )

        # --- NEW: Always populate globalTags with YOLO+GPT subclassing ---
        yolo_gpt_tags = await self._describe_yolo_tags_with_gpt(segment, yolo_tags_for_segment)
        if not parsed_llm_output or "error" in parsed_llm_output:
            logger.error(f"Failed to parse LLM JSON response for segment {segment.segment_name} (async). Response: {response_content[:500] if response_content else 'None'}")
            # Return only YOLO+GPT tags if LLM failed
            return {"globalTags": yolo_gpt_tags, "error": parsed_llm_output.get("error", "Failed to parse LLM response") if parsed_llm_output else "Parsing resulted in None", "raw_response": response_content}

        # Merge LLM tags with YOLO+GPT tags (LLM tags take precedence if present)
        if "globalTags" not in parsed_llm_output or not parsed_llm_output["globalTags"]:
            parsed_llm_output["globalTags"] = yolo_gpt_tags
        else:
            # Merge persons/objects
            for cat in ["persons", "objects"]:
                llm_tags = parsed_llm_output["globalTags"].get(cat, [])
                yolo_tags = yolo_gpt_tags.get(cat, [])
                # Add YOLO tags not already present by name
                llm_descriptions = {t["classDescription"] for t in llm_tags if "classDescription" in t} # MODIFIED: Use classDescription
                for t in yolo_tags:
                    if t["classDescription"] not in llm_descriptions: # MODIFIED: Use classDescription
                        llm_tags.append(t)
                parsed_llm_output["globalTags"][cat] = llm_tags
        final_segment_result = parsed_llm_output
        if "error" in final_segment_result:
            logger.warning(f"Segment {segment.segment_name} (async) completed but result contains an error key: {final_segment_result['error']}")
        else:
            final_segment_result.pop("error", None)
            logger.info(f"Async analysis for segment {segment.segment_name} completed successfully. Time: {time.time() - start_time_seg:.2f}s")

        # After describing YOLO tags with GPT, extract actions per frame
        frames_subset = segment.segment_frames_file_path
        times_subset = segment.segment_frame_time_intervals
        actions = await self._extract_actions_per_frame(segment, frames_subset, times_subset)
        # Merge actions into globalTags
        if "globalTags" not in final_segment_result:
            final_segment_result["globalTags"] = {}
        final_segment_result["globalTags"]["actions"] = actions
        return final_segment_result

    async def _extract_actions_per_frame(self, segment, frames_subset, times_subset):
        """
        For each frame, send to GPT to extract actions, and aggregate actions and their time intervals.
        Returns a list of action dicts: {"name": action_name, "timecodes": [{"start": ..., "end": ...}]}
        """
        actions_by_name = {}
        for i, frame_path in enumerate(frames_subset):
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            import base64
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
            prompt = "Describe any actions occurring in this frame. Return a list of action names only."
            messages = [
                {"role": "system", "content": "You are a vision assistant for video action recognition."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"}}
                ]}
            ]
            try:
                response = await self._call_llm_async(messages, model="gpt-4-vision-preview")
                # Parse the response for a list of actions
                import json
                actions = []
                if isinstance(response, list):
                    actions = response
                elif isinstance(response, str):
                    try:
                        actions = json.loads(response)
                        if not isinstance(actions, list):
                            actions = [actions]
                    except Exception:
                        actions = [response.strip()]
                for action in actions:
                    if not action or not isinstance(action, str):
                        continue
                    if action not in actions_by_name:
                        actions_by_name[action] = []
                    # Add a timecode for this frame
                    actions_by_name[action].append(times_subset[i])
            except Exception as e:
                continue
        # Aggregate consecutive frames for each action into intervals
        action_intervals = []
        for action, times in actions_by_name.items():
            if not times:
                continue
            times = sorted(times)
            intervals = []
            start = times[0]
            prev = times[0]
            for t in times[1:]:
                if t - prev > 1.0: # If gap > 1s, start new interval
                    intervals.append({"start": start, "end": prev})
                    start = t
                prev = t
            intervals.append({"start": start, "end": prev})
            action_intervals.append({"name": action, "timecodes": intervals})
        return action_intervals

    async def _describe_yolo_tags_with_gpt(self, segment, yolo_tags_for_segment):
        """
        For each YOLO tag (object/person), crop a representative frame and send to GPT for a descriptive label.
        Returns a dict: {"persons": [...], "objects": [...]} for globalTags.
        """
        from .cobra_utils import get_frame_crop_base64
        described_tags = {"persons": [], "objects": []}
        if not yolo_tags_for_segment:
            return described_tags
        for tag in yolo_tags_for_segment:
            tag_class = tag.get("class")
            tag_timecodes = tag.get("timecodes", [])
            # Determine the numeric ID to use
            final_numeric_id = None
            if self.using_refined_tags: # Check if the source was refined tags
                original_ids = tag.get("original_yolo_ids")
                if original_ids and isinstance(original_ids, list) and original_ids:
                    # Prefer the first original ID if available and it's an int
                    if isinstance(original_ids[0], int):
                        final_numeric_id = original_ids[0]
                    else: # If not an int, maybe it's a string representation
                        try:
                            final_numeric_id = int(original_ids[0])
                        except (ValueError, TypeError):
                            logger.warning(f"Could not parse original_id '{original_ids[0]}' to int. Will use as string if possible.")
                            final_numeric_id = str(original_ids[0]) # Keep as string if not parsable to int
                            
                    if final_numeric_id is None: # If no original_ids or first one wasn't suitable
                        refined_id_str = tag.get("refined_track_id")
                        if refined_id_str and isinstance(refined_id_str, str):
                            try:
                                final_numeric_id = int(refined_id_str.split('_')[-1])
                            except (ValueError, IndexError):
                                logger.warning(f"Could not parse numeric ID from refined_track_id '{refined_id_str}'. Using string value.")
                                final_numeric_id = refined_id_str # Fallback to string refined_track_id
                else: # Using raw tags
                    raw_id = tag.get("id")
                    if isinstance(raw_id, int):
                        final_numeric_id = raw_id
                    elif raw_id is not None:
                        try:
                            final_numeric_id = int(raw_id) # Attempt direct conversion first (e.g. "29")
                        except (ValueError, TypeError):
                            # If direct conversion fails, try splitting (e.g. if raw_id is unexpectedly "person_refined_29")
                            try:
                                # Ensure raw_id is a string before splitting
                                id_str_to_split = str(raw_id)
                                final_numeric_id = int(id_str_to_split.split('_')[-1])
                                logger.info(f"Parsed numeric ID '{final_numeric_id}' from raw tag ID '{raw_id}' using underscore splitting. This suggests the raw tag ID format might be unexpected.")
                            except (ValueError, TypeError, IndexError):
                                logger.warning(f"Raw tag ID '{raw_id}' is not an int and could not be parsed to a numeric ID directly or by splitting. Using string value.")
                                final_numeric_id = str(raw_id) # Fallback to string raw_id

                # Further fallbacks if final_numeric_id is still None
                if final_numeric_id is None:
                    logger.warning(f"final_numeric_id is None for tag {tag}. Checking further fallbacks.")
                    # This re-checks refined_track_id and tag.get('id') as strings if previous attempts failed
                    if self.using_refined_tags and tag.get("refined_track_id"):
                        final_numeric_id = str(tag.get("refined_track_id"))
                        logger.debug(f"Using string refined_track_id '{final_numeric_id}' as ID.")
                    elif tag.get("id") is not None:
                        final_numeric_id = str(tag.get("id"))
                        logger.debug(f"Using raw tag.id '{final_numeric_id}' as string ID.")
                    else:
                        final_numeric_id = -1 # Ultimate fallback to prevent None -> null
                        logger.warning(f"All ID sources failed for tag {tag}. Using -1 as ID.")
                elif not isinstance(final_numeric_id, int): # If it became a string, log it
                     logger.info(f"Tag ID for class '{tag_class}' is a string: '{final_numeric_id}'. This is a fallback.")


            thumb_path = tag.get("thumb")
            # Use GPT description if available, fallback to YOLO class
            gpt_desc = tag.get("gpt_description")
            current_class_description = gpt_desc if gpt_desc else tag_class # MODIFIED variable name and assignment
            # If no GPT description, generate one using the crop
            if not gpt_desc and thumb_path and os.path.exists(thumb_path):
                # Read and encode the crop
                with open(thumb_path, "rb") as f:
                    crop_bytes = f.read()
                import base64
                crop_base64 = base64.b64encode(crop_bytes).decode("utf-8")
                # Send to GPT for description
                prompt = f"Object class: '{tag_class}'. Provide a concise, specific, descriptive label for this item. Focus on its material, type, color, or key distinguishing features. Do not use introductory phrases or refer to 'the image'. For example, if the class is 'bag', a good label might be 'red leather handbag' or 'canvas backpack'. If class is 'car', 'blue sedan' or 'vintage sports car'. Output only the descriptive label itself, nothing else."
                messages = [
                    {"role": "system", "content": "You are a vision assistant skilled at providing concise object descriptions."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_base64}", "detail": "low"}}
                    ]}
                ]
                try:
                    response_obj = await self._call_llm_async(messages, model=self.env.vision.deployment or "gpt-4-vision-preview", log_token_category="yolo_describe")
                    
                    extracted_description_content = None
                    if response_obj and hasattr(response_obj, 'choices') and response_obj.choices:
                        first_choice = response_obj.choices[0]
                        if hasattr(first_choice, 'message') and first_choice.message:
                            message_content = getattr(first_choice.message, 'content', None)
                            if isinstance(message_content, str):
                                extracted_description_content = message_content.strip().replace('"', '')

                    if extracted_description_content: # Check if we got a valid, non-empty string
                        current_class_description = extracted_description_content # MODIFIED: Update current_class_description
                    else:
                        logger.warning(f"LLM did not return a valid string description for YOLO tag class '{tag_class}' (ID info: {final_numeric_id}). Using class name as fallback.")
                        # current_class_description remains tag_class (already set as fallback)
                        pass

                except Exception as e:
                    logger.error(f"Error calling LLM or parsing content for YOLO tag description for '{tag_class}' (ID info: {final_numeric_id}): {e}")
                    # current_class_description remains tag_class (already set as fallback)
                    pass
            
            tag_entry = {
                "classDescription": current_class_description, # MODIFIED: Use classDescription
                "id": final_numeric_id, # Use the determined numeric ID
                "yoloClass": tag_class,
                "timecodes": tag_timecodes
            }
            # Add refined_track_id if it exists and we used refined tags, for traceability
            if self.using_refined_tags and tag.get("refined_track_id"):
                tag_entry["refined_track_id_str"] = tag.get("refined_track_id")


            if tag_class == "person":
                described_tags["persons"].append(tag_entry)
            else:
                described_tags["objects"].append(tag_entry)
        return described_tags

    def _generate_segment_prompt(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig],
        frames_subset: List[str], # List of frame paths/URLs for this chunk/segment
        times_subset: List[float],  # Corresponding timestamps for frames_subset
        copyright_json_str: Optional[str] = None, # Pass through if needed by prompt
        yolo_tags_for_segment: Optional[List[Dict[str, Any]]] = None, # Active raw/refined YOLO tags
        previous_segment_summary: Optional[str] = None # For sequential analysis
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Generates the prompt messages for a segment (or a chunk of it).
        Uses frame paths/URLs and includes yolo_tags_for_segment context.
        """
        if not frames_subset:
            logger.warning(f"Segment {segment.id}: No frames provided to _generate_segment_prompt. Skipping prompt generation.")
            return None

        transcription_context = segment.transcription if segment.transcription and segment.transcription.strip() else "No transcription available for this segment."
        messages = []
        
        # System Prompt (generic, from analysis_config)
        # This needs to be adapted based on whether it's for chapters, tags, or a combined call
        # Assuming analysis_config provides a general system prompt template that can take context.
        system_prompt_template_str = getattr(analysis_config, 'system_prompt_template', "You are a helpful AI assistant analyzing video content.")
        try:
            # Basic formatting, analysis_config might have more specific keys to format
            system_prompt = system_prompt_template_str.format(
                lens_prompt=getattr(analysis_config, 'lens_prompt', ''),
                results_template_str_for_reminder=json.dumps(getattr(analysis_config, 'results_template', {}), indent=2)
                # Add other common formatting keys as needed by your configs
            )
        except KeyError as e:
            logger.error(f"KeyError formatting system prompt for segment {segment.id}: {e}. Using template as is.")
            system_prompt = system_prompt_template_str
        messages.append({"role": "system", "content": system_prompt})

        # User Content Construction
        user_content_parts = []

        # 1. Segment Context & Instructions
        user_content_parts.append({
            "type": "text",
            "text": f"Analyzing video segment from {segment.start_time:.2f}s to {segment.end_time:.2f}s (Duration: {segment.end_time - segment.start_time:.2f}s)."
        })
        if previous_segment_summary: # For sequential analysis
            user_content_parts.append({"type": "text", "text": f"Summary of immediately preceding segment: {previous_segment_summary}"})
        
        # Add lens prompt if provided by config
        lens_prompt_text = getattr(analysis_config, 'lens_prompt', None)
        if lens_prompt_text:
            user_content_parts.append({"type": "text", "text": f"Focus of this analysis (lens):\n{lens_prompt_text}"}) 

        # 2. Transcription
        user_content_parts.append({"type": "text", "text": f"Audio Transcription for this segment:\n{transcription_context}"}) 

        # 3. YOLO Context (Raw or Refined)
        if yolo_tags_for_segment:
            yolo_intro_text = f"The following {'refined' if self.using_refined_tags else 'raw'} YOLO tracks are active in this segment:"
            user_content_parts.append({"type": "text", "text": yolo_intro_text})
            yolo_tag_details_list = []
            for i, y_tag in enumerate(yolo_tags_for_segment):
                tag_id_field = "refined_track_id" if self.using_refined_tags else "id"
                tag_id = y_tag.get(tag_id_field, y_tag.get("id", f"track_{i}")) # Fallback ID
                tag_class = y_tag.get("class", "unknown")
                timecodes_str_parts = []
                for tc in y_tag.get("timecodes", []):
                    timecodes_str_parts.append(f"{tc.get('start',0):.2f}s-{tc.get('end',0):.2f}s")
                timecodes_str = ", ".join(timecodes_str_parts) if timecodes_str_parts else "N/A"
                
                detail = f"  - ID: {tag_id}, Class: {tag_class}, Active during: [{timecodes_str}]"
                # Optionally add representative thumb if available and prompt allows text references to it
                # if y_tag.get("thumb"):
                #     detail += f" (Ref: {os.path.basename(y_tag['thumb'])})" 
                yolo_tag_details_list.append(detail)
            
            user_content_parts.append({"type": "text", "text": "\n".join(yolo_tag_details_list)})
        else:
            user_content_parts.append({"type": "text", "text": "No pre-identified YOLO tracks are active in this specific segment or provided as context."})

        # 4. Frames
        user_content_parts.append({"type": "text", "text": "\nVideo Frames for Analysis:"})
        for i, frame_path_or_url in enumerate(frames_subset):
            timestamp = times_subset[i] if i < len(times_subset) else segment.start_time + (i * (1/self.manifest.processing_params.fps))
            user_content_parts.append({"type": "text", "text": f"Frame at {timestamp:.3f}s:"})
            if frame_path_or_url.startswith("http://") or frame_path_or_url.startswith("https://"):
                user_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": frame_path_or_url, "detail": self.image_detail_level}
                })
            else: # Assume local path, needs base64 encoding
                try:
                    # logger.debug(f"Encoding local frame {frame_path_or_url} for prompt.")
                    # This part was from cobra_utils, ensure encode_image_base64 is available or reimplemented
                    # For now, directly call a simplified local version if not importing cobra_utils here.
                    with open(frame_path_or_url, "rb") as image_file:
                        b64_img = base64.b64encode(image_file.read()).decode('utf-8')
                    user_content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": self.image_detail_level}
                    })
                except Exception as e:
                    logger.error(f"Error encoding local frame {frame_path_or_url} for segment {segment.id}: {e}")
                    user_content_parts.append({"type": "text", "text": f"[Error displaying frame at {timestamp:.3f}s]"})
        
        messages.append({"role": "user", "content": user_content_parts})
        
        # 5. Final reminder about output format (from analysis_config)
        results_template_reminder = getattr(analysis_config, 'results_template', None)
        if results_template_reminder:
            final_reminder_text = (
                "CRITICAL: Your output MUST be valid JSON strictly matching this schema. "
                "Return your JSON inside ```json ... ``` and nothing else.\n"
                f"Schema:\n```json\n{json.dumps(results_template_reminder, indent=2)}\n```"
            )
            # Append as a new user message or to the last text part of the previous user message
            # Appending as a new message can sometimes be clearer for the LLM.
            messages.append({"role": "user", "content": final_reminder_text})
            # Or, if appending to existing user_content_parts:
            # user_content_parts.append({"type": "text", "text": final_reminder_text})
            # messages[-1]["content"] = user_content_parts

        return messages

    @staticmethod
    def _safe_format(tmpl: str, **kwargs) -> str:
        """
        Like str.format(**kwargs) but leaves {unknown} untouched instead of
        raising KeyError – perfect for prompt templates that contain literal
        braces (JSON examples).
        """
        class _KeepMissing(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return Formatter().vformat(tmpl, (), _KeepMissing(kwargs))


    def _get_list_definitions(self, list_data: Optional[Dict], list_type: str) -> str:
         """Helper to extract definitions or instructions from loaded lists.
         Now prioritizes the 'instructions' key directly from the loaded data,
         assuming the file itself represents the category (person, object, action).
         """
         default_text = f"No specific {list_type} definitions provided."
         if not isinstance(list_data, dict):
              return default_text

         # --- UPDATED LOGIC ---
         # Prioritize the top-level 'instructions' key from the specific file.
         instructions = list_data.get("instructions")
         if instructions and isinstance(instructions, str) and instructions.strip():
              return instructions.strip()
         # --- END UPDATED LOGIC ---

         # Fallback if 'instructions' key is missing or empty
         return default_text

    # ----------------------------------------------------------------------
    # ‼️ NEW – generic helpers used by the merge/aggregate logic
    # ----------------------------------------------------------------------
    @staticmethod
    def _timecodes_to_intervals(timecodes: list) -> list[list[float]]:
        """
        Normalises whatever the LLM gave us into [start, end] float pairs.
        Accepts:
            • {"start": "...s","end":"...s"} dicts  (preferred)
            • plain floats / ints                  (single frame → 1 ms span)
            • "12.345s" strings
        Returns a list of [start, end] pairs (floats, 3 dp).
        """
        intervals = []
        for tc in timecodes:
            if isinstance(tc, dict) and "start" in tc and "end" in tc:
                try:
                    s = float(str(tc["start"]).rstrip("s")); e = float(str(tc["end"]).rstrip("s"))
                    if s > e: s, e = e, s
                    intervals.append([round(s, 3), round(e, 3)])
                except Exception:
                    continue
            elif isinstance(tc, (int, float)):
                s = round(float(tc), 3)
                intervals.append([s, s + 0.001])
            elif isinstance(tc, str) and tc.endswith("s"):
                try:
                    s = round(float(tc[:-1]), 3)
                    intervals.append([s, s + 0.001])
                except Exception:
                    continue
        return intervals
    def _merge_overlapping(intervals: list[list[float]], max_gap: float = 2.0) -> list[list[float]]:
        """
        Merges touching / overlapping / almost‑touching intervals.
        `max_gap` is the maximum gap (in seconds) that will still be bridged.
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0][:]]
        for cur_start, cur_end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            # Use a small tolerance for float comparison
            if cur_start - prev_end <= max_gap + 1e-6: 
                merged[-1][1] = max(prev_end, cur_end)
            else:
                merged.append([cur_start, cur_end])
        return merged

    def _aggregate_tag_timestamps(
        self,
        tag_data: dict[str, list[dict[str, float]]],
        max_gap_initial: float = 2.0,
    ) -> dict[str, list[dict[str, float]]]:
        """
        tag_data  ➜ {"Tag Name": [{"start":..,"end":..}, ...], ...}
        Returns the same structure but with merged/cleaned intervals.
        """
        out: dict[str, list[dict[str, float]]] = {}
        for name, intervals in tag_data.items():
            # normalise to list[list[float,float]]
            num_ints = []
            for i in intervals:
                if isinstance(i, dict) and "start" in i and "end" in i:
                    try:
                         # Handle potential None values gracefully
                         start_val = i.get("start")
                         end_val = i.get("end")
                         if start_val is not None and end_val is not None:
                              s = round(float(start_val), 3)
                              e = round(float(end_val), 3)
                              num_ints.append([s, e])
                         else:
                              logger.warning(f"Skipping interval with None start/end for tag '{name}': {i}")
                    except (ValueError, TypeError) as e:
                         logger.warning(f"Skipping invalid interval for tag '{name}': {i}. Error: {e}")

            merged = self._merge_overlapping(num_ints, max_gap=max_gap_initial)
            out[name] = [{"start": s, "end": e} for s, e in merged]
        return out

    def _parse_llm_json_response(self, raw_content_str: str, expecting_chapters=True, expecting_tags=True):
        """Parses the LLM response string, expecting JSON in a markdown block."""
        try:
            if not raw_content_str or not isinstance(raw_content_str, str):
                logger.error("Invalid raw_content_str provided to _parse_llm_json_response.")
                return {"error": "Invalid raw_content_str for parsing"}
            log_identifier = "Chapters" if expecting_chapters else "Tags"
            code_to_parse = raw_content_str.strip()
            if code_to_parse.startswith("```json"):
                code_to_parse = code_to_parse.split("```json", 1)[1].strip()
            if code_to_parse.endswith("```"):
                code_to_parse = code_to_parse.rsplit("```", 1)[0].strip()
            try:
                parsed_response = json.loads(code_to_parse)
                if not isinstance(parsed_response, dict):
                    logger.error("Parsed JSON is not a dictionary.")
                    return {"error": "Parsed JSON is not a dictionary"}
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from LLM response string: {e}")
                logger.error(f"Content attempted: {code_to_parse[:500]}...")
                return {"error": f"JSONDecodeError: {e}", "raw_content_attempted": code_to_parse[:500]}
        except Exception as e:
            logger.error(f"Error parsing LLM response string: {e}")
            logger.error(f"Failed parsing raw content string:\n{raw_content_str[:1000]}...\n")
            return {"error": f"General parsing error: {e}"}

    def _count_tokens(self, text, model="gpt-3.5-turbo"):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return 0

    def _count_image_tokens(self, images_base64):
        # Rough estimate: 1 token per ~4 chars for base64
        if not images_base64:
            return 0
        if isinstance(images_base64, list):
            total_chars = sum(len(img) for img in images_base64 if isinstance(img, str))
        elif isinstance(images_base64, str):
            total_chars = len(images_base64)
        else:
            total_chars = 0
        return total_chars // 4

    def _call_llm(self, messages, model="gpt-3.5-turbo", log_token_category=None):
        # Revert to user's original structure for calling Azure/OpenAI models
        # Use AzureOpenAI or AsyncAzureOpenAI clients as before
        # Token logging and counting logic is preserved
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("AzureOpenAI client not found. Please ensure the openai package < 1.0.0 or use the Azure SDK as before.")
        client = AzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )

        # --- EDIT: Conditionally set max tokens parameter ---
        completion_params = {}
        max_token_value = 5000 # Restore previous max tokens to allow larger completions
        if self.env.vision.deployment == "o4-mini":
             completion_params["max_completion_tokens"] = max_token_value
             logger.debug(f"Using 'max_completion_tokens={max_token_value}' for model '{self.env.vision.deployment}'")
        else:
             completion_params["max_tokens"] = max_token_value
             logger.debug(f"Using 'max_tokens={max_token_value}' for model '{self.env.vision.deployment}'")
        # --- END EDIT ---

        # ---- SIZE GUARD ----
        raw_body_len = len(json.dumps({"model": self.env.vision.deployment, "messages": messages}))
        if raw_body_len > 1_000_000:
            raise RuntimeError(
                f"Request body {raw_body_len:,} B exceeds Azure 1 MB limit – cut frames or switch to image URLs."
            )
        # ---- END SIZE GUARD ----

        # 1️⃣  Estimate image tokens *before* we send the request
        est_img_tokens = estimate_image_tokens(messages)

        response = client.chat.completions.create(
            model=self.env.vision.deployment,
            messages=messages,
            # --- EDIT: Use the dynamically created params dict ---
            **completion_params
            # --- END EDIT ---
        )
        
        # 2️⃣  Normal token bookkeeping as you already do
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else prompt_tokens + completion_tokens
        
        # 3️⃣  If Azure did NOT return image_tokens, inject the estimate
        image_tokens = getattr(response.usage, "image_tokens", None)
        if image_tokens is None:
            image_tokens = est_img_tokens          # ← our estimate
            
        total_with_images = total_tokens # This is the total from the API response.usage.total_tokens

        if log_token_category:
            # Map generic category → the three detailed counters you already track
            prompt_key      = f"{log_token_category}_prompt_tokens"
            completion_key  = f"{log_token_category}_completion_tokens"
            image_key       = f"{log_token_category}_image_tokens"

            # Ensure the dict has those keys (lets you add new categories without edits elsewhere)
            for k_init in (prompt_key, completion_key, image_key): # Renamed k to k_init to avoid conflict
                self.token_usage.setdefault(k_init, 0)

            self.token_usage[prompt_key]     += prompt_tokens
            self.token_usage[completion_key] += completion_tokens
            self.token_usage[image_key]      += image_tokens

            # Grand totals used by the final report
            self.token_usage["report_input_text_tokens"]  += prompt_tokens # prompt_tokens from API includes image tokens
            self.token_usage["report_output_text_tokens"] += completion_tokens
            self.token_usage["report_input_image_tokens"] += image_tokens # This is now from API or fallback
            self.token_usage["report_total_tokens"]       += total_with_images # This is total_tokens from API
            
            logger.debug(f"[TOKENS] {log_token_category}: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}. Updated detailed and report totals.")
        else:
            logger.debug(f"[TOKENS] Uncategorized: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}")
        return response

    async def _call_llm_async(self, messages, model="gpt-3.5-turbo", log_token_category=None):
        # Revert to user's original structure for async Azure/OpenAI calls
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            raise ImportError("AsyncAzureOpenAI client not found. Please ensure the openai package < 1.0.0 or use the Azure SDK as before.")
        client = AsyncAzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )

        # --- EDIT: Conditionally set max tokens parameter ---
        completion_params = {}
        max_token_value = 5000 # Restore previous max tokens to allow larger completions
        if self.env.vision.deployment == "o4-mini":
             completion_params["max_completion_tokens"] = max_token_value
             logger.debug(f"Using 'max_completion_tokens={max_token_value}' for model '{self.env.vision.deployment}' (async)")
        else:
             completion_params["max_tokens"] = max_token_value
             logger.debug(f"Using 'max_tokens={max_token_value}' for model '{self.env.vision.deployment}' (async)")
        # --- END EDIT ---

        # ---- SIZE GUARD ----
        raw_body_len = len(json.dumps({"model": self.env.vision.deployment, "messages": messages}))
        if raw_body_len > 1_000_000:
            raise RuntimeError(
                f"Request body {raw_body_len:,} B exceeds Azure 1 MB limit – cut frames or switch to image URLs."
            )
        # ---- END SIZE GUARD ----

        # 1️⃣  Estimate image tokens *before* we send the request
        est_img_tokens = estimate_image_tokens(messages)

        response = await client.chat.completions.create(
            model=self.env.vision.deployment,
            messages=messages,
            # --- EDIT: Use the dynamically created params dict ---
            **completion_params
            # --- END EDIT ---
        )

        # 2️⃣  Normal token bookkeeping as you already do
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else 0
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else prompt_tokens + completion_tokens

        # 3️⃣  If Azure did NOT return image_tokens, inject the estimate
        image_tokens = getattr(response.usage, "image_tokens", None)
        if image_tokens is None:
            image_tokens = est_img_tokens          # ← our estimate
            
        total_with_images = total_tokens # This is the total from the API response.usage.total_tokens

        if log_token_category:
            # Map generic category → the three detailed counters you already track
            prompt_key      = f"{log_token_category}_prompt_tokens"
            completion_key  = f"{log_token_category}_completion_tokens"
            image_key       = f"{log_token_category}_image_tokens"

            # Ensure the dict has those keys (lets you add new categories without edits elsewhere)
            for k_init in (prompt_key, completion_key, image_key): # Renamed k to k_init to avoid conflict
                self.token_usage.setdefault(k_init, 0)

            self.token_usage[prompt_key]     += prompt_tokens
            self.token_usage[completion_key] += completion_tokens
            self.token_usage[image_key]      += image_tokens

            # Grand totals used by the final report
            self.token_usage["report_input_text_tokens"]  += prompt_tokens # prompt_tokens from API includes image tokens
            self.token_usage["report_output_text_tokens"] += completion_tokens
            self.token_usage["report_input_image_tokens"] += image_tokens # This is now from API or fallback
            self.token_usage["report_total_tokens"]       += total_with_images # This is total_tokens from API
            
            logger.debug(f"[TOKENS] {log_token_category}: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}. Updated detailed and report totals.")
        else:
            logger.debug(f"[TOKENS] Uncategorized: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}")
        return response

    # --- ADDED: Cleanup Helper ---
    def _cleanup_temp_blobs(self):
        logger.info("Starting cleanup of temporary frame blobs...")
        cleanup_start_time = time.time()
        deleted_count = 0
        failed_count = 0
        try:
             if not self.env.blob_storage or not self.env.blob_storage.account_name or \
                not self.env.blob_storage.container_name or \
                (not self.env.blob_storage.connection_string and not self.env.blob_storage.sas_token):
                  logger.info("Blob storage not configured. Skipping cleanup.")
                  return
             blob_service_client = None
             if self.env.blob_storage.connection_string:
                 connect_str = self.env.blob_storage.connection_string.get_secret_value()
                 blob_service_client = BlobServiceClient.from_connection_string(connect_str)
             elif self.env.blob_storage.sas_token:
                  logger.warning("Attempting blob cleanup using SAS token. This might fail if token lacks delete permissions.")
                  account_url = f"https://{self.env.blob_storage.account_name}.blob.core.windows.net"
                  sas_token_str = self.env.blob_storage.sas_token.get_secret_value()
                  blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token_str)
             else:
                  logger.warning("No valid Blob Storage auth for cleanup.")
                  return
             container_client = blob_service_client.get_container_client(self.env.blob_storage.container_name)
             logger.info(f"Connected to container: {self.env.blob_storage.container_name}")
             for seg in self.manifest.segments:
                  blob_names_to_delete = getattr(seg, "_blob_names", [])
                  if blob_names_to_delete:
                       logger.info(f"Cleaning up {len(blob_names_to_delete)} blobs for {seg.segment_name}...")
                       for blob_name in blob_names_to_delete:
                            if not blob_name: continue
                            try:
                                 logger.info(f"  Deleting blob: {blob_name}")
                                 container_client.delete_blob(blob_name)
                                 deleted_count += 1
                            except Exception as e:
                                 logger.warning(f"  Could not delete blob {blob_name}: {e}")
                                 failed_count += 1
        except Exception as e:
             logger.error(f"ERROR during blob cleanup setup or iteration: {e}")
             failed_count += 1
        finally:
            cleanup_duration = time.time() - cleanup_start_time
            logger.info(f"Blob cleanup finished in {cleanup_duration:.2f}s. Deleted: {deleted_count}, Failed/Skipped: {failed_count}")

    # --- Add async cleanup for internal async client ---
    async def close_internal_async_client(self):
        """Closes the internal async LLM client if it exists."""
        if self._internal_async_client and hasattr(self._internal_async_client, 'aclose'):
            await self._internal_async_client.aclose()
            self._internal_async_client = None

    async def _get_async_llm_client(self):
        """Initializes and returns the internal async LLM client if not already set."""
        if self._internal_async_client is not None:
            return self._internal_async_client
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            raise ImportError("AsyncAzureOpenAI client not found. Please ensure the openai package < 1.0.0 or use the Azure SDK as before.")
        self._internal_async_client = AsyncAzureOpenAI(
            api_key=self.env.vision.api_key.get_secret_value(),
            api_version=self.env.vision.api_version,
            azure_endpoint=self.env.vision.endpoint,
        )
        return self._internal_async_client

    # --- ADDED: Placeholder for _gather_report_data ---
    def _gather_report_data(self, final_results: Dict[str, Any], analysis_name: str, elapsed_time: float, output_path: str) -> Dict[str, Any]:
        """Gathers data for the summary report. Placeholder implementation."""
        logger.info(f"Placeholder: _gather_report_data called for {analysis_name}. Data would be gathered here.")
        # Example data structure
        report_data = {
            "analysis_name": analysis_name,
            "video_name": self.manifest.name,
            "video_duration": self.manifest.source_video.duration,
            "analysis_runtime_seconds": elapsed_time,
            "results_output_path": output_path,
            "total_segments_processed": len(self.manifest.segments),
            "token_usage": self.token_usage,
            "errors_encountered": "No specific errors tracked in this placeholder.", # Placeholder
            "summary_of_findings": final_results.get("summary", "Summary not available."), # or actionSummary.summary
            "key_tags_found": {
                "persons": len(final_results.get("globalTags", {}).get("persons", [])),
                "actions": len(final_results.get("globalTags", {}).get("actions", [])),
                "objects": len(final_results.get("globalTags", {}).get("objects", []))
            }
        }
        return report_data
    # --- END ADDED ---

    # --- ADDED: Placeholder for _write_summary_report ---
    def _write_summary_report(self, report_data: Dict[str, Any]) -> None:
        """Writes the summary report. Placeholder implementation."""
        report_path = os.path.join(
            self.manifest.processing_params.output_directory,
            # Use a consistent naming convention for the report file
            f"AnalysisReport_{self.manifest.name}_{report_data.get('analysis_name', 'Generic')}.txt"
        )
        logger.info(f"Writing analysis summary report to {report_path}")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("--- Analysis Summary Report (Placeholder) ---\n\n")
                for key, value in report_data.items():
                    if isinstance(value, dict):
                        f.write(f"{key.replace('_', ' ').title()}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key.replace('_', ' ').title()}: {sub_value}\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n--- End of Report ---")
            logger.info(f"Placeholder report written to {report_path}")
        except Exception as e:
            logger.error(f"Placeholder: Failed to write summary report: {e}")
    # --- END ADDED ---

    # --- ADDED: System prompt for granular action extraction ---
    SYSTEM_PROMPT_ACTION_EXTRACTION: ClassVar[str] = """You are ActionDetectorGPT.
Given a sequence of {num_frames} frames from a video segment, referenced as Image #1, Image #2, ..., Image #{num_frames}, along with their precise timestamps relative to the start of the video (e.g., Image #1 at T1.XXXs, Image #2 at T2.XXXs), your task is to identify all distinct actions performed by persons or objects.

Output MUST be ONLY a valid JSON object with a single key "actions", which is a list of action objects.
Each action object must have "classDescription" (a concise description, 3-7 words max, e.g., "Man opening door", "Car turning left") and "timecodes" (a list of {{ "start": "T_start.XXXs", "end": "T_end.XXXs" }}).
- The "start" time is the timestamp of the first frame in this sequence where the action BEGINS or is clearly visible.
- The "end" time is the timestamp of the last frame in this sequence where the action ENDS or is still clearly visible.
- If an action spans multiple provided frames, create a single timecode entry covering its duration within these frames.
- If an action starts before the first frame or ends after the last frame provided in this chunk, use the timestamp of the first/last frame of this chunk respectively for its start/end.
- Base your analysis ONLY on the provided frames and their timestamps.

Example JSON Output:
```json
{{
  "actions": [
    {{
      "classDescription": "Person waving hand",
      "timecodes": [{{ "start": "10.123s", "end": "11.456s" }}]
    }},
    {{
      "classDescription": "Dog barking",
      "timecodes": [{{ "start": "12.000s", "end": "12.500s" }}]
    }}
  ]
}}
```
If no actions are detected in the provided frames, return:
```json
{{
  "actions": []
}}
```
"""
    # --- END ADDED ---

    async def _extract_actions_per_segment_async(self, segment: Segment) -> List[Dict[str, Any]]:
        """
        Extracts actions from a segment by processing its frames in small chunks.
        Uses a dedicated LLM prompt for fine-grained action detection.
        """
        all_segment_actions = []
        if not segment.segment_frames_file_path and not segment.frame_urls:
            logger.warning(f"Segment {segment.segment_name} has no frames or frame URLs for action extraction.")
            return []

        frame_sources = segment.frame_urls if segment.frame_urls else segment.segment_frames_file_path
        timestamps = segment.segment_frame_time_intervals

        if len(frame_sources) != len(timestamps):
            logger.error(f"Mismatched frame sources ({len(frame_sources)}) and timestamps ({len(timestamps)}) for segment {segment.segment_name}. Skipping action extraction.")
            return []

        if not frame_sources:
            return []
            
        logger.info(f"Starting granular action extraction for segment {segment.segment_name} with {len(frame_sources)} frames.")

        for i in range(0, len(frame_sources), self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE):
            chunk_frame_sources = frame_sources[i:i + self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE]
            chunk_timestamps = timestamps[i:i + self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE]

            if not chunk_frame_sources:
                continue

            messages = [{"role": "system", "content": self.SYSTEM_PROMPT_ACTION_EXTRACTION.format(num_frames=len(chunk_frame_sources))}]
            
            prompt_content = []
            prompt_content.append(f"Analyze the following {len(chunk_frame_sources)} frames for actions:")

            frame_details_for_prompt = []
            current_images_base64 = [] # For token counting if needed, and sending to LLM

            for idx, (frame_source, ts) in enumerate(zip(chunk_frame_sources, chunk_timestamps)):
                image_ref = f"Image #{idx + 1}"
                frame_details_for_prompt.append(f"{image_ref} at {ts:.3f}s")

                if segment.frame_urls: # Using URLs
                    current_images_base64.append({"type": "image_url", "image_url": {"url": frame_source, "detail": self.image_detail_level}})
                else: # Using local file paths, encode to base64
                    try:
                        base64_image = encode_image_base64(frame_source)
                        if base64_image:
                            current_images_base64.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": self.image_detail_level}})
                        else:
                            logger.warning(f"Could not encode frame {frame_source} for action extraction.")
                    except Exception as e:
                        logger.error(f"Error encoding frame {frame_source}: {e}")
            
            prompt_content.append("Frame timestamps: " + ", ".join(frame_details_for_prompt))
            
            user_message_content = [{"type": "text", "text": "\\n".join(prompt_content)}]
            user_message_content.extend(current_images_base64)
            messages.append({"role": "user", "content": user_message_content})
            
            try:
                llm_response = await self._call_llm_async(messages, model=self.env.vision.deployment, log_token_category="action_extraction")
                if llm_response and llm_response.choices:
                    raw_content = llm_response.choices[0].message.content
                    parsed_actions_data = self._parse_llm_json_response(raw_content_str=raw_content, expecting_chapters=False, expecting_tags=False) # Modify parsing if it expects "globalTags"
                    
                    if parsed_actions_data and "actions" in parsed_actions_data and isinstance(parsed_actions_data["actions"], list):
                        chunk_actions = parsed_actions_data["actions"]
                        # Ensure timecodes are valid and correctly formatted (e.g., converting to float if necessary for later processing)
                        for action in chunk_actions:
                            if "timecodes" in action and isinstance(action["timecodes"], list):
                                valid_timecodes = []
                                for tc in action["timecodes"]:
                                    try:
                                        start_s = tc["start"].replace("s","")
                                        end_s = tc["end"].replace("s","")
                                        start_time_float = float(start_s)
                                        end_time_float = float(end_s)
                                        # Ensure start <= end, could also validate against chunk_timestamps range
                                        if start_time_float <= end_time_float:
                                            valid_timecodes.append({"start": f"{start_time_float:.3f}s", "end": f"{end_time_float:.3f}s"})
                                        else:
                                            logger.warning(f"Invalid timecode for action '{action.get('classDescription')}': start ({start_time_float}) > end ({end_time_float}). Original: {tc}") # MODIFIED: Use classDescription for log
                                    except (ValueError, KeyError) as e:
                                        logger.warning(f"Malformed timecode {tc} for action '{action.get('classDescription')}': {e}") # MODIFIED: Use classDescription for log
                                action["timecodes"] = valid_timecodes
                        all_segment_actions.extend(chunk_actions)
                        logger.debug(f"Extracted {len(chunk_actions)} actions from chunk {i // self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE} for segment {segment.segment_name}")
                    else:
                        logger.warning(f"Could not parse actions from LLM response for segment {segment.segment_name}, chunk {i // self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE}. Response: {raw_content[:200]}")
                else:
                    logger.warning(f"No valid LLM response for action extraction in segment {segment.segment_name}, chunk {i // self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE}")

            except Exception as e:
                logger.error(f"Error during action extraction for segment {segment.segment_name}, chunk {i // self.ACTION_EXTRACTION_FRAME_CHUNK_SIZE}: {e}")
                # Optionally, implement retries or fallback
        
        logger.info(f"Completed granular action extraction for segment {segment.segment_name}, found {len(all_segment_actions)} actions in total.")
        return all_segment_actions

    # --- ADDED: Method to get active YOLO tags for a segment ---
    def _get_active_yolo_tags_for_segment(self, segment: Segment) -> Optional[List[Dict[str, Any]]]:
        """Determines the active set of YOLO tags (raw or refined) for a given segment."""
        segment_start = segment.start_time
        segment_end = segment.end_time
        active_tags_for_segment = []

        if self.using_refined_tags and self.manifest.refined_yolo_tags:
            logger.debug(f"Attempting to use REFINED YOLO tags for segment {segment.segment_name}.")
            source_tags = self.manifest.refined_yolo_tags
        elif self.manifest.raw_yolo_tags:
            logger.debug(f"Using RAW YOLO tags for segment {segment.segment_name}.")
            source_tags = self.manifest.raw_yolo_tags
        else:
            logger.debug(f"No YOLO tags (raw or refined) available in the manifest for segment {segment.segment_name}.")
            return None # Or an empty list, depending on desired behavior

        if not source_tags:
             return None

        # Filter tags that fall within the segment's time range
        # This assumes tags have timecodes; a more robust check might be needed
        # depending on the structure of your YOLO tag objects.
        # For now, we assume tags have a 'timecodes' list with 'start'/'end'.
        # If your YOLO tags are frame-specific without aggregated timecodes yet,
        # this logic would need to map frame numbers/timestamps to segment boundaries.
        # Given the error context, this method is called per segment, implying segment-level filtering is needed.
        
        # Simplified: If YOLO tags are global, we might pass all of them.
        # If they are already per-segment or need filtering, that logic goes here.
        # For now, let's assume the `active_yolo_tags_source` (set in __init__ or analyze_video)
        # is the correct list to use, and we need to filter it for the segment.
        # The previous setup used `self.active_yolo_tags_source` which was set once.
        # Let's refine this to use the manifest's tags and filter them.

        relevant_tags_for_segment = []
        for tag_group in source_tags: # Assuming source_tags is like [{'frame_ N': [tags]}, ...]
                                     # or a flat list of tags with timestamps.
                                     # Based on Re-ID output, it's likely a flat list of track objects.
            # Example structure for a refined track object:
            # { "id": "person_refined_1", "yoloClass": "person", 
            #   "timecodes": [{"start": 0.0, "end": 1.668}]} 
            if "timecodes" in tag_group and isinstance(tag_group["timecodes"], list):
                for tc in tag_group["timecodes"]:
                    try:
                        # Ensure timecodes are numeric for comparison
                        tag_start = float(str(tc.get("start")).replace("s", ""))
                        tag_end = float(str(tc.get("end")).replace("s", ""))
                        
                        # Check for overlap: (StartA <= EndB) and (EndA >= StartB)
                        if segment_start <= tag_end and segment_end >= tag_start:
                            relevant_tags_for_segment.append(tag_group)
                            break # Add the whole tag group if any part of it overlaps
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning(f"Could not parse timecode {tc} for tag {tag_group.get('id', 'Unknown')} while filtering for segment. Error: {e}")
            else:
                # If tags have a different structure (e.g., single timestamp per tag for a frame)
                # you'd need to check if that timestamp falls within segment_start and segment_end.
                # For now, we assume the timecodes structure from refined YOLO tags.
                pass # Or log a warning about unexpected tag structure

        if relevant_tags_for_segment:
            logger.debug(f"Found {len(relevant_tags_for_segment)} YOLO tags relevant to segment {segment.segment_name}.")
            return relevant_tags_for_segment
        else:
            logger.debug(f"No specific YOLO tags from the active source found for segment {segment.segment_name} based on time overlap.")
            return [] # Return empty list if no relevant tags found

    # --- END ADDED ---

    # --- ADDED METHOD: _finalize_token_usage ---
    def _finalize_token_usage(self):
        """
        Sums up token counts from detailed categories into the top-level generic counters.
        This should be called at the end of the analysis process.
        """
        # Define categories that contribute to the top-level sums
        # Excludes "report_" prefixed keys (as they are already aggregate-like)
        # and the top-level keys themselves.
        detailed_categories = [
            # "chapters", "tags", # Removed as they are no longer in _TOKEN_USAGE_TEMPLATE for separate tracking
            "summary", "reid_linking",
            "segment_generic_analysis", "yolo_describe", "action_extraction"
        ]
        
        sum_detailed_prompt_tokens = 0
        sum_detailed_completion_tokens = 0
        sum_detailed_image_tokens = 0

        for cat_key_base in detailed_categories:
            # Prompt tokens from API calls (for vision models, this includes image cost component)
            sum_detailed_prompt_tokens += self.token_usage.get(f"{cat_key_base}_prompt_tokens", 0)
            sum_detailed_completion_tokens += self.token_usage.get(f"{cat_key_base}_completion_tokens", 0)
            # Explicit image tokens (from API `image_tokens` field or our estimation)
            sum_detailed_image_tokens += self.token_usage.get(f"{cat_key_base}_image_tokens", 0)
        
        # Top-level 'prompt_tokens' reflects the total cost attributed to prompts (text + image components of prompt)
        self.token_usage["prompt_tokens"] = sum_detailed_prompt_tokens
        self.token_usage["completion_tokens"] = sum_detailed_completion_tokens
        # Top-level 'image_tokens' is the sum of all explicit image token costs (image part only)
        self.token_usage["image_tokens"] = sum_detailed_image_tokens
        
        # Top-level 'total_tokens' is the sum of all prompt costs and all completion costs.
        # Since sum_detailed_prompt_tokens already includes the image costs associated with the prompts,
        # we just add sum_detailed_completion_tokens.
        self.token_usage["total_tokens"] = sum_detailed_prompt_tokens + sum_detailed_completion_tokens
        
        logger.info("Finalized top-level generic token counts by summing detailed buckets.")
        logger.debug(f"Final top-level: prompt_tokens (incl. prompt image cost)={self.token_usage['prompt_tokens']}, "
                     f"completion_tokens={self.token_usage['completion_tokens']}, "
                     f"image_tokens (explicit image part only)={self.token_usage['image_tokens']}, "
                     f"total_tokens (sum of prompt & completion costs)={self.token_usage['total_tokens']}")
    # --- END ADDED METHOD ---


