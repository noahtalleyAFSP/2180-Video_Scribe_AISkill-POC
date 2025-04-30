import os
import json
import time
import asyncio
import nest_asyncio
from typing import Union, Type, Optional, List, Dict, Set, Tuple, Literal # Added Literal
from openai import AzureOpenAI, AsyncAzureOpenAI
import logging
from collections import defaultdict
import numpy as np
import math
import re
import tiktoken
from string import Formatter
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from .cobra_utils import seconds_to_iso8601_duration, validate_video_manifest, write_video_manifest, merge_overlapping, convert_string_to_seconds # Adjust import path if needed
from typing import Dict, List
import traceback # ADDED traceback import
import requests
import functools
import httpx

# --- ADDED: Imports for Frame Deduplication ---
import cv2 # Already imported below, ensure it's here
from skimage.metrics import structural_similarity as ssim
# --- END ADDED ---

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
    # --- ADDED: Import scoreboard helpers ---
    # _concat_read_text_scoreboard, # Renamed in utils to avoid conflict
    # parse_scores_from_read_api
    # --- END ADDED ---
)
# --- ADDED: Imports for Read API and Scoreboard ---
# from azure.ai.vision.imageanalysis import ImageAnalysisClient
# from azure.ai.vision.imageanalysis.models import VisualFeatures
# from azure.core.credentials import AzureKeyCredential
# --- END REMOVED ---

logger = logging.getLogger(__name__)

# utils -------------------------------------------------------------
# def _dump_prompt(messages: list[dict], dump_path: str) -> None:
#     """Write the exact JSON body that is sent to the LLM."""
#     try:
#         # Ensure directory exists
#         os.makedirs(os.path.dirname(dump_path), exist_ok=True)
#         with open(dump_path, "w", encoding="utf-8") as f:
#             json.dump(messages, f, indent=2, ensure_ascii=False)
#     except Exception as e:                             # never block the run
#         logger.warning("Prompt dump failed: %s", e)
# -------------------------------------------------------------------

class VideoAnalyzer:
    manifest: VideoManifest
    env: CobraEnvironment
    reprocess_segments: bool
    person_group_id: str
    peoples_list_path: Optional[str]
    peoples_list: Optional[dict]
    emotions_list_path: Optional[str]
    emotions_list: Optional[dict]
    themes_list_path: Optional[str]
    themes_list: Optional[dict]
    actions_list_path: Optional[str]
    actions_list: Optional[dict]
    MAX_FRAMES_PER_PROMPT: int = 45  # Maximum number of frames to send in a single prompt
    token_usage: Dict[str, int]  # Track chapters, tags, summary, total
    TAG_CHUNK_FRAME_COUNT: int = 5 # Consider making this dynamic later
    identified_people_in_segment: Dict[str, List[str]]
    LABEL_BLOCK: str # Added for cached labels

    # Add instance variables to track known tags during analysis
    _current_known_persons: Set[str]
    _current_known_actions: Set[str]

    # ---------------------------------------------------------------------------
    # Tag sets declared at class level
    _current_known_persons: Set[str]
    _current_known_actions: Set[str]
    # ---------------------------------------------------------------------------

    from typing import Dict, List, Set, Optional, Union, Tuple

    # Add storage for instructions
    persons_instructions: Optional[str]
    actions_instructions: Optional[str]

    def __init__(
        self,
        video_manifest: Union[str, VideoManifest],
        env: CobraEnvironment,
        person_group_id: Optional[str] = None,
        peoples_list_path: Optional[str] = None,
        emotions_list_path: Optional[str] = None,
        themes_list_path: Optional[str] = None,
        actions_list_path: Optional[str] = None,
    ):
        # ------------------------------------------------------------
        # 0 · boiler-plate / basic fields
        # ------------------------------------------------------------
        self.manifest = validate_video_manifest(video_manifest)
        self.env = env
        self.person_group_id = person_group_id

        self.identified_people_in_segment: Dict[str, List[str]]  # <- declaration
        self.identified_people_in_segment = {}                    # <- assignment

        # Store paths for prompt building
        self.peoples_list_path = peoples_list_path
        self.actions_list_path = actions_list_path

        # ------------------------------------------------------------
        # 1 · load the optional lookup files
        # ------------------------------------------------------------
        self.peoples_list, self.persons_instructions = self._load_json_list(peoples_list_path)
        self.emotions_list, self.emotions_instructions = self._load_json_list(emotions_list_path)
        self.themes_list,   self.themes_instructions   = self._load_json_list(themes_list_path)
        self.actions_list, self.actions_instructions = self._load_json_list(actions_list_path)

        # fallback string if there are no isntructions in the JSON
        self.emotions_instructions = self.emotions_instructions or "No specific emotion instructions provided."
        self.themes_instructions   = self.themes_instructions   or "No specific theme instructions provided."
        # --- ADDED: Precompute LABEL_BLOCK ---
        allowed_persons_list = sorted(list(self.peoples_list.keys())) if self.peoples_list else []
        allowed_actions_list = sorted(list(self.actions_list.keys())) if self.actions_list else []
        self.LABEL_BLOCK = json.dumps({
            "persons": allowed_persons_list,
            "actions": allowed_actions_list
        }, indent=0, separators=(",",":")) # Compact format
        # --- END ADDED ---

        # ------------------------------------------------------------
        # 2 · Create Canonical Label Sets
        # ------------------------------------------------------------
        # Helper to safely create sets from dictionary keys
        def _create_canonical_set(definitions_dict: Optional[dict]) -> Set[str]:
            if not isinstance(definitions_dict, dict): return set()
            return {label.strip() for label in definitions_dict.keys() if isinstance(label, str)}
        
        self._canonical_persons: Set[str] = _create_canonical_set(self.peoples_list)
        self._canonical_actions: Set[str] = _create_canonical_set(self.actions_list)
        # Keep lower versions if needed for looser matching or future canonicalization
        self._canonical_persons_lower: Set[str] = {p.lower() for p in self._canonical_persons}
        self._canonical_actions_lower: Set[str] = {a.lower() for a in self._canonical_actions}

        # Add defaults for instructions if not loaded
        self.persons_instructions = self.persons_instructions or "No specific person instructions provided."
        self.actions_instructions = self.actions_instructions or "No specific action instructions provided."

        # ------------------------------------------------------------
        # 3 · misc bookkeeping
        # ------------------------------------------------------------
        self.token_usage: Dict[str, int]  # declare
        self.token_usage = {"chapters": 0, "persons_tags": 0, "actions_tags": 0, "summary": 0, "total": 0}

        # quick sanity-check
        print("Loaded peoples_list:",  bool(self.peoples_list))
        print("Loaded actions_list:",  bool(self.actions_list))
        # print("Known persons:",  self._current_known_persons) # Removed old print
        # print("Known actions:",  self._current_known_actions) # Removed old print
        print("Canonical persons:", self._canonical_persons)
        print("Canonical actions:", self._canonical_actions)

    def _load_json_list(self, file_path) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """
        Load a JSON list file which should be a dict with:
        - either a "persons" or "peoples" key mapping labels -> descriptions
        - an optional "instructions" key containing a string
        """
        if not file_path or not os.path.exists(file_path):
            print(f"DEBUG: file_path missing or not found ({file_path})")
            return None, None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return None, None

        if not isinstance(data, dict):
            print(f"Warning: expected a JSON object at top-level in {file_path}")
            return None, None

        # auto-detect key from known categories
        KNOWN_CATEGORIES = ["persons", "peoples", "actions", "emotions", "themes"]
        for candidate in KNOWN_CATEGORIES:
            if candidate in data:
                key = candidate
                break
        else:
            print(f"Warning: JSON in {file_path} has none of {KNOWN_CATEGORIES!r} keys")
            return None, None
        # ---------------------------------------------
        definitions_map = data[key]
        instructions   = data.get("instructions")

        # validate definitions_map
        if not isinstance(definitions_map, dict):
            print(f"Warning: '{key}' in {file_path} is not a dict (is {type(definitions_map)})")
            definitions_map = None
        else:
            # verify all values are strings
            bad = [v for v in definitions_map.values() if not isinstance(v, str)]
            if bad:
                print(f"Warning: some values under '{key}' in {file_path} are not strings")

        # validate instructions
        if instructions is not None and not isinstance(instructions, str):
            print(f"Warning: 'instructions' in {file_path} is not a string (is {type(instructions)})")
            instructions = None

        if definitions_map is None and instructions is None:
            print(f"Warning: nothing valid loaded from {file_path}")
            return None, None

        return definitions_map, instructions


    # Primary method to analyze the video
    def analyze_video(
        self,
        analysis_config: Type[AnalysisConfig],
        run_async=False,
        max_concurrent_tasks=None,
        reprocess_segments=False,
        person_group_id=None,
        copyright_json_str: Optional[str] = None,
        # --- ADDED: Frame Deduplication Control ---
        deduplicate_frames: bool = True, # Default to True
        dedup_threshold: float = 0.9, # SSIM threshold
        dedup_min_time_diff: float = 0.5, # Minimum time difference (seconds)
        # --- END ADDED ---
        **kwargs,
    ):
       
         # Store deduplication settings
        self._deduplicate_frames = deduplicate_frames
        self._dedup_threshold = dedup_threshold
        self._dedup_min_time_diff = dedup_min_time_diff
        print(f"Frame deduplication: {'Enabled' if self._deduplicate_frames else 'Disabled'} (Threshold: {self._dedup_threshold}, Min Time Diff: {self._dedup_min_time_diff}s)")

        # ------------------------------------------------------------------
        # 1 · reset & repopulate the three "known-label" sets  --------------
        # ------------------------------------------------------------------
        # --- REMOVED incorrect inner helper function ---
        # def _populate_known_set(list_dict: Optional[dict], key: str) -> set[str]:
        #     ...
        # --- END REMOVAL ---

        # --- REPLACED with direct population from instance dictionaries ---
        self._current_known_persons = set(self.peoples_list.keys()) if self.peoples_list else set()
        self._current_known_actions = set(self.actions_list.keys()) if self.actions_list else set()
        self._current_known_actions_lower = {a.lower() for a in self._current_known_actions}
        # --- END REPLACEMENT ---

        self._attach_transcripts_to_segments()


        self.reprocess_segments = reprocess_segments
        self.person_group_id = person_group_id

        # Reset token usage for each run
        self.token_usage = {"chapters": 0, "persons_tags": 0, "actions_tags": 0, "summary": 0, "total": 0}

        stopwatch_start_time = time.time()

        print(
            f'Starting video analysis: "{analysis_config.name}" for {self.manifest.name}'
        )

        # If person_group_id is provided, verify it exists
        if self.person_group_id:
            if ensure_person_group_exists(self.person_group_id, self.env):
                print(f"Using face recognition with person group: {self.person_group_id}")
            else:
                print(f"Warning: Person group {self.person_group_id} not found or not accessible")
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
                    print("Running analysis asynchronously...")
                    raw_processed_results = asyncio.run(
                        self._analyze_segment_list_async(
                            analysis_config, max_concurrent_tasks, copyright_json_str
                        )
                    )
                else:
                    print("Running analysis sequentially...")
                    raw_processed_results = self._analyze_segment_list(
                        analysis_config, copyright_json_str
                    )

                # --- PATCH START ---
                if has_custom_aggregation:
                    print("Running custom aggregation method…")
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
            finally:
                self._cleanup_temp_blobs()


            final_results = {} # Initialize the dictionary to be saved as JSON

            if has_custom_aggregation:
                print("Using results from custom aggregation method.")
                # Assign the potentially processed results to final_results
                final_results = processed_results

                # --- Update manifest.global_tags from the actionSummary structure ---
                if "actionSummary" in final_results:
                    action_summary_content = final_results.get("actionSummary", {})
                    # Construct the global_tags structure expected by the manifest
                    # Use camelCase keys from action_summary_content, map to plural keys for manifest
                    # --- FIX: Point to correct nested structure for tags --- 
                    manifest_global_tags = {
                         "persons": action_summary_content.get("globalTags", {}).get("persons", []), # action_summary -> globalTags -> persons
                         "actions": action_summary_content.get("globalTags", {}).get("actions", []), # action_summary -> globalTags -> actions
                    }
                    # --- END FIX ---
                    self.manifest.global_tags = manifest_global_tags
                    print("DEBUG: Updated self.manifest.global_tags using data from actionSummary (persons, actions only).")
                else:
                    # Handle old custom structure (if needed) - unlikely with ActionSummary config
                    print("Warning: Custom aggregation result missing 'actionSummary'. Manifest global_tags not updated.")

            elif isinstance(processed_results, list): # This branch now handles non-custom agg OR refine
                # --- Apply Generic Aggregation to the list of segment results ---
                # This path handles non-custom aggregation and 'refine' sequence outputs.
                print("Running generic aggregation (produces standard structure).")
                try:
                    all_chapters_agg = []
                    global_tags_agg_dict = { "persons": {}, "actions": {} }

                    for result_container in processed_results:
                         if not isinstance(result_container, dict):
                             print(f"Warning: Skipping non-dict item in results list during generic agg: {type(result_container)}")
                             continue
                         # --- EDIT: Check camelCase 'analysisResult' ---
                         segment_response = result_container.get("analysisResult", {})
                         if not isinstance(segment_response, dict):
                              print(f"Warning: Skipping item with non-dict 'analysisResult' during generic agg: {type(segment_response)}")
                              continue

                         # Add chapters ('chapters' key is fine)
                         chapters_data = segment_response.get("chapters", [])
                         if isinstance(chapters_data, dict): all_chapters_agg.append(chapters_data)
                         elif isinstance(chapters_data, list): all_chapters_agg.extend(chap for chap in chapters_data if isinstance(chap, dict))
                         else: print(f"Warning: Unexpected data type for 'chapters' during generic agg: {type(chapters_data)}")

                         # Merge global tags ('globalTags' key is camelCase)
                         # --- EDIT: Check camelCase 'globalTags' ---
                         tags_data = segment_response.get("globalTags", {})
                         if not isinstance(tags_data, dict):
                             print(f"Warning: Unexpected data type for 'globalTags' during generic agg: {type(tags_data)}")
                             continue

                         for category in ["persons", "actions"]: # Internal keys 'persons', etc. are fine
                             if category not in tags_data or not isinstance(tags_data.get(category), list): continue

                             for tag_obj in tags_data[category]:
                                 if not isinstance(tag_obj, dict):
                                     print(f"Warning: Skipping non-dictionary tag in '{category}' during generic agg: {type(tag_obj)}")
                                     continue
                                 name = tag_obj.get("name") # 'name' fine
                                 if not name or not isinstance(name, str) or not name.strip():
                                     print(f"Warning: Skipping tag in '{category}' with missing/invalid name during generic agg: {tag_obj}")
                                     continue

                                 cleaned_name = name.strip()
                                 if cleaned_name not in global_tags_agg_dict[category]:
                                     global_tags_agg_dict[category][cleaned_name] = {"name": cleaned_name, "timecodes": []} # 'name', 'timecodes' fine

                                 timecodes = tag_obj.get("timecodes", []) # 'timecodes' fine
                                 if isinstance(timecodes, list):
                                     valid_timecodes = [tc for tc in timecodes if isinstance(tc, dict) and "start" in tc and "end" in tc] # 'start', 'end' fine
                                     global_tags_agg_dict[category][cleaned_name]["timecodes"].extend(valid_timecodes)
                                 else: print(f"Warning: Unexpected timecode format for tag '{cleaned_name}' during generic agg: {timecodes}")

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
                    # --- FIX: Ensure manifest update uses correct structure ---
                    manifest_global_tags_generic = {
                        "persons": final_global_tags_agg_list.get("persons", []),
                        "actions": final_global_tags_agg_list.get("actions", []),
                        # Add other keys if needed, ensure they exist in final_global_tags_agg_list or default
                    }
                    self.manifest.global_tags = manifest_global_tags_generic
                    # --- END FIX ---
                    print("DEBUG: Updated self.manifest.global_tags from generic aggregation (persons, actions only).")

                except Exception as e:
                    print(f"Error during generic aggregation: {e}")
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
                    print("Warning: No custom aggregation ran. Final results might not match ActionSummary format.")
                    final_results = _intermediate_generic_results # Use generic result as fallback


            # -- Summary Generation & Saving --
            try:
                if hasattr(analysis_config, "run_final_summary") and analysis_config.run_final_summary:
                    # --- EDIT: Target the correct dictionary based on structure ---
                    content_to_summarize = final_results # Default: summarize the whole thing
                    summary_target_dict = final_results # Default target
                    summary_text = "Analysis resulted in empty content."
                    description_text = "No description generated."
                    category_text = "Other" # Default category
                    sub_category_text = None # Default subCategory
                    should_generate_summary = False
                    video_duration = self.manifest.source_video.duration if self.manifest.source_video.duration else 0

                    # --- FIX: Implement _has_real_content check --- 
                    def _has_real_content(d: dict) -> bool:
                        """True if chapters or *any* tag list is non-empty."""
                        if not isinstance(d, dict):
                            return False
                        if d.get("chapters"):                         # plural
                            return True
                        gtags = d.get("globalTags", {})
                        if isinstance(gtags, dict) and (
                            gtags.get("persons") or gtags.get("actions")
                        ):
                            return True
                        return False

                    if "actionSummary" in final_results:
                         action_summary_content = final_results["actionSummary"]
                         content_to_summarize = action_summary_content # Summarize content inside actionSummary
                         summary_target_dict = action_summary_content # Put summary inside actionSummary
                         should_generate_summary = _has_real_content(action_summary_content)
                         if not should_generate_summary:
                              print("Skipping final summary generation due to empty actionSummary content.")
                              # Add default summary/desc to actionSummary
                              summary_target_dict["description"] = description_text
                              summary_target_dict["summary"] = summary_text
                              summary_target_dict["category"] = category_text
                              summary_target_dict["subCategory"] = sub_category_text

                    # Check standard structure (if not actionSummary - e.g., generic fallback)
                    elif _has_real_content(final_results):
                        should_generate_summary = True
                    else: # Standard structure but empty
                        print("Skipping final summary generation due to empty analysis results.")
                        # Add default summary/desc to top level
                        final_results["description"] = description_text
                        final_results["summary"] = summary_text
                        final_results["category"] = category_text
                        final_results["subCategory"] = sub_category_text
                    # --- END FIX ---

                    if should_generate_summary:
                        print(f"Generating summary and description for {self.manifest.name} with sports caster lens...") # Modified log

                        # --- Determine summary length instruction (existing logic) ---
                        if video_duration < 30:
                            summary_length_instruction = "1-3 sentences long."
                        elif 30 <= video_duration <= 300: # 30 seconds to 5 minutes
                            summary_length_instruction = "1-2 paragraphs long."
                        else: # Over 5 minutes
                            summary_length_instruction = "at least 3 paragraphs long."
                        print(f"Video duration {video_duration:.2f}s. Requesting summary length: {summary_length_instruction}")

                        # --- Format the prompt from AnalysisConfig (existing logic) ---
                        asset_cats_str = "Unknown"
                        if hasattr(analysis_config, 'ASSET_CATEGORIES') and isinstance(analysis_config.ASSET_CATEGORIES, list):
                             asset_cats_str = ", ".join([f'"{cat}"' for cat in analysis_config.ASSET_CATEGORIES])
                        else:
                             print("Warning: Could not find ASSET_CATEGORIES list in AnalysisConfig. Prompt will be incomplete.")

                        # --- MODIFICATION: Use _safe_format to avoid errors if placeholders missing ---
                        summary_prompt_template = getattr(analysis_config, 'summary_prompt', "Default summary prompt.") # Get template
                        try:
                            formatted_summary_prompt = self._safe_format(
                                summary_prompt_template, # Use the template from config
                                summary_length_instruction=summary_length_instruction,
                                asset_categories_list=asset_cats_str # Inject formatted categories
                            )
                        except Exception as e:
                            print(f"ERROR: Failed to format summary prompt: {e}. Using template as is.")
                            formatted_summary_prompt = summary_prompt_template # Fallback
                        # --- END MODIFICATION ---

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
                                # ... (existing JSON parsing for description/summary) ...
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
                                    print(f"Successfully parsed category ('{summary_target_dict['category']}') and subCategory ('{summary_target_dict['subCategory']}') from LLM response.")
                                    # --- END ADDED ---
                                    print("Successfully parsed description and summary from LLM response.")
                                else:
                                    print("Warning: LLM response for summaries was not a JSON dictionary.")
                                    summary_text = raw_response_content

                            except json.JSONDecodeError as json_e:
                                print(f"Warning: Failed to parse JSON summary/description response: {json_e}")
                                print(f"Raw response: {raw_response_content}")
                                summary_text = raw_response_content

                        except Exception as llm_e:
                            print(f"Warning: LLM call for summary/description failed: {llm_e}")

                        # --- ADD keys to the target dictionary ---
                        # --- EDIT: Use camelCase keys 'description', 'summary' ---
                        summary_target_dict["description"] = description_text
                        summary_target_dict["summary"] = summary_text
                        # --- FIX: Update category/subcategory assignment to use target dict ---
                        if category_text and isinstance(category_text, str):
                             summary_target_dict["category"] = category_text.strip()
                        else:
                             summary_target_dict["category"] = "Other" # Default if missing/invalid
                        if sub_category_text and isinstance(sub_category_text, str):
                             summary_target_dict["subCategory"] = sub_category_text.strip()
                        else:
                             summary_target_dict["subCategory"] = None # Ensure null if missing/invalid
                        # --- END FIX ---
                        print(f"DEBUG: Added description, summary, category ('{summary_target_dict.get('category')}') and subCategory ('{summary_target_dict.get('subCategory')}') to {'actionSummary object' if 'actionSummary' in final_results else 'top level'}.")

                    # Update the manifest's final summary field (use the main summary text)
                    # The manifest field itself is final_summary (snake_case)
                    self.manifest.final_summary = summary_text


                # Save the results
                final_results_output_path = os.path.join(
                    self.manifest.processing_params.output_directory,
                    f"_{analysis_config.name}.json",
                )
                print(f"Writing final results structure to {final_results_output_path}")
                os.makedirs(os.path.dirname(final_results_output_path), exist_ok=True)
                with open(final_results_output_path, "w", encoding="utf-8") as f:
                     json_obj = json.loads(json.dumps(final_results))
                     f.write(json.dumps(json_obj, indent=4, ensure_ascii=False))

            except Exception as e:
                print(f"Error during final summary generation or saving results: {e}")
                # ... (existing error handling for final step) ...
                raise ValueError(f"Failed during summary/saving. Original error: {e}")

            stopwatch_end_time = time.time()
            elapsed_time = stopwatch_end_time - stopwatch_start_time

            print(
                f'Video analysis completed in {round(elapsed_time, 3)}: "{analysis_config.name}" for {self.manifest.name}'
            )

        # --- Inject runtime into action summary if ActionSummary config is used ---
        # --- FIX: Don't re-call process_segment_results. Update existing dict. ---
        # If we already ran custom aggregation (which populates final_results as a dict)
        # and it's the ActionSummary config, just update its runtime fields.
        if (hasattr(analysis_config, "process_segment_results") and
            analysis_config.name == "ActionSummary" and
            isinstance(final_results, dict) and
            "actionSummary" in final_results):

            print("DEBUG: Injecting final runtime into existing actionSummary.")
            action_summary_dict = final_results.get("actionSummary", {}) # Get the inner dict

            # Check if action_summary_dict is actually a dictionary before updating
            if isinstance(action_summary_dict, dict):
                 # Use seconds_to_iso8601_duration if available, otherwise format manually
                 try:
                     from .cobra_utils import seconds_to_iso8601_duration
                     action_summary_dict["runtime"] = seconds_to_iso8601_duration(elapsed_time)
                 except ImportError:
                     print("Warning: cobra_utils not found for runtime formatting. Using basic format.")
                     action_summary_dict["runtime"] = f"PT{elapsed_time:.3f}S"

                 action_summary_dict["runtimeSeconds"] = round(elapsed_time, 3)
                 # No need to reassign final_results['actionSummary'] if it's a mutable dict
            else:
                 print(f"Warning: Expected 'actionSummary' to be a dict, but got {type(action_summary_dict)}. Cannot inject runtime.")

        # --- END FIX ---

        # ------------------------------------------------------------------
        # 4 · Use the aggregated results (no extra summary step needed here)
        # ------------------------------------------------------------------
        summary_target_dict = {} # Default in case actionSummary is missing

        if "actionSummary" in final_results:
            action_summary_content = final_results["actionSummary"]
            # Use the actionSummary content directly as it already contains the final summary, desc, etc.
            summary_target_dict = action_summary_content
            print("DEBUG: Using actionSummary content generated by process_segment_results.")
        else:
             # This case should ideally not happen if ActionSummary config is used
             print("ERROR: 'actionSummary' key not found in final_results. Cannot process summary.")
             # Create a basic fallback structure if absolutely needed
             summary_target_dict = {
                "description": "Error: actionSummary missing.",
                "summary": "Error: actionSummary missing.",
                "category": "Error",
                "globalTags": {"persons": [], "actions": []}, # Add empty tags for structure
                "chapters": [] # Add empty chapters for structure
             }

        # Update the manifest's final_summary field using the summary from actionSummary
        self.manifest.final_summary = summary_target_dict.get("summary", "Summary generation failed or was skipped.")

        # Update manifest globalTags from the actionSummary
        if isinstance(summary_target_dict.get("globalTags"), dict):
            self.manifest.global_tags = summary_target_dict["globalTags"]
            print("DEBUG: Updated self.manifest.global_tags using data from actionSummary.")
        elif "persons" in summary_target_dict or "actions" in summary_target_dict: # Handle older potential structure
             self.manifest.global_tags = {
                  "persons": summary_target_dict.get("persons", []),
                  "actions": summary_target_dict.get("actions", [])
             }
             print("DEBUG: Updated self.manifest.global_tags using fallback keys from actionSummary.")

        # Append the full transcription details if available
        if "transcriptionDetails" in final_results and final_results["transcriptionDetails"]:
            summary_target_dict["transcriptionDetails"] = final_results["transcriptionDetails"]

        # Save the actionSummary content (which is now directly summary_target_dict)
        action_summary_file_path = os.path.join(
            self.manifest.processing_params.output_directory, "_ActionSummary.json"
        )
        try:
            with open(action_summary_file_path, "w", encoding="utf-8") as f:
                 json.dump(summary_target_dict, f, indent=4, ensure_ascii=False)
            print(f"Writing final results structure to {action_summary_file_path}")
        except Exception as e:
            print(f"ERROR: Failed to write _ActionSummary.json: {e}")

        # Print final token usage statistics
        print("--- Token Usage Summary ---")
        print(f"  Chapters: {self.token_usage.get('chapters', 0):,}")
        print(f"  Persons Tags: {self.token_usage.get('persons_tags', 0):,}")
        print(f"  Actions Tags: {self.token_usage.get('actions_tags', 0):,}")
        print(f"  Summary: {self.token_usage.get('summary', 0):,}")
        print(f"  -------------------------")
        print(f"  Total Estimated Tokens (including images): {self.token_usage.get('total', 0):,}")
        print("-------------------------")

        # --- Ensure FINAL manifest write happens HERE --- 
        write_video_manifest(self.manifest)
        # --- End FINAL manifest write --- 

        # Return the enriched results (which now include the final summary within actionSummary)
        return final_results

    def generate_segment_prompts(self, analysis_config: Type[AnalysisConfig]):
        for segment in self.manifest.segments:
            # This likely needs updating if generating prompts separately is still needed
            # For now, prompt generation happens inside analyze_segment methods
            pass
            # self._generate_segment_prompt(segment, analysis_config, generate_chapters=True) # Example

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
            # --- EDIT: Use camelCase keys for the container dict ---
            results_list.append({
                "analysisResult": parsed_response, # camelCase
                "segmentName": segment.segment_name, # camelCase
                "startTime": segment.start_time, # camelCase
                "endTime": segment.end_time,   # camelCase
                "framePaths": segment.segment_frames_file_path, # camelCase
                "fullTranscriptionObject": self.manifest.audio_transcription if not results_list else None # camelCase
             })

        # After all segments have been analyzed
        if hasattr(analysis_config, 'process_segment_results'):
            parsed_copyright = None
            # --- EDIT: Use passed copyright_json_str ---
            # copyright_json_str = kwargs.get('copyright_json_str') # No longer needed from kwargs
            if copyright_json_str:
                try:
                    parsed_copyright = json.loads(copyright_json_str)
                    print("DEBUG (SYNC): Successfully parsed copyright JSON string.")
                except json.JSONDecodeError as e:
                    print(f"Warning (SYNC): Failed to parse copyright JSON string: {e}. Content: {copyright_json_str[:100]}...")
            # --- End Parse ---

            # Pass the list with camelCase keys
            final_results = analysis_config.process_segment_results(
                results_list,
                self.manifest,
                self.env,
                parsed_copyright_info=parsed_copyright
            )
            # --- EDIT: Check for 'actionSummary' and update manifest tags ---
            # The logic to update manifest.global_tags was moved to analyze_video
            # as it needs the final processed result, which is returned here.
            print(f"DEBUG (SYNC): Returning results from custom process_segment_results.")
            return final_results
        else:
             # Return the raw list (with camelCase keys) for analyze_video generic aggregation
             return results_list

    def _analyze_segment_list_sequentially(
        self, analysis_config: Type[SequentialAnalysisConfig]
    ):
        # if the analysis config is not a SequentialAnalysisConfig, raise an error
        if not isinstance(analysis_config, SequentialAnalysisConfig):
            raise ValueError(
                f"Sequential analysis can only be run with an obect that is a subclass of SequentialAnalysisConfig. You have provided an object of type {type(analysis_config)}"
            )

        # Start the timer
        stopwatch_start_time = time.time()

        results_list = []

        for i, segment in enumerate(self.manifest.segments):
            # check if the segment has already been analyzed, if so, skip it
            if (
                self.reprocess_segments is False
                and analysis_config.name in segment.analysis_completed
            ):
                print(
                    f"Segment {segment.segment_name} has already been analyzed, loading the stored value."
                )
                results_list.append(segment.analyzed_result[analysis_config.name])
                continue
            else:
                print(f"Analyzing segment {segment.segment_name}")

            messages = []
            number_of_previous_results_to_refine = (
                analysis_config.number_of_previous_results_to_refine
            )
            # generate the prompt for the segment
            # include the right number of previous results to refine and generate the prompt
            if len(results_list) == 0:
                result_list_subset = None
            if len(results_list) <= number_of_previous_results_to_refine:
                result_list_subset = results_list[: len(results_list)]
            else:
                result_list_subset = results_list[:number_of_previous_results_to_refine]

            result_list_subset_string = json.dumps(result_list_subset)

            # if it's the first segment, generate without the refine prompt; if it is not the first segment, generate with the refine prompt
            if i == 0:
                system_prompt_template = (
                    analysis_config.generate_system_prompt_template(
                        is_refine_step=False
                    )
                )

                system_prompt = system_prompt_template.format(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    segment_duration=segment.segment_duration,
                    number_of_frames=segment.number_of_frames,
                    number_of_previous_results_to_refine=number_of_previous_results_to_refine,
                    video_duration=self.manifest.source_video.duration,
                    analysis_lens=analysis_config.lens_prompt,
                    results_template=analysis_config.results_template,
                    current_summary=result_list_subset_string,
                )
            else:
                system_prompt_template = (
                    analysis_config.generate_system_prompt_template(is_refine_step=True)
                )

                system_prompt = system_prompt_template.format(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    segment_duration=segment.segment_duration,
                    number_of_frames=segment.number_of_frames,
                    number_of_previous_results_to_refine=number_of_previous_results_to_refine,
                    video_duration=self.manifest.source_video.duration,
                    analysis_lens=analysis_config.lens_prompt,
                    results_template=analysis_config.results_template,
                    current_summary=result_list_subset_string,
                )

            messages.append({"role": "system", "content": system_prompt})

            # Form the user prompt with the refine prompt, the audio transcription (if available), and the video frames
            user_content = []
            if segment.transcription is not None:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Audio Transcription for the next {segment.segment_duration} seconds: {segment.transcription}",
                    }
                )
            user_content.append(
                {
                    "type": "text",
                    "text": f"Next are the {segment.number_of_frames} frames from the next {segment.segment_duration} seconds of the video:",
                }
            )
            # Include the frames
            for i, frame in enumerate(segment.segment_frames_file_path):
                frame_time = segment.segment_frame_time_intervals[i]
                base64_image = encode_image_base64(frame)
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Below is the frame at start_time {frame_time} seconds. Use this to provide timestamps and understand time.",
                    }
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                    }
                )

            # add user content to the messages
            messages.append({"role": "user", "content": user_content})

            # write the prompt to the manifest
            # prompt_output_path = os.path.join(
            #     segment.segment_folder_path, f"{segment.segment_name}_prompt.json"
            # )
            #
            # with open(prompt_output_path, "w", encoding="utf-8") as f:
            #     f.write(json.dumps(messages, indent=4))
            #
            # segment.segment_prompt_path = prompt_output_path

            # call the LLM to analyze the segment
            response = self._call_llm(messages, log_token_category="chapters")
            parsed_response = self._parse_llm_json_response(response)

            # append the result to the results list
            results_list.append(parsed_response)
            elapsed_time = time.time() - stopwatch_start_time
            print(
                f"Segment {segment.segment_name} analyzed in {round(elapsed_time, 2)} seconds."
            )

            # update the segment object with the analyzed results
            segment.analyzed_result[analysis_config.name] = parsed_response
            segment.analysis_completed.append(analysis_config.name)

            # update the manifest on disk (allows for checkpointing)
            write_video_manifest(self.manifest)

        elapsed_time = time.time() - stopwatch_start_time
        print(f"Analysis completed in {round(elapsed_time,2)} seconds.")

        return results_list

    async def _analyze_segment_list_async(
        self,
        analysis_config: Type[AnalysisConfig],
        max_concurrent_tasks=None,
        copyright_json_str: Optional[str] = None
    ):
        if max_concurrent_tasks is None:
            # --- THROTTLING: Set a lower default concurrency ---+
            max_concurrent_tasks = 4 # Default to 4 concurrent tasks
            print(f"DEBUG: max_concurrent_tasks not specified, defaulting to {max_concurrent_tasks}")
            # --- END THROTTLING ---
        else:
            max_concurrent_tasks = min(
                int(max_concurrent_tasks), len(self.manifest.segments)
            )

        sempahore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_task(segment):
            async with sempahore:
                parsed_response = await self._analyze_segment_async(segment, analysis_config)
                # --- EDIT: Use camelCase keys for the container dict ---
                return {
                    "analysisResult": parsed_response, # camelCase
                    "segmentName": segment.segment_name, # camelCase
                    "startTime": segment.start_time, # camelCase
                    "endTime": segment.end_time,   # camelCase
                    "framePaths": segment.segment_frames_file_path, # camelCase
                    "fullTranscriptionObject": self.manifest.audio_transcription # camelCase
                }

        async def return_value_task(segment):
            # --- EDIT: Return the analysis result directly --- 
            # Assuming the structure stored in segment.analyzed_result is already correct (or will be fixed by analyze_segment)
            stored_result = segment.analyzed_result or {}
            return {
                "segment_name": segment.segment_name,
                "analysisResult": stored_result # <-- USE CAMELCASE
            }
            # --- END EDIT ---

        segment_task_list = []
        needs_processing = False # Flag to check if any segment actually needs processing
        for segment in self.manifest.segments:
            if (
                self.reprocess_segments is False
                and analysis_config.name in segment.analysis_completed
            ):
                print(
                    f"Segment {segment.segment_name} has already been analyzed, creating value task."
                )
                # --- EDIT: Append a task that returns the structured result, not just the value ---
                # We need the surrounding keys like 'segmentName' etc., if generic aggregation runs later.
                # Let's make it return the same structure as sem_task for consistency.
                # Assume the stored result under analysis_config.name is the 'analysisResult' part.
                stored_result = segment.analyzed_result.get(analysis_config.name, {})
                segment_task_list.append(asyncio.create_task(asyncio.sleep(0, result={
                    "analysisResult": stored_result,
                    "segmentName": segment.segment_name,
                    "startTime": segment.start_time,
                    "endTime": segment.end_time,
                    "framePaths": segment.segment_frames_file_path,
                    "fullTranscriptionObject": self.manifest.audio_transcription
                })))
            else:
                needs_processing = True
                segment_task_list.append(sem_task(segment))

        # --- EDIT: Gather results into a list ---
        results_list = await asyncio.gather(*segment_task_list)
        # Filter out potential None results if tasks failed unexpectedly (though gather should raise)
        results_list = [res for res in results_list if res is not None]


        # --- Aggregation Logic ---
        if hasattr(analysis_config, 'process_segment_results'):
            parsed_copyright_agg = None
            if copyright_json_str:
                try:
                    parsed_copyright_agg = json.loads(copyright_json_str)
                    print("DEBUG (ASYNC): Successfully parsed copyright JSON string.")
                except json.JSONDecodeError as e:
                    print(f"Warning (ASYNC): Failed to parse copyright JSON string: {e}. Content: {copyright_json_str[:100]}...")
            # --- End Parse ---

            # Pass the list with camelCase keys
            final_results_agg = analysis_config.process_segment_results(
                results_list,
                self.manifest,
                self.env,
                parsed_copyright_info=parsed_copyright_agg
            )
            # --- EDIT: Manifest update logic moved to analyze_video ---
            print(f"DEBUG (ASYNC): Returning results from custom process_segment_results.")
            return final_results_agg # Return aggregated results
        else:
             # Return the raw list (with camelCase keys) for analyze_video generic aggregation
             return results_list

    def _analyze_segment(
        self,
        segment: Segment,
        analysis_config: Type[AnalysisConfig]
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed, skipping.")
            # --- EDIT: Return structure consistent with camelCase ---
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": []}}) # Use globalTags

        print(f"Analyzing segment {segment.segment_name} ({segment.start_time:.3f}s - {segment.end_time:.3f}s)")
        stopwatch_segment = time.time()

        max_retries = 3
        initial_delay = 1.0

        # --- Stage 1: Get Chapters ---
        print(f"  - Stage 1: Requesting Chapters...")
        chapters_result = {"chapters": []}
        for attempt in range(max_retries):
            try:
                # --- Subsample frames for chapter prompt --- 
                # Use max_count=10 as requested
                full_frames_ch = segment.frame_urls or segment.segment_frames_file_path
                full_times_ch = segment.segment_frame_time_intervals or []
                frames_for_chapters, times_for_chapters = self._sample_frames(
                    full_frames_ch, full_times_ch, max_count=10
                )
                print(f"  - Stage 1: Using {len(frames_for_chapters)} frames for chapter generation.")

                # --- FIX: Pass correct arguments (Task = Chapters, use subsampled frames) ---
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment,
                    analysis_config=analysis_config,
                    task="chapters", # Specify chapter generation task
                    frames_subset=frames_for_chapters, # Pass sampled frames
                    times_subset=times_for_chapters   # Pass sampled times
                )
                # --- END FIX ---
                if chapter_prompt:
                    chapter_llm_response = self._call_llm(chapter_prompt, log_token_category="chapters")
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                        chapter_llm_response,
                        expecting_chapters=True,
                        expecting_tags=False # Chapters call doesn't expect tags
                    ) # Expects {"chapters": ...}
                    if not chapters_result.get("chapters"): # Check if key has content
                        raw_content_check = chapter_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                             raise ValueError("LLM response parsed to empty chapters, possibly indicating an issue.")

                    print(f"  - Stage 1: Received {len(chapters_result.get('chapters',[]))} chapter(s).")
                    break # Success, exit retry loop
                else:
                    print("  - Stage 1: Skipping chapter generation due to empty prompt.")
                    break # No prompt generated, don't retry

            except Exception as e:
                # ... (retry logic) ...
                 print(f"  - Stage 1: Attempt {attempt + 1}/{max_retries} failed for Chapters. Error: {e}")
                 if attempt + 1 == max_retries:
                     print(f"  - Stage 1: Max retries reached for Chapters. Proceeding without chapters.")
                 else:
                     wait_time = initial_delay * (2 ** attempt) # Exponential backoff
                     print(f"    - Retrying in {wait_time:.1f} seconds...")
                     time.sleep(wait_time)


        # --- Stage 2: Get Tags (Persons & Actions - Combined Call) ---
        print(f"  - Stage 2: Requesting Tags (Persons & Actions) per chunk...")
        all_frame_data_orig = segment.frame_urls if segment.frame_urls else segment.segment_frames_file_path
        all_times_orig = segment.segment_frame_time_intervals
        # Initialize final dict for merged tags for the whole segment
        merged_global_tags = {"persons": [], "actions": []}
        all_frame_data = all_frame_data_orig
        all_times = all_times_orig

        if not all_frame_data_orig:
            print("  - Stage 2: No frames/URLs found for segment. Skipping tag analysis.")
        else:
            # --- ADDED: Frame Deduplication (Sync - using async helper in thread for now) ---
            if self._deduplicate_frames:
                 try:
                      # Run async deduplication in a separate thread for sync context
                      all_frame_data, all_times = asyncio.run(self._unique_frames_async(
                           all_frame_data_orig, all_times_orig, self._dedup_threshold, self._dedup_min_time_diff
                      ))
                      if not all_frame_data: # Handle case where deduplication removed everything
                           print("Warning: Frame deduplication resulted in zero frames. Skipping tag analysis.")
                           all_frame_data = [] # Ensure it's an empty list
                 except Exception as dedup_err:
                      print(f"Error during frame deduplication: {dedup_err}. Proceeding with original frames.")
                      all_frame_data = all_frame_data_orig
                      all_times = all_times_orig
            # --- END ADDED ---

            num_frames = len(all_frame_data)
            if num_frames == 0:
                 print("  - Stage 2: Skipping tag analysis as no frames remain after deduplication.")
            else:
                 # --- ADJUSTED CHUNK SIZE (Example: Dynamic based on segment duration) ---
                 # Example: Aim for roughly one chunk every N seconds (e.g., 10s)
                 # Adjust this logic as needed. Using fixed TAG_CHUNK_FRAME_COUNT for now.
                 # target_chunk_duration = 10.0 # seconds
                 # num_chunks = max(1, round((segment.end_time - segment.start_time) / target_chunk_duration))
                 # dynamic_chunk_frame_count = max(3, math.ceil(num_frames / num_chunks)) # Ensure at least 3 frames
                 # print(f"DEBUG: Dynamic chunk frame count: {dynamic_chunk_frame_count} (based on {num_chunks} chunks)")
                 # current_chunk_frame_count = dynamic_chunk_frame_count
                 current_chunk_frame_count = self.TAG_CHUNK_FRAME_COUNT # Using fixed count for now
                 # --- END ADJUSTMENT ---

                 for i in range(0, num_frames, current_chunk_frame_count):
                     # ------------------------------------------------------------------
                     #  Patch to define chunk variables robustly
                     # ------------------------------------------------------------------
                     chunk_frames = all_frame_data[i:i + current_chunk_frame_count]

                     # Use corresponding times for the kept frames
                     chunk_times = all_times[i:i + current_chunk_frame_count]

                     # derived helpers ---------------------------------------------------
                     if not chunk_times:
                         logger.warning(f"Skipping chunk {i // current_chunk_frame_count + 1} due to empty chunk_times.")
                         continue
                     chunk_idx        = (i // current_chunk_frame_count) + 1
                     chunk_start_time = chunk_times[0]
                     chunk_end_time   = chunk_times[-1]
                     # ------------------------------------------------------------------

                     print(f"    - Processing Tag Chunk {chunk_idx} ({len(chunk_frames)} frames)...")

                     # --- Single Call for Tags ---
                     parsed_tags_for_chunk = {"persons": [], "actions": []} # Default empty
                     # --- FIX: Use specific log category --- 
                     log_cat = "tags" # Default, but we override below
                     # if task == "persons_tags": log_cat = "persons_tags"
                     # elif task == "actions_tags": log_cat = "actions_tags"
                     # --- END FIX --- 

                     for attempt in range(max_retries):
                         try:
                             # --- Generate combined tag prompt ---
                             tag_prompt = self._generate_segment_prompt(
                                 segment=segment,
                                 analysis_config=analysis_config,
                                 task="tags", # Specify tag generation task
                                 frames_subset=chunk_frames,
                                 times_subset=chunk_times,
                                 chunk_start_time=chunk_start_time,
                                 chunk_end_time=chunk_end_time
                             )

                             if tag_prompt:
                                 # --- FIX: Pass correct log_cat --- 
                                 tag_llm_response = self._call_llm(tag_prompt, log_token_category=log_cat)
                                 # --- END FIX ---
                                 # Parse the raw LLM JSON response (expects {"persons": ..., "actions": ...})
                                 parsed_llm_output = self._parse_llm_json_response(
                                     tag_llm_response,
                                     expecting_chapters=False, # Not expecting chapters
                                     expecting_tags=True     # Expecting tags (both types)
                                 )

                                 # --- Filter the raw parsed output ---
                                 # Pass the 'globalTags' dict if present, otherwise the whole dict
                                 tags_to_filter = parsed_llm_output.get("globalTags", parsed_llm_output)
                                 parsed_tags_for_chunk = self._filter_unknowns(tags_to_filter) # Pass the dict containing persons/actions

                                 print(f"        Chunk {chunk_idx}: Received {len(parsed_tags_for_chunk.get('persons',[]))} persons, {len(parsed_tags_for_chunk.get('actions',[]))} actions (after filtering).")
                                 break # Success for this chunk

                             else:
                                 print(f"        Chunk {chunk_idx}: Skipping due to empty prompt.")
                                 break # Exit retry loop

                         except Exception as e:
                             print(f"        Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} failed for tags. Error: {e}")
                             if attempt + 1 == max_retries:
                                 print(f"        Chunk {chunk_idx}: Max retries reached for tags.")
                             else:
                                 wait_time = initial_delay * (2 ** attempt)
                                 print(f"          Retrying chunk tags in {wait_time:.1f} seconds...")
                                 time.sleep(wait_time)
                     # --- End Retry Loop for Chunk ---

                     # --- Collect tags from the successfully processed chunk ---
                     # Extend uses the filtered results stored in parsed_tags_for_chunk
                     merged_global_tags["persons"].extend(parsed_tags_for_chunk.get("persons", []))
                     merged_global_tags["actions"].extend(parsed_tags_for_chunk.get("actions", []))

                     # --- Snap Timestamps AFTER parsing and filtering, BEFORE merging --- 
                     try:
                         # Ensure we have float timestamps for the current chunk for snapping
                         # chunk_times should already be floats
                         valid_chunk_times_float = [t for t in chunk_times if isinstance(t, (int, float))]
                         if not valid_chunk_times_float:
                              print(f"Warning (Snap): No valid float timestamps for chunk {chunk_idx}. Skipping snapping.")
                         else:
                             for tag_type in ("persons", "actions"):
                                 tags_list = parsed_tags_for_chunk.get(tag_type, [])
                                 if not isinstance(tags_list, list): continue

                                 for entry in tags_list:
                                     if isinstance(entry, dict) and isinstance(entry.get("timecodes"), list):
                                         for tc in entry["timecodes"]:
                                             if isinstance(tc, dict) and "start" in tc and "end" in tc:
                                                 try:
                                                     start_snap = self._snap_to_nearest(tc["start"], valid_chunk_times_float)
                                                     end_snap = self._snap_to_nearest(tc["end"], valid_chunk_times_float)
                                                     # Ensure end is not before start after snapping
                                                     if end_snap < start_snap:
                                                          end_snap = start_snap # Set end equal to start if inverted
                                                     tc["start"], tc["end"] = f"{start_snap:.3f}s", f"{end_snap:.3f}s"
                                                 except ValueError as snap_err:
                                                      print(f"Warning (Snap): Could not snap timecode {tc} in chunk {chunk_idx}: {snap_err}")
                                                 except Exception as general_snap_err:
                                                      print(f"Warning (Snap): Unexpected error snapping timecode {tc} in chunk {chunk_idx}: {general_snap_err}")
                     except Exception as outer_snap_err:
                          print(f"Error during timestamp snapping setup for chunk {chunk_idx}: {outer_snap_err}")
                     # --- End Snap Timestamps ---

                 print(f"  - Stage 2: Finished processing all tag chunks for segment.")
        # --- End Main Chunk Loop ---

        # --- Stage 3: Run Scoreboard Analysis (Sync) ---
        print(f"  - Stage 3: Requesting Scoreboard Analysis...")
        scoreboard_events = [] # Default to empty list
        try:
             if self.env.runtime_overrides.get("scorebug"): # Only run if ROI is defined
                scoreboard_events = self._analyze_segment_scoreboard_sync(segment)
                print(f"  - Stage 3: Received {len(scoreboard_events)} scoreboard event(s).")
             else:
                  print("  - Stage 3: Skipping scoreboard analysis - Scorebug ROI not defined in runtime overrides.")
        except Exception as score_ex:
             print(f"  - Stage 3: Scoreboard analysis failed: {score_ex}")
        # --- END Stage 3 ---

        # --- Stage 4: Combine (Chapters + Tags + Scoreboard) ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []),
            "globalTags": merged_global_tags, # Contains ONLY filtered persons and actions from LLM
            "scoreboard_events": scoreboard_events # ADDED scoreboard results here
        }

        # --- ADDED: Shot Oracle Integration (Sync) ---
        # self._run_shot_oracle_sync(segment, final_combined_result['globalTags'])
        # --- END Shot Oracle Integration (Sync) ---

        # --- Stage 5: Update Segment, Save, Update Known Tags ---
        # ... (update segment, save manifest) ...
        # Update known tags logic remains the same (already only iterates persons/actions)

        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis in {elapsed_segment:.2f} seconds.")

        return final_combined_result

    async def _analyze_segment_async(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed (async), skipping.")
            # --- EDIT: Return structure consistent with camelCase ---
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": []}}) # Use globalTags

        print(f"Analyzing segment {segment.segment_name} asynchronously...")
        stopwatch_segment = time.time()

        # --- Retry Parameters ---
        max_retries = 3
        initial_delay = 1.0 # seconds

        # --- Stage 1: Get Chapters (Async) with Retry ---
        print(f"  - Stage 1 (Async): Requesting Chapters...")
        chapters_result = {"chapters": []}
        for attempt in range(max_retries):
            try:
                # --- Subsample frames for chapter prompt --- 
                # Use max_count=10 as requested
                full_frames_ch = segment.frame_urls or segment.segment_frames_file_path
                full_times_ch = segment.segment_frame_time_intervals or []
                frames_for_chapters, times_for_chapters = self._sample_frames(
                    full_frames_ch, full_times_ch, max_count=10
                )
                print(f"  - Stage 1 (Async): Using {len(frames_for_chapters)} frames for chapter generation.")

                # --- FIX: Pass correct arguments (Task = Chapters, use subsampled frames) ---
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment,
                    analysis_config=analysis_config,
                    task="chapters", # Specify chapter generation task
                    frames_subset=frames_for_chapters, # Pass sampled frames
                    times_subset=times_for_chapters   # Pass sampled times
                )
                # --- END FIX --- 
                if chapter_prompt:
                    chapter_llm_response = await self._call_llm_async(chapter_prompt, log_token_category="chapters")
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                         chapter_llm_response,
                         expecting_chapters=True,
                         expecting_tags=False # Chapters call doesn't expect tags
                    ) # Expects {"chapters": ...}
                    # Add check for failed parsing returning default
                    if not chapters_result.get("chapters"):
                        raw_content_check = chapter_llm_response.choices[0].message.content.strip()
                        if not raw_content_check or raw_content_check == "{}":
                            raise ValueError("LLM response parsed to empty chapters, possibly indicating an issue.")

                    print(f"  - Stage 1 (Async): Received {len(chapters_result.get('chapters',[]))} chapter(s).")
                    break # Success
                else:
                    print("  - Stage 1 (Async): Skipping chapter generation due to empty prompt.")
                    break # No prompt

            except Exception as e:
                # ... (retry logic) ...
                print(f"  - Stage 1 (Async): Attempt {attempt + 1}/{max_retries} failed for Chapters. Error: {e}")
                if attempt + 1 == max_retries:
                    print(f"  - Stage 1 (Async): Max retries reached for Chapters. Proceeding without chapters.")
                else:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"    - Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time) # Use asyncio.sleep for async

        # --- Stage 2: Get Tags (Persons & Actions - Combined Call, Async) ---
        print(f"  - Stage 2 (Async): Requesting Tags (Persons & Actions) per chunk...")
        all_frame_data_orig = segment.frame_urls if segment.frame_urls else segment.segment_frames_file_path
        all_times_orig = segment.segment_frame_time_intervals
        merged_global_tags = {"persons": [], "actions": []} # Initialize
        all_frame_data = all_frame_data_orig
        all_times = all_times_orig

        if not all_frame_data_orig:
            print("  - Stage 2 (Async): No frames/URLs found. Skipping tag analysis.")
        else:
            # --- ADDED: Frame Deduplication (Async) ---
            if self._deduplicate_frames:
                try:
                     all_frame_data, all_times = await self._unique_frames_async(
                          all_frame_data_orig, all_times_orig, self._dedup_threshold, self._dedup_min_time_diff
                     )
                     if not all_frame_data:
                          print("Warning (Async): Frame deduplication resulted in zero frames. Skipping tag analysis.")
                          all_frame_data = []
                except Exception as dedup_err:
                     print(f"Error during async frame deduplication: {dedup_err}. Proceeding with original frames.")
                     all_frame_data = all_frame_data_orig
                     all_times = all_times_orig
            # --- END ADDED ---

            num_frames = len(all_frame_data)
            if num_frames == 0:
                 print("  - Stage 2 (Async): Skipping tag analysis as no frames remain after deduplication.")
            else:
                 # --- ADJUSTED CHUNK SIZE (Example - see sync version) ---
                 current_chunk_frame_count = self.TAG_CHUNK_FRAME_COUNT # Using fixed count for now
                 # --- END ADJUSTMENT ---

                 for i in range(0, num_frames, current_chunk_frame_count):
                     # ------------------------------------------------------------------
                     #  Patch to define chunk variables robustly (Async Version)
                     # ------------------------------------------------------------------
                     chunk_frames = all_frame_data[i:i + current_chunk_frame_count]
                     chunk_times = all_times[i:i + current_chunk_frame_count] # Use corresponding times
                     # derived helpers ---------------------------------------------------
                     if not chunk_times:
                         logger.warning(f"Skipping chunk {i // current_chunk_frame_count + 1} (Async) due to empty chunk_times.")
                         continue
                     chunk_idx        = (i // current_chunk_frame_count) + 1
                     chunk_start_time = chunk_times[0]
                     chunk_end_time   = chunk_times[-1]
                     # ------------------------------------------------------------------

                     print(f"    - Processing Tag Chunk {chunk_idx} ({len(chunk_frames)} frames) asynchronously...")

                     # --- Single Call for Tags (Async) ---
                     parsed_tags_for_chunk = {"persons": [], "actions": []} # Default empty
                     # --- FIX: Use specific log category --- 
                     log_cat = "tags"
                     # if task == "persons_tags": log_cat = "persons_tags"
                     # elif task == "actions_tags": log_cat = "actions_tags"
                     # --- END FIX --- 

                     for attempt in range(max_retries):
                         try:
                             # Generate combined tag prompt
                             tag_prompt = self._generate_segment_prompt(
                                 segment=segment,
                                 analysis_config=analysis_config,
                                 task="tags", # Specify tag generation task
                                 frames_subset=chunk_frames,
                                 times_subset=chunk_times,
                                 chunk_start_time=chunk_start_time,
                                 chunk_end_time=chunk_end_time
                             )
                             if tag_prompt:
                                 # --- FIX: Pass correct log_cat --- 
                                 tag_llm_response = await self._call_llm_async(tag_prompt, log_token_category=log_cat)
                                 # --- END FIX ---
                                 parsed_llm_output = self._parse_llm_json_response(
                                     tag_llm_response,
                                     expecting_chapters=False, # Not expecting chapters
                                     expecting_tags=True     # Expecting tags (both types)
                                 )
                                 # Filter the parsed output
                                 tags_to_filter = parsed_llm_output.get("globalTags", parsed_llm_output)
                                 parsed_tags_for_chunk = self._filter_unknowns(tags_to_filter)

                                 print(f"        Chunk {chunk_idx}: Received {len(parsed_tags_for_chunk.get('persons',[]))} persons, {len(parsed_tags_for_chunk.get('actions',[]))} actions (Async, after filtering).")
                                 break # Success
                             else:
                                 print(f"        Chunk {chunk_idx}: Skipping due to empty prompt (Async).")
                                 break # Exit retry
                         except Exception as e:
                             print(f"        Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} failed for tags (Async). Error: {e}")
                             if attempt + 1 == max_retries:
                                 print(f"        Chunk {chunk_idx}: Max retries reached for tags (Async).")
                             else:
                                 wait_time = initial_delay * (2 ** attempt)
                                 print(f"          Retrying chunk tags in {wait_time:.1f} seconds (Async)...")
                                 await asyncio.sleep(wait_time)
                     # --- End Retry Loop ---

                     # --- Collect tags ---
                     merged_global_tags["persons"].extend(parsed_tags_for_chunk.get("persons", []))
                     merged_global_tags["actions"].extend(parsed_tags_for_chunk.get("actions", []))

                     # --- Snap Timestamps AFTER parsing and filtering, BEFORE merging (Async) --- 
                     try:
                         valid_chunk_times_float = [t for t in chunk_times if isinstance(t, (int, float))]
                         if not valid_chunk_times_float:
                             print(f"Warning (Snap/Async): No valid float timestamps for chunk {chunk_idx}. Skipping snapping.")
                         else:
                             for tag_type in ("persons", "actions"):
                                 tags_list = parsed_tags_for_chunk.get(tag_type, [])
                                 if not isinstance(tags_list, list): continue

                                 for entry in tags_list:
                                     if isinstance(entry, dict) and isinstance(entry.get("timecodes"), list):
                                         for tc in entry["timecodes"]:
                                             if isinstance(tc, dict) and "start" in tc and "end" in tc:
                                                 try:
                                                     start_snap = self._snap_to_nearest(tc["start"], valid_chunk_times_float)
                                                     end_snap = self._snap_to_nearest(tc["end"], valid_chunk_times_float)
                                                     if end_snap < start_snap: end_snap = start_snap
                                                     tc["start"], tc["end"] = f"{start_snap:.3f}s", f"{end_snap:.3f}s"
                                                 except ValueError as snap_err:
                                                      print(f"Warning (Snap/Async): Could not snap timecode {tc} in chunk {chunk_idx}: {snap_err}")
                                                 except Exception as general_snap_err:
                                                      print(f"Warning (Snap/Async): Unexpected error snapping timecode {tc} in chunk {chunk_idx}: {general_snap_err}")
                     except Exception as outer_snap_err:
                          print(f"Error during timestamp snapping setup for chunk {chunk_idx} (Async): {outer_snap_err}")
                     # --- End Snap Timestamps (Async) ---

                     # --- Collect tags from the successfully processed (and snapped) chunk ---
                     merged_global_tags["persons"].extend(parsed_tags_for_chunk.get("persons", []))
                     merged_global_tags["actions"].extend(parsed_tags_for_chunk.get("actions", []))

                     # --- MODIFIED: Convert/Validate timecodes before merging (Async) ---
                     for tag_type in ("persons", "actions"):
                         tags_list = parsed_tags_for_chunk.get(tag_type, [])
                         if not isinstance(tags_list, list): continue

                         processed_timecodes_for_tag = [] # Temporary list for validated timecodes of this tag type
                         for entry in tags_list:
                             if isinstance(entry, dict) and "name" in entry and isinstance(entry.get("timecodes"), list):
                                 tag_name = entry["name"]
                                 validated_tcs_for_entry = [] # Timecodes for this specific entry
                                 for tc in entry["timecodes"]:
                                     if isinstance(tc, dict) and "start" in tc and "end" in tc:
                                         try:
                                             # Convert start and end using utility function, format to X.XXXs
                                             start_sec_str = f"{convert_string_to_seconds(tc['start']):.3f}s"
                                             end_sec_str = f"{convert_string_to_seconds(tc['end']):.3f}s"
                                             # Append validated timecode dict
                                             validated_tcs_for_entry.append({"start": start_sec_str, "end": end_sec_str})
                                         except (ValueError, TypeError) as conv_err:
                                             print(f"Warning (Async): Could not convert timecode '{tc}' for tag '{tag_name}': {conv_err}. Skipping.")
                                         except Exception as general_err:
                                             print(f"Warning (Async): Unexpected error processing timecode '{tc}' for tag '{tag_name}': {general_err}. Skipping.")
                                     else:
                                         print(f"Warning (Async): Skipping malformed timecode entry for tag '{tag_name}': {tc}")
                                 # Add the entry with its validated timecodes to the processed list for this tag type
                                 if validated_tcs_for_entry:
                                      processed_timecodes_for_tag.append({"name": tag_name, "timecodes": validated_tcs_for_entry})

                         # Extend the main merged_global_tags with the processed & validated tags for this type
                         merged_global_tags[tag_type].extend(processed_timecodes_for_tag)
                     # --- END MODIFICATION ---

                 print(f"  - Stage 2 (Async): Finished processing all tag chunks.")
        # --- End Main Chunk Loop ---

        # --- Stage 3: Run Scoreboard Analysis (Async) ---
        print(f"  - Stage 3 (Async): Requesting Scoreboard Analysis...")
        scoreboard_events = [] # Default to empty list
        try:
             if self.env.runtime_overrides.get("scorebug"): # Only run if ROI is defined
                scoreboard_events = await self._analyze_segment_scoreboard_async(segment)
                print(f"  - Stage 3 (Async): Received {len(scoreboard_events)} scoreboard event(s).")
             else:
                  print("  - Stage 3 (Async): Skipping scoreboard analysis - Scorebug ROI not defined.")
        except Exception as score_ex:
             print(f"  - Stage 3 (Async): Scoreboard analysis failed: {score_ex}")
        # --- END Stage 3 (Async) ---

        # --- Stage 4: Combine (Async - Chapters + Tags + Scoreboard) ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []),
            "globalTags": merged_global_tags,
            "scoreboard_events": scoreboard_events # ADDED scoreboard results here
        }


        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis (async) in {elapsed_segment:.2f} seconds.")

        return final_combined_result

    # ------------------------------------------------------------------
# ALWAYS return a dict; never propagate None.
# ------------------------------------------------------------------
    def _parse_llm_json_response(
        self,
        response,
        expecting_chapters: bool = True,
        expecting_tags: bool = True
    ) -> dict:
        """
        Pull the first JSON object out of an LLM response.
        Never returns None – falls back to an empty structure that
        still satisfies downstream .get() calls.
        """
        safe_empty = {}
        if expecting_chapters:
            safe_empty["chapters"] = []
        if expecting_tags:
            # Adjusted to match the combined structure {"persons": ..., "actions": ...}
            # This parsing assumes the LLM returns the keys directly,
            # or they are nested under globalTags. The filter step handles the nesting check.
            safe_empty["persons"] = []
            safe_empty["actions"] = []
            safe_empty["globalTags"] = {"persons": [], "actions": []} # Include nested default too

        # ── guard rails ────────────────────────────────────────────────
        if not response or not getattr(response, "choices", None):
            return safe_empty

        # Access message content safely
        try:
            message = response.choices[0].message
            raw = message.content if message else ""
            if not raw: raw = "" # Ensure raw is a string
        except (IndexError, AttributeError):
            raw = ""

        # Strip triple back-tick blocks if present
        raw = re.sub(r"```(?:json)?", "", raw).strip("` \\n")
        # Find first brace → last brace (fallback for models that wrap text)
        start_brace = raw.find("{")
        end_brace = raw.rfind("}")
        if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
            raw = raw[start_brace : end_brace + 1]
        else:
            # If no JSON structure found, return the safe empty structure
            return safe_empty

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Make sure expected top-level keys exist based on flags
                if expecting_chapters and "chapters" not in parsed:
                    parsed["chapters"] = safe_empty["chapters"]
                if expecting_tags:
                     if "globalTags" not in parsed: # Check for the parent key if expecting tags
                        # If globalTags is missing, check if persons/actions are top-level
                        if "persons" not in parsed: parsed["persons"] = []
                        if "actions" not in parsed: parsed["actions"] = []
                        # Also add the default globalTags structure for consistency downstream
                        parsed["globalTags"] = {"persons": parsed["persons"], "actions": parsed["actions"]}
                     else: # Ensure nested keys exist if globalTags is present
                         if not isinstance(parsed["globalTags"], dict): # Handle case where globalTags is not a dict
                              print(f"Warning: Expected 'globalTags' to be a dict, got {type(parsed['globalTags'])}. Resetting.")
                              parsed["globalTags"] = {"persons": [], "actions": []}
                         else:
                             if "persons" not in parsed["globalTags"]: parsed["globalTags"]["persons"] = []
                             if "actions" not in parsed["globalTags"]: parsed["globalTags"]["actions"] = []
                             # Ensure top-level also reflects nested if they existed at top before
                             if "persons" not in parsed: parsed["persons"] = parsed["globalTags"]["persons"]
                             if "actions" not in parsed: parsed["actions"] = parsed["globalTags"]["actions"]

                return parsed
        except json.JSONDecodeError as e:
            # Log the parsing error and the raw content for debugging
            logger.warning(f"Failed to parse LLM JSON response: {e}. Raw content: '{raw[:500]}...'") # Log first 500 chars
            pass # Fall through to return safe_empty
        except Exception as e: # Catch other potential errors
            logger.warning(f"Unexpected error parsing LLM JSON response: {e}. Raw content: '{raw[:500]}...'")
            pass

        return safe_empty

    def _call_llm(self, messages, log_token_category=None):
        # Initialize token counts before try block
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        image_tokens = 0
        total_with_images = 0
        prompt_type = log_token_category or 'Unknown'

        try:
            # --- Client Setup ---
            # (This part remains the same - ensure client setup is correct)
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("AzureOpenAI client not found...") # Shortened error
            client = AzureOpenAI(
                api_key=self.env.vision.api_key.get_secret_value(),
                api_version=self.env.vision.api_version,
                azure_endpoint=self.env.vision.endpoint,
            )
            completion_params = {}
            max_token_value = 5000 # Adjusted max completion tokens
            # --- Use the correct parameter name based on model type ---
            # Check your specific model documentation if issues persist
            completion_params["max_tokens"] = max_token_value
            # --- ADD TEMPERATURE ---
            completion_params["temperature"] = 0.0
            # --- END ADD TEMPERATURE ---
            logger.debug(f"Using 'max_tokens={max_token_value}', 'temperature=0.0' for model '{self.env.vision.deployment}'")
            # --- End Client Setup ---

            # ---- SIZE GUARD ----
            # (Remains the same)
            raw_body_len = len(json.dumps({"model": self.env.vision.deployment, "messages": messages}))
            if raw_body_len > 1_000_000: # Example limit, adjust if needed
                raise RuntimeError(f"Request body {raw_body_len:,} B exceeds limit.")
            # ---- END SIZE GUARD ----

            # --- API Call ---
            response = client.chat.completions.create(
                model=self.env.vision.deployment,
                messages=messages,
                **completion_params
            )
            # --- End API Call ---

            # --- Process Response and Tokens (AFTER successful call) ---
            if hasattr(response, 'usage') and response.usage is not None:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
                if total_tokens == 0: total_tokens = prompt_tokens + completion_tokens

            # Estimate image tokens based on the input messages
            image_tokens = self._estimate_image_tokens(messages)
            total_with_images = total_tokens + image_tokens

            # Log the calculated tokens
            logger.info(f"[TOKENS] Category='{log_token_category}': prompt={prompt_tokens}, completion={completion_tokens}, estimated_images={image_tokens}, total_api={total_tokens}, estimated_total={total_with_images}")

            # Update overall token usage
            if log_token_category:
                # Ensure category exists (initialize if first time)
                if log_token_category not in self.token_usage:
                    self.token_usage[log_token_category] = 0
                if "total" not in self.token_usage:
                    self.token_usage["total"] = 0

                self.token_usage[log_token_category] += total_with_images
                self.token_usage["total"] += total_with_images

            return response
            # --- End Process Response ---

        except Exception as api_err:
             logger.error(f"Error calling OpenAI API ({prompt_type}): {api_err}", exc_info=True)
             raise

    async def _call_llm_async(self, messages, log_token_category=None):
        # Initialize token counts before try block
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        image_tokens = 0
        total_with_images = 0
        prompt_type = log_token_category or 'Unknown'

        try:
            # --- Async Client Setup ---
            # (This part remains the same)
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                 raise ImportError("AsyncAzureOpenAI client not found...") # Shortened error
            client = AsyncAzureOpenAI(
                api_key=self.env.vision.api_key.get_secret_value(),
                api_version=self.env.vision.api_version,
                azure_endpoint=self.env.vision.endpoint,
            )
            completion_params = {}
            max_token_value = 5000 # Adjusted max completion tokens
            completion_params["max_tokens"] = max_token_value
            # --- ADD TEMPERATURE ---
            completion_params["temperature"] = 0.0
            # --- END ADD TEMPERATURE ---
            logger.debug(f"Using 'max_tokens={max_token_value}', 'temperature=0.0' for model '{self.env.vision.deployment}' (async)")
            # --- End Async Client Setup ---

            # ---- SIZE GUARD ----
            # (Remains the same)
            raw_body_len = len(json.dumps({"model": self.env.vision.deployment, "messages": messages}))
            if raw_body_len > 1_000_000: # Example limit
                raise RuntimeError(f"Request body {raw_body_len:,} B exceeds limit.")
            # ---- END SIZE GUARD ----

            # --- API Call ---
            response = await client.chat.completions.create(
                model=self.env.vision.deployment,
                messages=messages,
                **completion_params
            )
            # --- End API Call ---

            # --- Process Response and Tokens (AFTER successful call) ---
            if hasattr(response, 'usage') and response.usage is not None:
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                total_tokens = getattr(response.usage, 'total_tokens', 0)
                if total_tokens == 0: total_tokens = prompt_tokens + completion_tokens

            # Estimate image tokens based on the input messages
            image_tokens = self._estimate_image_tokens(messages)
            total_with_images = total_tokens + image_tokens

            # Log the calculated tokens
            logger.info(f"[TOKENS] Category='{log_token_category}': prompt={prompt_tokens}, completion={completion_tokens}, estimated_images={image_tokens}, total_api={total_tokens}, estimated_total={total_with_images}")

            # Update overall token usage
            if log_token_category:
                 # Ensure category exists (initialize if first time)
                if log_token_category not in self.token_usage:
                    self.token_usage[log_token_category] = 0
                if "total" not in self.token_usage:
                    self.token_usage["total"] = 0

                self.token_usage[log_token_category] += total_with_images
                self.token_usage["total"] += total_with_images

            return response
            # --- End Process Response ---

        except Exception as api_err:
             logger.error(f"Error calling OpenAI API asynchronously ({prompt_type}): {api_err}", exc_info=True)
             raise

    # --- ADDED: Cleanup Helper ---
    def _cleanup_temp_blobs(self):
        """Deletes temporary frame blobs from Azure Storage."""
        print(f"Starting cleanup of temporary frame blobs...")
        start_cleanup_time = time.time()
        deleted_count = 0
        skipped_count = 0
        blob_service_client = None

        try:
            # Reuse connection logic from upload_blob (simplified)
            if self.env.blob_storage.connection_string:
                connect_str = self.env.blob_storage.connection_string.get_secret_value()
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            elif self.env.blob_storage.sas_token:
                account_url = f"https://{self.env.blob_storage.account_name}.blob.core.windows.net"
                sas_token_str = self.env.blob_storage.sas_token.get_secret_value()
                blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token_str)
            
            if not blob_service_client:
                 print("Warning: Blob Storage client could not be initialized for cleanup.")
                 return
            
            print(f"Connected to container: {self.env.blob_storage.container_name}")
            container_client = blob_service_client.get_container_client(self.env.blob_storage.container_name)
            
            # Collect all blob names to delete
            all_blob_names = []
            for segment in self.manifest.segments:
                if hasattr(segment, '_blob_names') and segment._blob_names:
                    all_blob_names.extend(segment._blob_names)

            if not all_blob_names:
                 print("No temporary frame blobs found to clean up.")
                 return

            print(f"Attempting to delete {len(all_blob_names)} blobs...")

            # Use ThreadPoolExecutor for concurrent deletion (optional, for many blobs)
            # Adjust max_workers as needed
            # from concurrent.futures import ThreadPoolExecutor
            # with ThreadPoolExecutor(max_workers=16) as pool:
            #     results = list(pool.map(container_client.delete_blob, all_blob_names))
            # deleted_count = len(results) # Assumes pool.map raises exceptions on failure

            # Simple sequential deletion (safer for logging/error handling)
            for blob_name in all_blob_names:
                 try:
                      # print(f"  Deleting blob: {blob_name}") # Remove per-blob logging
                      container_client.delete_blob(blob_name)
                      deleted_count += 1
                 except Exception as e:
                      print(f"Warning: Failed to delete blob {blob_name}: {e}")
                      skipped_count += 1

        except Exception as e:
            print(f"ERROR during blob cleanup initialization or connection: {e}")
            # Don't try to count if connection failed
            skipped_count = len(all_blob_names) - deleted_count 
        finally:
            cleanup_duration = time.time() - start_cleanup_time
            print(f"Blob cleanup finished in {cleanup_duration:.2f}s. Deleted: {deleted_count}, Failed/Skipped: {skipped_count}")
            # Explicitly close client if needed (depends on SDK version/usage)
            # if blob_service_client: blob_service_client.close()

    # ... (rest of _analyze_segment) ...

    def _get_llm_client_async(self) -> AsyncAzureOpenAI:
        """Gets an asynchronous Azure OpenAI client instance."""
        try:
            client = AsyncAzureOpenAI(
                api_key=self.env.vision.api_key.get_secret_value(),
                api_version=self.env.vision.api_version,
                azure_endpoint=self.env.vision.endpoint,
                # Add other necessary client params like timeouts if needed
            )
            return client
        except ImportError:
             raise ImportError("AsyncAzureOpenAI client not found. Please install the 'openai' package.")
        except Exception as e:
             logger.error(f"Failed to create asynchronous AzureOpenAI client: {e}")
             raise # Re-raise the exception

    # --- ADDED: Method to attach transcripts to segments --- 
    def _attach_transcripts_to_segments(self) -> None:
        """
        Parses the full audio transcription (if available) and assigns 
        relevant text to each segment based on time overlap.
        """
        if not self.manifest.audio_transcription or not isinstance(self.manifest.audio_transcription, dict):
            print("No valid audio transcription found in manifest. Skipping attachment.")
            for segment in self.manifest.segments:
                segment.transcription = "No transcription available for this segment."
            return

        print("Attaching transcription segments to video segments...")
        # Expected structure from Batch API v3.2
        recognized_phrases = self.manifest.audio_transcription.get("recognizedPhrases", [])
        if not recognized_phrases or not isinstance(recognized_phrases, list):
             print("Transcription object found, but 'recognizedPhrases' key is missing or empty.")
             for segment in self.manifest.segments:
                  segment.transcription = "Transcription available but format unexpected."
             return

        segment_index = {idx: seg for idx, seg in enumerate(self.manifest.segments)}
        segment_transcriptions = {idx: [] for idx in segment_index}

        for phrase in recognized_phrases:
            if not isinstance(phrase, dict): continue
            try:
                # Ticks to seconds conversion
                offset_ticks = phrase.get("offsetInTicks", 0)
                duration_ticks = phrase.get("durationInTicks", 0)
                if offset_ticks is None or duration_ticks is None: continue # Skip if timing info missing

                phrase_start = offset_ticks / 10_000_000.0
                phrase_duration = duration_ticks / 10_000_000.0
                phrase_end = phrase_start + phrase_duration

                # Extract the display text from the best hypothesis
                best_recognition = phrase.get("nBest", [{}])[0]
                display_text = best_recognition.get("display", None)
                if not display_text: continue # Skip if no text

                # Find overlapping segments
                for idx, segment in segment_index.items():
                    if (phrase_start < segment.end_time) and (phrase_end > segment.start_time):
                        # --- FIX: Append dict instead of tuple ---
                        segment_transcriptions[idx].append({
                            "start": round(phrase_start, 3), # Keep precision
                            "text": display_text
                        })
                        # --- END FIX ---

            except (TypeError, IndexError, KeyError, ValueError) as e:
                print(f"Warning: Error processing transcript phrase during attachment: {e}. Data: {phrase}")
                continue

        # Assign combined text to each segment
        for idx, segment in segment_index.items():
            # Sort phrases by their start time before joining
            # --- FIX: Use the dict format for sorting --- 
            sorted_phrases = sorted(segment_transcriptions[idx], key=lambda item: item["start"])
            segment.transcription = " ".join([item["text"] for item in sorted_phrases]) if sorted_phrases else "No transcription for this segment."
            # --- FIX: Store the list of dicts --- 
            segment.transcription_segments = sorted_phrases 
            # --- END FIX ---
        
        print("Finished attaching transcriptions.")

    # --- ADDED: Restored _generate_segment_prompt Method Definition ---
    # (Copied from previous read, ensure imports like Formatter are present)
    from string import Formatter # Make sure Formatter is imported
    from .models.environment import CobraEnvironment # Ensure env model is available
    from .models.video import Segment # Ensure Segment model is available
    from .analysis.base_analysis_config import AnalysisConfig # Ensure base config is available
    # --- Corrected Relative Import ---
    from .cobra_utils import encode_image_base64 # Ensure helper is available
    # --- End Correction ---

    def _generate_segment_prompt(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
        # --- Parameters for different modes ---
        task: Literal["chapters", "tags"], # Use task Literal
        # generate_chapters: bool = False, # Removed
        # tag_type: Optional[str] = None, # Removed
        # --- Frame/Time subset parameters (remain the same) ---
        frames_subset: List[str] = None,
        times_subset: List[float] = None,
        chunk_start_time: Optional[float] = None,
        chunk_end_time: Optional[float] = None,
    ):
        """
        Generates a prompt for either:
        1. Chapter generation (if generate_chapters=True)
        2. Specific tag type generation (if tag_type is provided)
        Uses frame URLs if available.
        """
        # --- REMOVED check for mutually exclusive modes ---
        # if generate_chapters and tag_type:
        #     raise ValueError("Cannot generate chapters and tags in the same prompt call.")
        # if not generate_chapters and not tag_type:
        #      print("Warning: _generate_segment_prompt called without specifying chapters or a tag_type.")
        #      return None
        # --- END REMOVAL ---

        # --- Frame selection logic ---
        if segment.frame_urls and frames_subset is None:
             frames_to_process = segment.frame_urls
             times_to_process = segment.segment_frame_time_intervals if times_subset is None else times_subset
        elif frames_subset is not None:
             frames_to_process = frames_subset
             times_to_process = times_subset
        else:
             frames_to_process = segment.segment_frames_file_path
             times_to_process = segment.segment_frame_time_intervals if times_subset is None else times_subset
             print(f"Warning: Frame URLs not found for {segment.segment_name}. Falling back to local paths.")

        if not frames_to_process:
             print(f"Warning: No frames to process for prompt generation in segment {segment.segment_name}.")
             return None

        if not times_to_process or len(times_to_process) != len(frames_to_process):
             print(f"Warning: Mismatch or missing times for frames in segment {segment.segment_name}. Times: {len(times_to_process)}, Frames: {len(frames_to_process)}. Recalculating...")
             effective_start = chunk_start_time if chunk_start_time is not None else segment.start_time
             effective_end = chunk_end_time if chunk_end_time is not None else segment.end_time
             if len(frames_to_process) > 0:
                  # Use linspace and ensure timestamps are floats
                  times_to_process = np.linspace(effective_start or 0, effective_end or (effective_start or 0), len(frames_to_process), endpoint=True)
                  times_to_process = [round(float(t), 3) for t in times_to_process]
             else: times_to_process = []
        # Ensure all timestamps are floats
        times_to_process = [float(t) for t in times_to_process]
        # --- End Frame Selection ---

        messages = []
        transcription_context = segment.transcription if segment.transcription else "No transcription available"

        # --- CHAPTER GENERATION PATH ---
        if task == "chapters":
            system_prompt_template_ch = getattr(analysis_config, 'system_prompt_chapters', "Default chapter prompt.")
            shot_types_str = "Unknown"
            if hasattr(analysis_config, 'SHOT_TYPES') and isinstance(analysis_config.SHOT_TYPES, list):
                shot_types_str = ", ".join([f'\"{shot}\"' for shot in analysis_config.SHOT_TYPES])
            else: print("Warning: Could not find SHOT_TYPES list in AnalysisConfig.")
            person_labels_str = self._get_list_labels("persons")
            action_labels_str = self._get_list_labels("actions")
            format_data_ch = {
                "start_time": f"{{{(segment.start_time or 0):.3f}}}",
                "end_time": f"{{{(segment.end_time or 0):.3f}}}",
                "shot_types_list": shot_types_str,
                "person_labels_list": person_labels_str,
                "action_labels_list": action_labels_str,
            }
            # --- MODIFIED: Print actual exception on format error ---
            try: 
                system_prompt_ch = self._safe_format(system_prompt_template_ch, **format_data_ch)
            except Exception as e: 
                print(f"ERROR: Failed to format chapter system prompt: {e}. Using template as is.") # Print the actual exception
                system_prompt_ch = system_prompt_template_ch # Fallback to unformatted
            # --- END MODIFICATION ---
            messages.append({"role": "system", "content": system_prompt_ch})
            user_content = []
            user_content.append({"type": "text", "text": "Provide analysis as a sports commentator."})
            if transcription_context and transcription_context != "No transcription available":
                user_content.append({"type": "text", "text": f"Audio transcription ({{segment.segment_duration or 0:.3f}}s): {{transcription_context}}"})
            else: user_content.append({"type": "text", "text": "No transcription provided."})
            user_content.append({"type": "text", "text": "\nFrames:"})
            for idx, (frame_url_or_path, ts) in enumerate(zip(frames_to_process, times_to_process), 1):
                user_content.append({"type": "text", "text": f"\nImage #{{idx}} at {{ts:.3f}}s"})
                if frame_url_or_path.startswith("http"):
                     user_content.append({"type": "image_url", "image_url": {"url": frame_url_or_path, "detail": "low"}})
                else:
                    try:
                         b64 = encode_image_base64(frame_url_or_path);
                         if b64: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{{b64}}", "detail": "low"}})
                    except Exception as e: print(f"Warning: Error encoding frame path {{frame_url_or_path}} for chapter prompt: {{e}}")
            messages.append({"role": "user", "content": user_content})

        # --- TAG GENERATION PATH ---
        elif task == "tags":
            system_prompt_template = getattr(analysis_config, 'system_prompt_tags', None) # Get combined tags prompt
            # definitions_str = "No definitions available." # Definitions are now in LABEL_BLOCK
            # instructions_str = "No instructions available." # Instructions are in system prompt

            if not system_prompt_template: print(f"ERROR: Could not find 'system_prompt_tags' in AnalysisConfig."); return None
            
            # --- REMOVED: Unnecessary formatting attempt for static tag system prompt ---
            # System prompt formatting (if needed - current example is static)
            # --- MODIFIED: Use _safe_format cautiously, only if template has placeholders ---
            # format_data = {} # Add any needed dynamic fields here (e.g., game context if it varies)
            # try:
            #     # Check if the template actually contains format specifiers before formatting
            #     if any(list(Formatter().parse(system_prompt_template))[i][1] is not None for i in range(len(list(Formatter().parse(system_prompt_template))))):
            #          system_prompt_formatted = self._safe_format(system_prompt_template, **format_data)
            #     else: # No placeholders found, use as is
            #          system_prompt_formatted = system_prompt_template
            # except Exception as e:
            #     print(f"ERROR: Failed to format tag system prompt: {e}. Using template as is.")
            #     system_prompt_formatted = system_prompt_template # Fallback
            # # --- END MODIFICATION ---
            # messages.append({"role": "system", "content": system_prompt_formatted})
            # --- END REMOVAL ---

            # --- Use the static system prompt template directly --- 
            messages.append({"role": "system", "content": system_prompt_template})
            # --- End --- 

            # --- Construct User Message using Lean Format ---
            # Ensure times_subset are formatted correctly for JSON
            frame_timestamps_list = [f"{ts:.3f}s" for ts in times_to_process]
            frame_data_block = {
                # Use frame_urls if they were the source, else paths
                "frame_source": frames_to_process,
                "frame_timestamps": frame_timestamps_list
            }
            # Compact JSON separators
            frame_data_json = json.dumps(frame_data_block, separators=(",",":"))

            # Assemble user content string
            user_content_text = f"Allowed labels:\n{self.LABEL_BLOCK}\n\nFrame data:\n{frame_data_json}"
            if transcription_context and transcription_context != "No transcription available":
                 user_content_text += f"\n\nTranscription Context:\n{transcription_context}"

            # Add text part
            user_content = [{"type": "text", "text": user_content_text}]

            # --- Add Images (No change needed here) ---
            for i, (frame_url_or_path, timestamp) in enumerate(zip(frames_to_process, times_to_process)):
                 frame_id = str(i + 1)
                 ts_str = f"{timestamp:.3f}s"
                 # Note: We don't add per-frame text prompts like "Image #X at Y.YYYs" in the lean format
                 # The frame_timestamps list in the JSON serves this purpose.
                 if frame_url_or_path.startswith("http"):
                      user_content.append({"type": "image_url", "image_url": {"url": frame_url_or_path, "detail": "low"}})
                 else:
                     try:
                          b64 = encode_image_base64(frame_url_or_path)
                          if b64: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
                     except Exception as e: print(f"Warning: Error encoding frame path {frame_url_or_path} for tag prompt: {e}")
            # --- End Add Images ---

            messages.append({"role": "user", "content": user_content})
            # --- End Construct User Message ---

        else:
            print(f"Error: Invalid task '{task}' specified for _generate_segment_prompt.")
            return None
        return messages

    def _filter_unknowns(
        self,
        tag_block: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Safety Filter: Ensure ONLY tags with names EXACTLY matching
        the canonical lists (persons, actions) remain.
        Operates on the input dictionary, potentially modifying nested lists.
        """
        if not isinstance(tag_block, dict):
            print(f"Warning: _filter_unknowns expected dict, got {type(tag_block)}. Returning original block.")
            return tag_block

        # Handle potential nesting under globalTags
        actual_tags_to_filter = tag_block
        if "globalTags" in tag_block and isinstance(tag_block["globalTags"], dict):
            actual_tags_to_filter = tag_block["globalTags"]

        allowed_persons = self._canonical_persons
        allowed_actions = self._canonical_actions
        # --- REMOVED: Explicitly allow scoreboard tags ---
        # allowed_scoreboard_tags = {\"madeFreeThrow\", \"2-pointer\", \"3-pointer\"}
        # --- END REMOVED ---

        for tag_type, allowed_set in [("persons", allowed_persons), ("actions", allowed_actions)]:
            original_tags = actual_tags_to_filter.get(tag_type, [])
            if not isinstance(original_tags, list):
                print(f"Warning: Expected '{tag_type}' key to contain a list, found {type(original_tags)}. Skipping filtering for this type.")
                continue # Skip to next tag type if format is wrong

            kept_tags = []
            dropped_count = 0
            for entry in original_tags:
                if isinstance(entry, dict):
                    label = entry.get("name", "").strip()
                    # --- MODIFIED: ONLY Check allowed set ---
                    is_allowed = label in allowed_set
                    # --- END MODIFIED ---

                    if is_allowed:
                        kept_tags.append(entry)
                    else:
                        if label: # Only log if there was a label to drop
                            print(f"DEBUG (Filter): Dropping tag '{label}' (Not in allowed set for {tag_type})")
                            # print(f"Info (Filter): Dropping unknown/non-canonical {tag_type} tag '{label}'") # Original info log
                            dropped_count += 1
                else:
                    print(f"Warning (Filter): Skipping non-dict item in {tag_type} list: {entry}")

            # Replace the original list with the filtered list
            actual_tags_to_filter[tag_type] = kept_tags

            # --- FIX: Ensure top-level dict mirrors nested if needed --- 
            if tag_block is not actual_tags_to_filter and tag_type in tag_block:
                tag_block[tag_type] = actual_tags_to_filter[tag_type]
            # --- END FIX --- 

        # Return the original input block (which might have been modified)
        return tag_block

    def _get_list_definitions(self, list_type: str) -> str:
        """Gets definitions from the corresponding *_for_prompt list."""
        definitions = []
        list_for_prompt = []
        instructions = ""

        if list_type == "persons":
            list_for_prompt = getattr(self, '_persons_for_prompt', []) # Use getattr
            instructions = getattr(self, 'persons_instructions', "Identify people based on the following:") # Use getattr
        elif list_type == "actions":
            list_for_prompt = getattr(self, '_actions_for_prompt', []) # Use getattr
            instructions = getattr(self, 'actions_instructions', "Identify actions based on the following:") # Use getattr
        # Add cases for emotions, themes if their structure/prompting needs change

        if not list_for_prompt:
            return f"No predefined {list_type} provided."

        # Prepend instructions
        definitions.append(f"{instructions}\n")

        for item in list_for_prompt:
            if isinstance(item, dict) and "label" in item and "description" in item:
                definitions.append(f"- \"{item['label']}\": {item['description']}")

        return "\n".join(definitions) if definitions else f"No valid {list_type} definitions found."

    def _get_list_labels(self, list_type: str) -> str: # Changed return type to string
        """Gets labels from the corresponding *_for_prompt list as a comma-separated string."""
        labels = []
        list_for_prompt = []

        if list_type == "persons":
            list_for_prompt = getattr(self, '_persons_for_prompt', []) # Use getattr
        elif list_type == "actions":
            list_for_prompt = getattr(self, '_actions_for_prompt', []) # Use getattr
        # Add cases for emotions, themes if needed

        for item in list_for_prompt:
            if isinstance(item, dict) and "label" in item:
                # Simpler way to add quotes
                labels.append('"' + item["label"] + '"') 

        return ", ".join(labels) if labels else "No labels defined"

    # --- ADDED: Restored _estimate_image_tokens Method Definition ---
    def _estimate_image_tokens(self, messages: List[Dict]) -> int:
        """Estimates image tokens based on image_url entries in messages."""
        image_token_estimate = 0
        # Constants based on OpenAI documentation (as of early 2024)
        # https://openai.com/pricing - Vision Pricing
        LOW_DETAIL_COST = 85
        HIGH_DETAIL_BASE_COST = 85
        HIGH_DETAIL_TILE_COST = 170
        TILE_SIZE = 512
        HIGH_DETAIL_AVG_ESTIMATE = 765 # Placeholder average

        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        detail = item.get("image_url", {}).get("detail", "auto")
                        if detail == "low":
                            image_token_estimate += LOW_DETAIL_COST
                        else: # Treat 'high' or 'auto' as high for estimation
                            image_token_estimate += HIGH_DETAIL_AVG_ESTIMATE
        return image_token_estimate
    # --- END ADDED ---

    # --- ADDED: Restored _generate_segment_prompt Method Definition ---
    # (Copied from previous read, ensure imports like Formatter are present)
    from string import Formatter # Make sure Formatter is imported
    from .models.environment import CobraEnvironment # Ensure env model is available
    from .models.video import Segment # Ensure Segment model is available
    from .analysis.base_analysis_config import AnalysisConfig # Ensure base config is available
    # --- Corrected Relative Import ---
    from .cobra_utils import encode_image_base64 # Ensure helper is available
    # --- End Correction ---

  
    # --- ADDED: Restored _safe_format Method Definition ---
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
        # Ensure Formatter is imported
        from string import Formatter 
        return Formatter().vformat(tmpl, (), _KeepMissing(kwargs))
    # --- END Restored Method ---


    async def _unique_frames_async(
        self,
        frame_paths_or_urls: List[str],
        timestamps: List[float],
        threshold: float = 0.9,
        min_time_diff: float = 0.5
    ) -> Tuple[List[str], List[float]]:
        """
        Asynchronously filters frames based on SSIM similarity and time difference.
        Keeps the first frame, then subsequent frames if their SSIM to the *last kept frame*
        is below the threshold OR if the time difference is >= min_time_diff.
        Handles both local paths and URLs.
        """
        if not frame_paths_or_urls or len(frame_paths_or_urls) != len(timestamps):
            print("Warning (_unique_frames_async): Invalid input frames or timestamps. Returning original.")
            return frame_paths_or_urls, timestamps
        if len(frame_paths_or_urls) <= 1:
            return frame_paths_or_urls, timestamps # No deduplication needed for 0 or 1 frame

        print(f"Deduplicating {len(frame_paths_or_urls)} frames (async)... Threshold={threshold}, MinTimeDiff={min_time_diff}s")
        start_dedup_time = time.time()

        kept_frames = [frame_paths_or_urls[0]]
        kept_timestamps = [timestamps[0]]
        last_kept_ts = timestamps[0]
        last_kept_frame_data = None # Will store the image data (numpy array) of the last kept frame

        # Function to read/download and decode image (async)
        async def get_image_data(path_or_url):
            try:
                if path_or_url.startswith("http"):
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(path_or_url)
                        response.raise_for_status()
                        img_array = np.frombuffer(response.content, np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        return frame
                else:
                    if not await asyncio.to_thread(os.path.exists, path_or_url): return None
                    loop = asyncio.get_running_loop()
                    frame = await loop.run_in_executor(None, cv2.imread, path_or_url)
                    return frame
            except Exception as e:
                print(f"Warning (_unique_frames_async): Failed to load/decode image {path_or_url}: {e}")
                return None

        # Process the first frame
        last_kept_frame_data = await get_image_data(kept_frames[0])
        if last_kept_frame_data is None:
             print("Warning (_unique_frames_async): Could not process the first frame. Deduplication might be ineffective.")
             # Return original list if first frame fails? Or continue cautiously? Let's continue.
             # If first frame fails, comparison base is lost. Maybe keep all?
             # For now, let's assume subsequent frames might load and comparisons can happen later if one is kept.

        # Iterate through the rest
        for i in range(1, len(frame_paths_or_urls)):
            current_path_or_url = frame_paths_or_urls[i]
            current_ts = timestamps[i]

            # Check time difference first (cheaper)
            if current_ts - last_kept_ts >= min_time_diff:
                # print(f"DEBUG: Keeping frame {i} due to time diff >= {min_time_diff}s")
                current_frame_data = await get_image_data(current_path_or_url)
                if current_frame_data is not None:
                    kept_frames.append(current_path_or_url)
                    kept_timestamps.append(current_ts)
                    last_kept_frame_data = current_frame_data # Update comparison base
                    last_kept_ts = current_ts
                continue # Move to next frame

            # If time diff is small, check similarity (expensive)
            if last_kept_frame_data is None:
                 # If we don't have a previous frame to compare to, try loading current
                 # and if successful, keep it as the new baseline.
                 current_frame_data = await get_image_data(current_path_or_url)
                 if current_frame_data is not None:
                      kept_frames.append(current_path_or_url)
                      kept_timestamps.append(current_ts)
                      last_kept_frame_data = current_frame_data
                      last_kept_ts = current_ts
                 continue

            current_frame_data = await get_image_data(current_path_or_url)
            if current_frame_data is None:
                continue # Skip if current frame fails to load

            # Ensure frames have same dimensions for SSIM (resize if necessary, though costly)
            # Basic check: if shapes don't match, maybe keep it? Or resize smaller one?
            # For simplicity, let's skip comparison if shapes mismatch significantly.
            # A better approach would be robust resizing.
            if last_kept_frame_data.shape != current_frame_data.shape:
                 print(f"Warning (_unique_frames_async): Frame shapes mismatch ({last_kept_frame_data.shape} vs {current_frame_data.shape}). Keeping frame {i}.")
                 kept_frames.append(current_path_or_url)
                 kept_timestamps.append(current_ts)
                 last_kept_frame_data = current_frame_data # Update comparison base
                 last_kept_ts = current_ts
                 continue

            # Calculate SSIM (run in executor as it can be CPU intensive)
            try:
                 loop = asyncio.get_running_loop()
                 # Ensure grayscale conversion happens inside the executor function
                 def calculate_ssim_sync(img1, img2):
                      gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                      gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                      # Use data_range appropriate for uint8 images
                      return ssim(gray1, gray2, data_range=gray2.max() - gray2.min())

                 similarity = await loop.run_in_executor(None, calculate_ssim_sync, last_kept_frame_data, current_frame_data)

                 # print(f"DEBUG: Frame {i} vs last kept: SSIM = {similarity:.4f}")
                 if similarity < threshold:
                    # print(f"DEBUG: Keeping frame {i} due to SSIM < {threshold}")
                    kept_frames.append(current_path_or_url)
                    kept_timestamps.append(current_ts)
                    last_kept_frame_data = current_frame_data # Update comparison base
                    last_kept_ts = current_ts
            except Exception as ssim_err:
                 print(f"Warning (_unique_frames_async): SSIM calculation failed for frame {i}: {ssim_err}. Keeping frame.")
                 kept_frames.append(current_path_or_url)
                 kept_timestamps.append(current_ts)
                 last_kept_frame_data = current_frame_data # Update comparison base
                 last_kept_ts = current_ts

        dedup_duration = time.time() - start_dedup_time
        print(f"Frame deduplication finished in {dedup_duration:.2f}s. Kept {len(kept_frames)} / {len(frame_paths_or_urls)} frames.")
        return kept_frames, kept_timestamps
