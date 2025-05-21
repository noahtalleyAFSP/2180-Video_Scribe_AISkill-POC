import os
import json
import time
import asyncio
import nest_asyncio
from typing import Union, Type, Optional, List, Dict, Set
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
from .cobra_utils import seconds_to_iso8601_duration # Adjust import path if needed
import csv # Added for TSV writing
from urllib.parse import urlparse, urlunparse # Added for URL manipulation
from PIL import Image # ADDED: For image dimension reading
from math import ceil # ADDED: For tiles_needed helper

# --- ADDED: Single-tariff constants for GPT-4o/Vision ---
GPT4O_BASE      = 85      # tokens per image – low‑detail     (also the "base" for high‑detail)
GPT4O_PER_TILE  = 170     # tokens per 512 × 512 tile – high‑detail

# --- DELETED: Old IMAGE_TOKEN_COST ---

# --- ADDED: Helper for tile calculation ---
def tiles_needed(w: int, h: int, tile: int = 512) -> int:
    return ceil(w / tile) * ceil(h / tile)

# --- UPDATED: estimate_image_tokens function ---
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
)

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    manifest: VideoManifest
    env: CobraEnvironment
    reprocess_segments: bool
    person_group_id: str
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

    # take either a video manifest object or a path to a video manifest file
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
        self.manifest = validate_video_manifest(video_manifest)
        self.env = env
        # --- ADDED: Store video_blob_url ---
        self.video_blob_url = video_blob_url
        # --- END ADDED ---
        self.person_group_id = person_group_id
        self.identified_people_in_segment = {}

        # Load peoples list if provided
        self.peoples_list = self._load_json_list(peoples_list_path, "persons") # Corrected key to persons

        # Load emotions list if provided
        self.emotions_list = self._load_json_list(emotions_list_path, "emotions")

        # Load objects list if provided
        self.objects_list = self._load_json_list(objects_list_path, "objects")

        # Load themes list if provided
        self.themes_list = self._load_json_list(themes_list_path, "themes")

        # Load actions list if provided
        self.actions_list = self._load_json_list(actions_list_path, "actions")

        # Initialize known tag sets
        self._current_known_persons = set()
        self._current_known_actions = set()
        self._current_known_objects = set()

        # Optionally pre-populate from lists if provided
        # --- FIX: Use correct key from loaded list data ---
        if self.peoples_list:
             people_items = self.peoples_list.get("persons", []) # Use 'persons' key
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

        # --- UPDATED: Initialize detailed token usage tracking ---
        self.token_usage = {
            # Detailed breakdown (optional internal use)
            "chapters_prompt_tokens": 0, "chapters_completion_tokens": 0, "chapters_image_tokens": 0,
            "tags_prompt_tokens": 0, "tags_completion_tokens": 0, "tags_image_tokens": 0,
            "summary_prompt_tokens": 0, "summary_completion_tokens": 0,

            # Totals needed for the report
            "report_input_text_tokens": 0,    # Sum of all prompt_tokens
            "report_output_text_tokens": 0,   # Sum of all completion_tokens
            "report_input_image_tokens": 0,   # Sum of all image_tokens
            "report_total_tokens": 0          # Sum of the three above
        }
        # --- END UPDATED ---

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
    def analyze_video(
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
        if self.peoples_list:
             self._current_known_persons.update(self.peoples_list.get("persons", []))
        if self.actions_list:
             self._current_known_actions.update(self.actions_list.get("actions", []))
        if self.objects_list:
             self._current_known_objects.update(self.objects_list.get("objects", []))

        # =====  ADD THIS CALL  =====
        self._attach_transcripts_to_segments()
        # ===========================

        self.reprocess_segments = reprocess_segments
        self.person_group_id = person_group_id

        # --- UPDATED: Reset detailed token usage ---
        self.token_usage = {
            "chapters_prompt_tokens": 0, "chapters_completion_tokens": 0, "chapters_image_tokens": 0,
            "tags_prompt_tokens": 0, "tags_completion_tokens": 0, "tags_image_tokens": 0,
            "summary_prompt_tokens": 0, "summary_completion_tokens": 0,
            "report_input_text_tokens": 0, "report_output_text_tokens": 0,
            "report_input_image_tokens": 0, "report_total_tokens": 0
        }
        # --- END UPDATED ---

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
                # --- PATCH END ---

            finally:
                # --- ADDED: Call cleanup ---
                self._cleanup_temp_blobs()
                # --- END ADDED ---

            # --- Analysis complete, now process results ---
            # (Aggregation and summary logic remains here, using the 'processed_results' variable)
            # ... (rest of the logic starting from `final_results = {}`) ...

            # --- EDIT: Standardize final_results structure & Update Manifest ---
            final_results = {} # Initialize the dictionary to be saved as JSON

            # --- PATCH START ---
            if has_custom_aggregation:
                print("Using results from custom aggregation method.")
                # Assign the potentially processed results to final_results
                final_results = processed_results

                # --- Update manifest.global_tags from the actionSummary structure ---
                if "actionSummary" in final_results:
                    action_summary_content = final_results.get("actionSummary", {})
                    # Construct the global_tags structure expected by the manifest
                    # Use camelCase keys from action_summary_content, map to plural keys for manifest
                    manifest_global_tags = {
                         "persons": action_summary_content.get("person", []), # person -> persons
                         "actions": action_summary_content.get("action", []), # action -> actions
                         "objects": action_summary_content.get("object", [])  # object -> objects
                    }
                    self.manifest.global_tags = manifest_global_tags
                    print("DEBUG: Updated self.manifest.global_tags using data from actionSummary (mapped to plural keys).")
                else:
                    # Handle old custom structure (if needed) - unlikely with ActionSummary config
                    print("Warning: Custom aggregation result missing 'actionSummary'. Manifest global_tags not updated.")

            elif isinstance(processed_results, list): # This branch now handles non-custom agg OR refine
                # --- Apply Generic Aggregation to the list of segment results ---
                # This path handles non-custom aggregation and 'refine' sequence outputs.
                print("Running generic aggregation (produces standard structure).")
                try:
                    all_chapters_agg = []
                    global_tags_agg_dict = { "persons": {}, "actions": {}, "objects": {} }

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

                         for category in ["persons", "actions", "objects"]: # Internal keys 'persons', etc. are fine
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
                    self.manifest.global_tags = final_global_tags_agg_list
                    print("DEBUG: Updated self.manifest.global_tags from generic aggregation.")

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
                              print("Skipping final summary generation due to empty actionSummary content.")
                              # Add default summary/desc to actionSummary
                              summary_target_dict["description"] = description_text
                              summary_target_dict["summary"] = summary_text

                    # Check standard structure (if not actionSummary - e.g., generic fallback)
                    elif final_results and (final_results.get("chapters") or final_results.get("globalTags")):
                        should_generate_summary = True
                    else: # Standard structure but empty
                        print("Skipping final summary generation due to empty analysis results.")
                        # Add default summary/desc to top level
                        final_results["description"] = description_text
                        final_results["summary"] = summary_text


                    if should_generate_summary:
                        print(f"Generating summary and description for {self.manifest.name}")

                        # --- Determine summary length instruction ---
                        if video_duration < 30:
                            summary_length_instruction = "1-3 sentences long."
                        elif 30 <= video_duration <= 300: # 30 seconds to 5 minutes
                            summary_length_instruction = "1-2 paragraphs long."
                        else: # Over 5 minutes
                            summary_length_instruction = "at least 3 paragraphs long."
                        print(f"Video duration {video_duration:.2f}s. Requesting summary length: {summary_length_instruction}")

                        # --- Format the selected instruction into the prompt ---
                        # --- ADDED: Format asset categories list --- +
                        asset_cats_str = "Unknown"
                        if hasattr(analysis_config, 'ASSET_CATEGORIES') and isinstance(analysis_config.ASSET_CATEGORIES, list):
                             asset_cats_str = ", ".join([f'"{cat}"' for cat in analysis_config.ASSET_CATEGORIES])
                        else:
                             print("Warning: Could not find ASSET_CATEGORIES list in AnalysisConfig. Prompt will be incomplete.")
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
                        print(f"DEBUG: Added description and summary (camelCase) to {'actionSummary object' if 'actionSummary' in final_results else 'top level'}.")

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

        # --- ADDED: Generate Summary Report ---
        try:
            print("-" * 20)
            print("Generating Analysis Summary Report...")
            report_data = self._gather_report_data(final_results, analysis_config.name, elapsed_time, final_results_output_path)
            self._write_summary_report(report_data)
            print("Analysis Summary Report generated.")
            print("-" * 20)
        except Exception as report_e:
            print(f"Warning: Failed to generate analysis summary report: {report_e}")
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
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": self.image_detail_level},
                    }
                )

            # add user content to the messages
            messages.append({"role": "user", "content": user_content})

            # write the prompt to the manifest
            prompt_output_path = os.path.join(
                segment.segment_folder_path, f"{segment.segment_name}_prompt.json"
            )

            with open(prompt_output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(messages, indent=4))

            segment.segment_prompt_path = prompt_output_path

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
            return segment.analyzed_result.get(analysis_config.name)

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
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": [], "objects": []}}) # Use globalTags

        print(f"Analyzing segment {segment.segment_name} ({segment.start_time:.3f}s - {segment.end_time:.3f}s)")
        stopwatch_segment = time.time()

        max_retries = 3
        initial_delay = 1.0

        # --- Stage 1: Get Chapters ---
        print(f"  - Stage 1: Requesting Chapters...")
        chapters_result = {"chapters": []}
        for attempt in range(max_retries):
            try:
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment,
                    analysis_config=analysis_config,
                    generate_chapters=True,
                    generate_tags=False
                )
                if chapter_prompt:
                    chapter_llm_response = self._call_llm(chapter_prompt, log_token_category="chapters")
                    # --- EDIT: Print Raw Chapter Response ---
                    if chapter_llm_response and chapter_llm_response.choices:
                        raw_content = chapter_llm_response.choices[0].message.content
                        print(f"--- RAW CHAPTER RESPONSE (Segment: {segment.segment_name}, Attempt: {attempt+1}) ---")
                        print(raw_content)
                        print(f"--- END RAW CHAPTER RESPONSE (Segment: {segment.segment_name}) ---")
                    # --- END EDIT ---
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                        chapter_llm_response,
                        expecting_chapters=True,
                        expecting_tags=False
                    ) # Expects {"chapters": ...}
                    if not chapters_result.get("chapters"): # Check if key has content
                        # Check if the raw response was likely empty or problematic
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


        # --- Stage 2: Get Tags (Chunked by TAG_CHUNK_FRAME_COUNT) ---
        print(f"  - Stage 2: Requesting Tags in chunks of {self.TAG_CHUNK_FRAME_COUNT} frames...")
        all_frame_data = segment.frame_urls if segment.frame_urls else segment.segment_frames_file_path
        all_times = segment.segment_frame_time_intervals
        # Use defaultdict to easily append tags from chunks
        collected_chunk_tags = defaultdict(lambda: {"persons": [], "actions": [], "objects": []})
        # Initialize the final dict for this segment
        merged_global_tags = {"persons": [], "actions": [], "objects": []}

        if not all_frame_data:
            print("  - Stage 2: No frames/URLs found for segment. Skipping tag analysis.")
        else:
            num_frames = len(all_frame_data)
            for i in range(0, num_frames, self.TAG_CHUNK_FRAME_COUNT):
                chunk_frames = all_frame_data[i : i + self.TAG_CHUNK_FRAME_COUNT]
                chunk_times = all_times[i : i + self.TAG_CHUNK_FRAME_COUNT]

                if not chunk_frames: # Should not happen if range logic is correct
                    continue

                chunk_idx = (i // self.TAG_CHUNK_FRAME_COUNT) + 1
                print(f"    - Processing Tag Chunk {chunk_idx} ({len(chunk_frames)} frames)...")

                chunk_start_time = chunk_times[0]
                chunk_end_time = chunk_times[-1]

                chunk_parsed_tags = None # Result for this specific chunk
                for attempt in range(max_retries): # Retry loop per chunk
                    try:
                        tag_prompt = self._generate_segment_prompt(
                            segment=segment,
                            analysis_config=analysis_config,
                            frames_subset=chunk_frames, # Pass only chunk frames
                            times_subset=chunk_times,   # Pass only chunk times
                            generate_chapters=False,
                            generate_tags=True,
                            chunk_start_time=chunk_start_time, # Pass chunk boundaries
                            chunk_end_time=chunk_end_time
                        )

                        if tag_prompt:
                            tag_llm_response = self._call_llm(tag_prompt, log_token_category="tags") # Log as 'tags'
                            chunk_parsed_tags = self._parse_llm_json_response(
                                tag_llm_response, expecting_chapters=False, expecting_tags=True
                            )
                            # Basic validation of chunk result
                            if chunk_parsed_tags and isinstance(chunk_parsed_tags.get("globalTags"), dict):
                                print(f"      Chunk {chunk_idx}: Received valid tags.")
                                break # Success for this chunk
                            else:
                                print(f"      Chunk {chunk_idx}: Invalid or empty tags response (Attempt {attempt+1}). Content: {chunk_parsed_tags}")
                                # Optional: Add check for empty raw response like before
                                if attempt + 1 == max_retries:
                                     print(f"      Chunk {chunk_idx}: Max retries reached. Skipping tags for this chunk.")
                                     chunk_parsed_tags = None # Ensure it's None on failure
                                else: time.sleep(initial_delay * (2 ** attempt)) # Backoff

                        else:
                            print(f"    - Chunk {chunk_idx}: Skipping tag generation due to empty prompt.")
                            chunk_parsed_tags = None # Ensure None if no prompt
                            break # Exit retry loop for this chunk

                    except Exception as e:
                        print(f"      Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} failed. Error: {e}")
                        if attempt + 1 == max_retries:
                            print(f"      Chunk {chunk_idx}: Max retries reached. Skipping tags for this chunk.")
                        else:
                            wait_time = initial_delay * (2 ** attempt)
                            print(f"        Retrying chunk in {wait_time:.1f} seconds...")
                            time.sleep(wait_time)
                # --- End Chunk Retry Loop ---

                # --- Collect tags from the successfully processed chunk ---
                if chunk_parsed_tags and isinstance(chunk_parsed_tags.get("globalTags"), dict):
                    chunk_global_tags = chunk_parsed_tags["globalTags"]
                    for category in ["persons", "actions", "objects"]:
                        if category in chunk_global_tags and isinstance(chunk_global_tags[category], list):
                             # Just append the raw tag items with their chunk-specific timecodes
                             merged_global_tags[category].extend(chunk_global_tags[category])

            print(f"  - Stage 2: Finished processing all tag chunks for segment.")
        # --- End Main Chunk Loop ---

        # Now merged_global_tags contains all tags collected from chunks
        # The structure is still {"persons": [...], "actions": [...], "objects": [...]}
        # Proceed to Stage 4 using this merged_global_tags

        # --- Stage 4: Combine ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []),
            "globalTags": merged_global_tags # Use the tags collected from chunks
        }

        # --- Stage 5: Update Segment Object and Save ---
        segment.analyzed_result[analysis_config.name] = final_combined_result
        # Ensure analysis_completed list tracks unique names
        if analysis_config.name not in segment.analysis_completed:
            segment.analysis_completed.append(analysis_config.name)

        # Save intermediate results (optional but recommended)
        try:
             segment_result_path = os.path.join(segment.segment_folder_path, f"_segment_analyzed_result_{analysis_config.name}.json")
             with open(segment_result_path, "w", encoding="utf-8") as f:
                  json.dump(final_combined_result, f, indent=4, ensure_ascii=False)
        except Exception as e:
             print(f"Warning: Could not save intermediate segment result for {segment.segment_name}: {e}")

        # Update manifest (This might need locking if multiple async segments write simultaneously)
        # For simplicity, we assume write_video_manifest handles potential concurrent writes gracefully,
        # or we accept the risk of potential race conditions if updates are very frequent.
        # A safer approach would involve collecting all results and writing the manifest once at the end
        # of _analyze_segment_list_async, or using an async lock.
        write_video_manifest(self.manifest)

        # Update known tags (potential race condition if shared state isn't handled carefully)
        # Similar to manifest writing, modifying shared sets like _current_known_* needs care.
        # For map-reduce, this update might be less critical for cross-segment influence during the run.
        if isinstance(merged_global_tags, dict):
             for tag_type, tag_list in merged_global_tags.items():
                  if isinstance(tag_list, list):
                      for item in tag_list:
                          if isinstance(item, dict):
                              name = item.get("name")
                              if name and isinstance(name, str):
                                   cleaned_name = name.strip()
                                   if cleaned_name:
                                       if tag_type == "persons": self._current_known_persons.add(cleaned_name)
                                       elif tag_type == "actions": self._current_known_actions.add(cleaned_name)
                                       elif tag_type == "objects": self._current_known_objects.add(cleaned_name)


        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis in {elapsed_segment:.2f} seconds.")

        return final_combined_result # Return the combined result

    async def _analyze_segment_async(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
    ):
        if (not self.reprocess_segments
            and analysis_config.name in segment.analysis_completed):
            print(f"Segment {segment.segment_name} already analyzed (async), skipping.")
            # --- EDIT: Return structure consistent with camelCase ---
            return segment.analyzed_result.get(analysis_config.name, {"chapters": [], "globalTags": {"persons": [], "actions": [], "objects": []}}) # Use globalTags

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
                chapter_prompt = self._generate_segment_prompt(
                    segment=segment, analysis_config=analysis_config, generate_chapters=True, generate_tags=False
                )
                if chapter_prompt:
                    chapter_llm_response = await self._call_llm_async(chapter_prompt, log_token_category="chapters")
                    # --- EDIT: Print Raw Chapter Response ---
                    if chapter_llm_response and chapter_llm_response.choices:
                        raw_content = chapter_llm_response.choices[0].message.content
                        print(f"--- RAW CHAPTER RESPONSE (Segment: {segment.segment_name}, Attempt: {attempt+1}, Async) ---")
                        print(raw_content)
                        print(f"--- END RAW CHAPTER RESPONSE (Segment: {segment.segment_name}, Async) ---")
                    # --- END EDIT ---
                    # --- EDIT: _parse_llm_json_response returns camelCase keys internally ---
                    chapters_result = self._parse_llm_json_response(
                         chapter_llm_response, expecting_chapters=True, expecting_tags=False
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

        # --- Stage 2: Get Tags (Chunked by TAG_CHUNK_FRAME_COUNT) ---
        print(f"  - Stage 2: Requesting Tags in chunks of {self.TAG_CHUNK_FRAME_COUNT} frames...")
        all_frame_data = segment.frame_urls if segment.frame_urls else segment.segment_frames_file_path
        all_times = segment.segment_frame_time_intervals
        # Use defaultdict to easily append tags from chunks
        collected_chunk_tags = defaultdict(lambda: {"persons": [], "actions": [], "objects": []})
        # Initialize the final dict for this segment
        merged_global_tags = {"persons": [], "actions": [], "objects": []}

        if not all_frame_data:
            print("  - Stage 2: No frames/URLs found for segment. Skipping tag analysis.")
        else:
            num_frames = len(all_frame_data)
            for i in range(0, num_frames, self.TAG_CHUNK_FRAME_COUNT):
                chunk_frames = all_frame_data[i : i + self.TAG_CHUNK_FRAME_COUNT]
                chunk_times = all_times[i : i + self.TAG_CHUNK_FRAME_COUNT]

                if not chunk_frames: # Should not happen if range logic is correct
                    continue

                chunk_idx = (i // self.TAG_CHUNK_FRAME_COUNT) + 1
                print(f"    - Processing Tag Chunk {chunk_idx} ({len(chunk_frames)} frames)...")

                chunk_start_time = chunk_times[0]
                chunk_end_time = chunk_times[-1]

                chunk_parsed_tags = None # Result for this specific chunk
                for attempt in range(max_retries): # Retry loop per chunk
                    try:
                        tag_prompt = self._generate_segment_prompt(
                            segment=segment,
                            analysis_config=analysis_config,
                            frames_subset=chunk_frames, # Pass only chunk frames
                            times_subset=chunk_times,   # Pass only chunk times
                            generate_chapters=False,
                            generate_tags=True,
                            chunk_start_time=chunk_start_time, # Pass chunk boundaries
                            chunk_end_time=chunk_end_time
                        )

                        if tag_prompt:
                            tag_llm_response = self._call_llm(tag_prompt, log_token_category="tags") # Log as 'tags'
                            chunk_parsed_tags = self._parse_llm_json_response(
                                tag_llm_response, expecting_chapters=False, expecting_tags=True
                            )
                            # Basic validation of chunk result
                            if chunk_parsed_tags and isinstance(chunk_parsed_tags.get("globalTags"), dict):
                                print(f"      Chunk {chunk_idx}: Received valid tags.")
                                break # Success for this chunk
                            else:
                                print(f"      Chunk {chunk_idx}: Invalid or empty tags response (Attempt {attempt+1}). Content: {chunk_parsed_tags}")
                                # Optional: Add check for empty raw response like before
                                if attempt + 1 == max_retries:
                                     print(f"      Chunk {chunk_idx}: Max retries reached. Skipping tags for this chunk.")
                                     chunk_parsed_tags = None # Ensure it's None on failure
                                else: time.sleep(initial_delay * (2 ** attempt)) # Backoff

                        else:
                            print(f"    - Chunk {chunk_idx}: Skipping tag generation due to empty prompt.")
                            chunk_parsed_tags = None # Ensure None if no prompt
                            break # Exit retry loop for this chunk

                    except Exception as e:
                        print(f"      Chunk {chunk_idx}: Attempt {attempt + 1}/{max_retries} failed. Error: {e}")
                        if attempt + 1 == max_retries:
                            print(f"      Chunk {chunk_idx}: Max retries reached. Skipping tags for this chunk.")
                        else:
                            wait_time = initial_delay * (2 ** attempt)
                            print(f"        Retrying chunk in {wait_time:.1f} seconds...")
                            time.sleep(wait_time)
                # --- End Chunk Retry Loop ---

                # --- Collect tags from the successfully processed chunk ---
                if chunk_parsed_tags and isinstance(chunk_parsed_tags.get("globalTags"), dict):
                    chunk_global_tags = chunk_parsed_tags["globalTags"]
                    for category in ["persons", "actions", "objects"]:
                        if category in chunk_global_tags and isinstance(chunk_global_tags[category], list):
                             # Just append the raw tag items with their chunk-specific timecodes
                             merged_global_tags[category].extend(chunk_global_tags[category])

            print(f"  - Stage 2: Finished processing all tag chunks for segment.")
        # --- End Main Chunk Loop ---

        # Now merged_global_tags contains all tags collected from chunks
        # The structure is still {"persons": [...], "actions": [...], "objects": [...]}
        # Proceed to Stage 4 using this merged_global_tags

        # --- Stage 4: Combine ---
        final_combined_result = {
            "chapters": chapters_result.get("chapters", []),
            "globalTags": merged_global_tags # Use the tags collected from chunks
        }

        # --- Stage 5: Update Segment & Save ---
        segment.analyzed_result[analysis_config.name] = final_combined_result
        if analysis_config.name not in segment.analysis_completed:
            segment.analysis_completed.append(analysis_config.name)
        # Save intermediate results (Consider async file writing if it becomes a bottleneck)
        try:
             segment_result_path = os.path.join(segment.segment_folder_path, f"_segment_analyzed_result_{analysis_config.name}.json")
             with open(segment_result_path, "w", encoding="utf-8") as f:
                  json.dump(final_combined_result, f, indent=4, ensure_ascii=False)
        except Exception as e:
             print(f"Warning: Could not save intermediate segment result for {segment.segment_name} (async): {e}")

        # Update manifest (This might need locking if multiple async segments write simultaneously)
        # For simplicity, we assume write_video_manifest handles potential concurrent writes gracefully,
        # or we accept the risk of potential race conditions if updates are very frequent.
        # A safer approach would involve collecting all results and writing the manifest once at the end
        # of _analyze_segment_list_async, or using an async lock.
        write_video_manifest(self.manifest)

        # Update known tags (potential race condition if shared state isn't handled carefully)
        # Similar to manifest writing, modifying shared sets like _current_known_* needs care.
        # For map-reduce, this update might be less critical for cross-segment influence during the run.
        if isinstance(merged_global_tags, dict):
             for tag_type, tag_list in merged_global_tags.items():
                  if isinstance(tag_list, list):
                      for item in tag_list:
                          if isinstance(item, dict):
                              name = item.get("name")
                              if name and isinstance(name, str):
                                   cleaned_name = name.strip()
                                   if cleaned_name:
                                       if tag_type == "persons": self._current_known_persons.add(cleaned_name)
                                       elif tag_type == "actions": self._current_known_actions.add(cleaned_name)
                                       elif tag_type == "objects": self._current_known_objects.add(cleaned_name)


        elapsed_segment = time.time() - stopwatch_segment
        print(f"Segment {segment.segment_name} finished analysis (async) in {elapsed_segment:.2f} seconds.")

        return final_combined_result # Return camelCase structure

    def _generate_segment_prompt(
        self,
        segment: Segment,
        analysis_config: AnalysisConfig,
        # --- MODIFIED: These subsets will now contain URLs if using blob approach ---
        frames_subset: List[str] = None,
        times_subset: List[float] = None,
        # --- END MODIFIED ---
        generate_chapters: bool = True,
        generate_tags: bool = True,
        chunk_start_time: Optional[float] = None,
        chunk_end_time: Optional[float] = None,
        is_partial_chunk: bool = False
    ):
        """
        Generate a prompt for a segment, adaptable for full context or tag-specific chunks.
        Handles chapters OR tags, not both simultaneously.
        Uses frame URLs if available on the segment, otherwise falls back (though fallback shouldn't occur with new workflow).
        """
        # --- MODIFIED: Prioritize URLs if available ---
        # Check if segment has URLs populated AND if a subset wasn't explicitly passed
        if segment.frame_urls and frames_subset is None:
             frames_to_process = segment.frame_urls
             # Use corresponding full time intervals if using full URL list
             times_to_process = segment.segment_frame_time_intervals if times_subset is None else times_subset
             print(f"DEBUG: Using {len(frames_to_process)} frame URLs for prompt generation in {segment.segment_name}.")
        elif frames_subset is not None:
             # If a subset is passed, assume it's the correct type (URLs after preprocessing)
             frames_to_process = frames_subset
             times_to_process = times_subset # Assume times_subset matches frames_subset
             print(f"DEBUG: Using {len(frames_to_process)} frame URLs (subset) for prompt generation in {segment.segment_name}.")
        else:
             # Fallback to file paths ONLY if URLs aren't available (shouldn't happen)
             frames_to_process = segment.segment_frames_file_path
             times_to_process = segment.segment_frame_time_intervals if times_subset is None else times_subset
             print(f"Warning: Frame URLs not found for {segment.segment_name}. Falling back to local paths (potential issue).")
        # --- END MODIFIED ---


        transcription_context = segment.transcription if segment.transcription else "No transcription available"
        messages = []
        results_template_str_for_reminder = "{}"

        # Check if we have *anything* to process (URLs or paths)
        if not frames_to_process:
             print(f"Warning: No frames (URLs or paths) to process for prompt generation in segment {segment.segment_name}.")
             return None

        # Ensure frame times match frames (logic remains useful)
        if not times_to_process or len(times_to_process) != len(frames_to_process):
             print(f"Warning: Mismatch or missing times for frames in segment {segment.segment_name}. Times: {len(times_to_process)}, Frames: {len(frames_to_process)}. Recalculating...")
             # Recalculation logic (remains the same)
             effective_start = chunk_start_time if chunk_start_time is not None else segment.start_time
             effective_end = chunk_end_time if chunk_end_time is not None else segment.end_time
             if len(frames_to_process) > 0:
                  times_to_process = np.linspace(effective_start, effective_end, len(frames_to_process), endpoint=False) # Use endpoint=False if time corresponds to start of frame interval
                  times_to_process = [round(float(t), 3) for t in times_to_process]
             else:
                  times_to_process = []


        # --- Determine which prompt components to use ---
        if generate_chapters and generate_tags:
             raise ValueError("Cannot generate chapters and tags in the same prompt.")

        elif generate_chapters:
            # ... (Fetch system_prompt_template, lens_prompt, results_template_str_for_reminder)
            # --- FIX: Use correct system prompt --- 
            # Get template from config
            if not hasattr(analysis_config, 'system_prompt_chapters'):
                 print("Warning: AnalysisConfig missing 'system_prompt_chapters'. Falling back.")
                 system_prompt_template = "Default system prompt for chapters. Please provide chapter summary."
            else:
                 system_prompt_template = analysis_config.system_prompt_chapters
            # --- END FIX ---

            # --- ADDED: Prepare shot types list for formatting ---
            shot_types_str = "Unknown"
            if hasattr(analysis_config, 'SHOT_TYPES') and isinstance(analysis_config.SHOT_TYPES, list):
                shot_types_str = ", ".join([f'"{shot}"' for shot in analysis_config.SHOT_TYPES])
            else:
                print("Warning: Could not find SHOT_TYPES list in AnalysisConfig. Prompt will be incomplete.")
            # --- END ADDED ---

            # --- ADDED: Prepare format_data_chapters ---
            format_data_chapters = {
                "shot_types_list": shot_types_str,
                "lens_prompt": analysis_config.lens_prompt,
                "results_template_str_for_reminder": results_template_str_for_reminder,
                # Note: Lens is added later to user prompt for chapters
            }
            # *** Use format_map instead of format ***
            try:
                system_prompt = self._safe_format(system_prompt_template, **format_data_chapters)
                # --- FIX Nit 2: Correctly set reminder string for chapters --- 
                results_template_obj_ch = getattr(analysis_config, 'results_template_chapters', {"chapters": []})
                results_template_str_for_reminder = json.dumps(results_template_obj_ch, indent=2)
                # --- END FIX --- 
            except KeyError as e:
                 print(f"ERROR: Missing key {repr(e.args[0])} during chapter system_prompt formatting. Using template as is.")
                 system_prompt = system_prompt_template # Fallback
            except Exception as e:
                 print(f"ERROR: Unexpected error during chapter system_prompt formatting: {e}. Using template as is.")
                 system_prompt = system_prompt_template
            messages.append({"role": "system", "content": system_prompt})
            # --- END ADDED ---

            # --- User Content for Chapters (Using URLs) ---
            user_content = []
            if transcription_context and transcription_context != "No transcription available":
                user_content.append(
                    {"type": "text",
                     "text": f"Audio transcription ({segment.segment_duration:.3f}s): {transcription_context}"}
                )
            else:
                 user_content.append({"type": "text", "text": "No transcription provided."})

            user_content.append({"type": "text", "text": "\nFrames:"}) # Add separator
            for idx, (frame_url_or_path, ts) in enumerate(zip(frames_to_process, times_to_process), 1):
                user_content.append({"type": "text",
                                     "text": f"\nImage #{idx} at {ts:.3f}s"})
                # --- MODIFIED: Use URL directly ---
                if frame_url_or_path.startswith("http"): # Simple check for URL
                     user_content.append({"type": "image_url",
                                          "image_url": {"url": frame_url_or_path,
                                                        "detail": self.image_detail_level}})
                else:
                    # Fallback to base64 if it's a path (shouldn't happen now)
                    print(f"Warning: Frame {idx} for chapter prompt is a path, encoding base64: {frame_url_or_path}")
                    try:
                         b64 = encode_image_base64(frame_url_or_path)
                         if not b64: continue
                         user_content.append({"type": "image_url",
                                             "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                                           "detail": self.image_detail_level}})
                    except Exception as e:
                         print(f"Warning: Error encoding frame path {frame_url_or_path} for chapter prompt: {e}")
                # --- END MODIFIED ---

            messages.append({"role": "user", "content": user_content})
            # --- End User Content ---


        elif generate_tags:
            # --- FIX: Use correct system prompt ---
            system_prompt_template = getattr(
                analysis_config,
                "system_prompt_tags",
                "Default system prompt for tags."
            )

            # Look up user‑supplied instruction files (may return "No … provided.")
            person_instr = self._get_list_definitions(self.peoples_list, "persons")
            object_instr = self._get_list_definitions(self.objects_list, "objects")
            action_instr = self._get_list_definitions(self.actions_list, "actions")

                       # FALLBACK, DOMAIN‑AGNOSTIC INSTRUCTIONS – used only if no custom file was found
            if person_instr == "No specific persons definitions provided.":
                person_instr = """
                                Tag each person **by the exact frame** they first appear and the frame they disappear, using the provided `frame_timestamps`:
                                - For each individual visible ≥ 1 s, set `start` to the timestamp of their first visible frame and `end` to the timestamp of their last visible frame.
                                - If they leave view for > 1 s and then reappear, create a new interval under the same name.
                                - Use a real name if known; otherwise a short descriptive label (e.g. "woman in blue scarf", "man with briefcase").
                                - Ignore static images, reflections, posters, or mannequins that never move—those are **objects**.
                                """.strip()

            if object_instr == "No specific objects definitions provided.":
                object_instr = """
                                Tag each physical object **by the exact frame** it first and last appears, using the provided `frame_timestamps`:
                                - For an object visible ≥ 1 s, set `start` to its first visible‐frame timestamp and `end` to its last visible‐frame timestamp.
                                - If it goes off‐screen for > 1 s and returns, record a new interval under the same name.
                                - Only include objects relevant to understanding the scene or the people's actions.
                                - When multiple identical items appear simultaneously, use a plural name (e.g. "chairs") with one interval covering the full union of their visibility.
                                """.strip()

            if action_instr == "No specific actions definitions provided.":
                action_instr = """
                                Tag each human action **by the exact frame** it begins and ends, using the provided `frame_timestamps`:
                                - For any action lasting ≥ 1 s, use its first action‐frame timestamp as `start` and its last as `end`.
                                - If the same person stops and restarts that action after > 1 s, record separate intervals.
                                - Describe actions in present‐participle (e.g. "walking", "typing on laptop").
                                - If multiple people perform the same action simultaneously, one tag is enough—the interval should span the union of their activity.
                                """.strip()

            # ------------------------------------------------------------------

            # Build the final system prompt
            try:
                system_prompt = self._safe_format(
                    system_prompt_template,
                    person_instructions=person_instr,
                    object_instructions=object_instr,
                    action_instructions=action_instr,
                )

                # For the "final reminder" later on
                results_template_obj = getattr(
                    analysis_config,
                    "results_template_tags",
                    {"globalTags": {"persons": [], "actions": [], "objects": []}},
                )
                results_template_str_for_reminder = json.dumps(
                    results_template_obj,
                    indent=2
                )
            except Exception as e:
                print(
                    f"ERROR: Unexpected error during tag system_prompt formatting: {e}. "
                    "Using template as is."
                )
                system_prompt = system_prompt_template  # Fallback

            messages.append({"role": "system", "content": system_prompt})
            # --- END FIX ---


            # --- User Content for Tags (Using URLs) - Keep existing logic ---
            user_content = [] # Start empty list
            # ... (Existing logic to build metadata_block, metadata_text, append text) ...
            effective_start = chunk_start_time if chunk_start_time is not None else segment.start_time
            effective_end = chunk_end_time if chunk_end_time is not None else segment.end_time

            frame_timestamps_map = { str(i + 1): f"{timestamp:.3f}s" for i, timestamp in enumerate(times_to_process) }
            metadata_block = {
                "segment_start": f"{effective_start:.3f}s",
                "segment_end": f"{effective_end:.3f}s",
                "frame_timestamps": frame_timestamps_map,
                "transcription_context": transcription_context
            }
            metadata_text = f"Input Data:\n```json\n{json.dumps(metadata_block, indent=2)}\n```\n\nFrames:"
            user_content.append({"type": "text", "text": metadata_text})

            # Append images (existing logic is correct)
            for i, (frame_url_or_path, timestamp) in enumerate(zip(frames_to_process, times_to_process)):
                 frame_id = str(i + 1)
                 if f"{timestamp:.3f}s" not in frame_timestamps_map.values():
                      print(f"Warning: Frame timestamp {timestamp:.3f}s (Index {i}) missing from timestamp map for tag prompt. Skipping.")
                      continue
                 user_content.append({"type": "text", "text": f"\nImage #{frame_id}:"})
                 if frame_url_or_path.startswith("http"):
                      user_content.append({"type": "image_url", "image_url": {"url": frame_url_or_path, "detail": self.image_detail_level}})
                 else:
                     print(f"Warning: Frame {frame_id} for tag prompt is a path, encoding base64: {frame_url_or_path}")
                     try:
                          b64 = encode_image_base64(frame_url_or_path)
                          if not b64: continue
                          user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": self.image_detail_level}})
                     except Exception as e:
                          print(f"Warning: Error encoding frame path {frame_url_or_path} for tag prompt: {e}")

            messages.append({"role": "user", "content": user_content})
            # --- End User Content ---

        else:
             print("Warning: _generate_segment_prompt called without specifying chapters or tags.")
             return None

        # --- Final Reminder Logic (remains the same, but should now work reliably) ---
        final_user_content = messages[-1]["content"]
        if not isinstance(final_user_content, list):
            print("Warning: Could not retrieve user content list for final reminder. Skipping.")
            # If this happens now, something is fundamentally wrong earlier
            return messages # Return messages as is

        # --- FIX: Enhance Final Reminder --- 
        if generate_chapters:
             final_reminder_text = (
                 "CRITICAL: Your output MUST be valid JSON exactly matching this schema. " # Added space
                 "Return your JSON **inside \`\`\`json ... \`\`\`** and nothing else.\n"
                 f"schema:\n```json\n{results_template_str_for_reminder}\n```"
             )
        elif generate_tags:
             start_t = chunk_start_time if chunk_start_time is not None else segment.start_time
             end_t = chunk_end_time if chunk_end_time is not None else segment.end_time
             if not results_template_str_for_reminder or results_template_str_for_reminder == "{}":
                  results_template_str_for_reminder = json.dumps({"globalTags": {"persons":[],"actions":[],"objects":[]}}, indent=2)
             final_reminder_text = (
                 "CRITICAL: Your output MUST be valid JSON exactly matching this schema. " # Added space
                 "Return your JSON **inside \`\`\`json ... \`\`\`** and nothing else.\n"
                 f"schema (Tags MUST be strictly within {start_t:.3f}s to {end_t:.3f}s):\n"
                 f"```json\n{results_template_str_for_reminder}\n```"
             )
        else: # Should not happen
             final_reminder_text = "CRITICAL: Your output MUST be VALID JSON."
        # --- END FIX ---

        final_user_content.append({"type": "text", "text": final_reminder_text})
        messages[-1]["content"] = final_user_content # Update the last message

        return messages # Return the complete messages list

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
    def _timecodes_to_intervals(self, timecodes: list) -> list[list[float]]:
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
    # ----------------------------------------------------------------------
    def _merge_overlapping(self, intervals: list[list[float]], max_gap: float = 2.0) -> list[list[float]]:
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
                              print(f"Warning: Skipping interval with None start/end for tag '{name}': {i}")
                    except (ValueError, TypeError) as e:
                         print(f"Warning: Skipping invalid interval for tag '{name}': {i}. Error: {e}")

            merged = self._merge_overlapping(num_ints, max_gap=max_gap_initial)
            out[name] = [{"start": s, "end": e} for s, e in merged]
        return out

    def _parse_llm_json_response(self, response, expecting_chapters=True, expecting_tags=True):
         """Parses the LLM response, expecting JSON in a markdown block."""
         try:
             if not response or not hasattr(response, 'choices') or not response.choices:
                  print("Error: Invalid LLM response object or no choices found.")
                  return {"error": "Invalid LLM response object"}

             raw_content = response.choices[0].message.content
             # --- ADDED: Log raw response ---
             log_identifier = "Chapters" if expecting_chapters else "Tags"
             print(f"\n--- DEBUG RAW LLM RESPONSE ({log_identifier}) ---\n{raw_content[:2000]}...\n---------------------------------------\n") # Log first 2k chars
             # --- END ADDED ---


             # --- FIX: Make parsing more tolerant --- 
             code_to_parse = raw_content.strip()
             if code_to_parse.startswith("```json"):
                 code_to_parse = code_to_parse.split("```json", 1)[1].strip()
             if code_to_parse.endswith("```"):
                 code_to_parse = code_to_parse.rsplit("```", 1)[0].strip()

             # Now try parsing the extracted/cleaned code
             try:
                  parsed_response = json.loads(code_to_parse)
                  # Basic validation (ensure it's a dict)
                  if not isinstance(parsed_response, dict):
                       print("Error: Parsed JSON is not a dictionary.")
                       return {} # Return empty dict on type error
                  return parsed_response
             except json.JSONDecodeError as e:
                  print(f"Error: Failed to decode JSON from LLM response: {e}")
                  print(f"Content attempted: {code_to_parse[:500]}...")
                  return {} # Return empty dict on JSON decode error
             # --- END FIX ---

         except Exception as e:
               print(f"Error parsing LLM response: {e}")
               # Log the raw content on general parsing error too
               raw_content_on_error = getattr(response.choices[0].message, 'content', 'Raw content unavailable')
               print(f"Failed parsing raw content:\n{raw_content_on_error[:1000]}...\n")
               return {} # Return empty dict on other errors
    # --- END ADDED ---

    def _count_tokens(self, text, model="gpt-3.5-turbo"):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception as e:
            print(f"Token counting failed: {e}")
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
             print(f"DEBUG: Using 'max_completion_tokens={max_token_value}' for model '{self.env.vision.deployment}'")
        else:
             completion_params["max_tokens"] = max_token_value
             print(f"DEBUG: Using 'max_tokens={max_token_value}' for model '{self.env.vision.deployment}'")
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
            
            print(f"[TOKENS] {log_token_category}: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}. Updated detailed and report totals.")
        else:
            print(f"[TOKENS] Uncategorized: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}")
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
             print(f"DEBUG: Using 'max_completion_tokens={max_token_value}' for model '{self.env.vision.deployment}' (async)")
        else:
             completion_params["max_tokens"] = max_token_value
             print(f"DEBUG: Using 'max_tokens={max_token_value}' for model '{self.env.vision.deployment}' (async)")
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
            
            print(f"[TOKENS] {log_token_category}: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}. Updated detailed and report totals.")
        else:
            print(f"[TOKENS] Uncategorized: prompt={prompt_tokens}, completion={completion_tokens}, images={image_tokens}, total={total_with_images}")
        return response

    # --- ADDED: Cleanup Helper ---
    def _cleanup_temp_blobs(self):
        """Deletes temporary frame blobs from Azure Storage."""
        print("Starting cleanup of temporary frame blobs...")
        cleanup_start_time = time.time()
        deleted_count = 0
        failed_count = 0
        try:
             # Ensure blob storage is configured
             if not self.env.blob_storage or not self.env.blob_storage.account_name or \
                not self.env.blob_storage.container_name or \
                (not self.env.blob_storage.connection_string and not self.env.blob_storage.sas_token):
                  print("Blob storage not configured. Skipping cleanup.")
                  return

             blob_service_client = None
             # --- Recreate BlobServiceClient as in upload_blob ---
             if self.env.blob_storage.connection_string:
                 connect_str = self.env.blob_storage.connection_string.get_secret_value()
                 blob_service_client = BlobServiceClient.from_connection_string(connect_str)
             elif self.env.blob_storage.sas_token:
                  # Deleting might require more permissions than the upload SAS token provides
                  # Best practice is to use connection string or an account-level SAS/principal for management tasks
                  print("Warning: Attempting blob cleanup using SAS token. This might fail if token lacks delete permissions.")
                  account_url = f"https://{self.env.blob_storage.account_name}.blob.core.windows.net"
                  sas_token_str = self.env.blob_storage.sas_token.get_secret_value()
                  blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token_str)
             else:
                  print("No valid Blob Storage auth for cleanup.")
                  return
             # --- End Recreate Client ---

             container_client = blob_service_client.get_container_client(self.env.blob_storage.container_name)
             print(f"Connected to container: {self.env.blob_storage.container_name}")

             for seg in self.manifest.segments:
                  # Use getattr for safety in case _blob_names wasn't set
                  blob_names_to_delete = getattr(seg, "_blob_names", [])
                  if blob_names_to_delete:
                       print(f"Cleaning up {len(blob_names_to_delete)} blobs for {seg.segment_name}...")
                       for blob_name in blob_names_to_delete:
                            if not blob_name: continue # Skip empty names
                            try:
                                 print(f"  Deleting blob: {blob_name}")
                                 container_client.delete_blob(blob_name)
                                 deleted_count += 1
                            except Exception as e:
                                 # Log specific error but continue cleanup
                                 print(f"  Warning: Could not delete blob {blob_name}: {e}")
                                 failed_count += 1
        except Exception as e:
             print(f"ERROR during blob cleanup setup or iteration: {e}")
             failed_count += 1 # Count general failure too
        finally:
            cleanup_duration = time.time() - cleanup_start_time
            print(f"Blob cleanup finished in {cleanup_duration:.2f}s. Deleted: {deleted_count}, Failed/Skipped: {failed_count}")
    # --- END ADDED ---

    # ------------------------------------------------------------------
    # Transcript helpers (ADD THESE METHODS)
    # ------------------------------------------------------------------
    def _normalise_phrase(self, phrase_json: Dict, seg_id: int) -> Dict:
        """Turn one Azure 'recognizedPhrase' into the target schema."""
        off = phrase_json.get("offsetInTicks", 0) / 10_000_000.0
        dur = phrase_json.get("durationInTicks", 0) / 10_000_000.0
        text = (phrase_json.get("nBest", [{}])[0]).get("display", "").strip()

        # Handle speaker info - prefer 'speaker' field, fallback to 'speakerId'
        speaker_val = phrase_json.get('speaker') # Prefer explicit speaker number
        if speaker_val is None:
             speaker_id = phrase_json.get('speakerId', 0) # Fallback to speakerId
             speaker_val = f"Speaker_{speaker_id}"
        else:
             speaker_val = f"Speaker_{speaker_val}" # Ensure format is Speaker_N

        return {
            "segmentId":          seg_id,
            "speaker":            speaker_val,
            "startSec":           round(off, 3), # Round seconds
            "endSec":             round(off + dur, 3), # Round seconds
            "confidence":         (phrase_json.get("nBest", [{}])[0]).get("confidence"),
            "textVariants": {
                "display":        text,
                "lexical":        (phrase_json.get("nBest", [{}])[0]).get("lexical", "").strip(),
                "maskedITN":      (phrase_json.get("nBest", [{}])[0]).get("maskedITN", "").strip()
            },
            "offsetInTicks":      phrase_json.get("offsetInTicks"),
            "durationInTicks":    phrase_json.get("durationInTicks"),
            "duration":           seconds_to_iso8601_duration(dur)
        }

    def _attach_transcripts_to_segments(self) -> None:
        """
        • Builds `self.manifest.transcription_details.segments` (intended structure, field doesn't exist yet)
        • Adds `transcription` *and* `transcription_segments` to each Segment
        """
        tr_obj = self.manifest.audio_transcription
        if not isinstance(tr_obj, dict) or "recognizedPhrases" not in tr_obj:
            print("No Batch-STT JSON attached – skipping transcript attach.")
            # Clear existing segment transcript fields if no source data
            for seg in self.manifest.segments:
                 seg.transcription = "No transcription available."
                 seg.transcription_segments = []
            return

        # ---- 1 · normalise the whole list --------------------------------
        phrases = tr_obj.get("recognizedPhrases", [])
        norm_segments = []
        if isinstance(phrases, list):
            norm_segments = [self._normalise_phrase(p, i + 1) for i, p in enumerate(phrases) if isinstance(p, dict)] # Use 1-based segmentId
        else:
             print(f"Warning: Expected 'recognizedPhrases' to be a list, got {type(phrases)}. Skipping.")
             return

        # Store the full normalized list for potential later use (e.g., in final JSON)
        # Note: VideoManifest model doesn't have a dedicated 'transcription_details' field yet.
        # We can store it temporarily or decide where it should live permanently.
        # For now, let's store it in a temporary attribute or just use norm_segments locally.
        # Let's assume for now we only attach slices to segments.
        print(f"DEBUG: Normalized {len(norm_segments)} transcription phrases.")

        # ---- 2 · slice per video-segment ---------------------------------
        for seg in self.manifest.segments:
             # pick phrases whose time window overlaps this video segment
             slice_json = []
             if seg.start_time is not None and seg.end_time is not None:
                  slice_json = [
                      ph for ph in norm_segments
                      if ph["startSec"] < seg.end_time and ph["endSec"] > seg.start_time
                  ]
             else:
                  print(f"Warning: Segment {seg.segment_name} missing start/end time, cannot attach transcription slice.")

             seg.transcription_segments = slice_json
             seg.transcription = " ".join(p["textVariants"]["display"] for p in slice_json).strip()
             # Add a default if no overlapping segments found
             if not seg.transcription:
                 seg.transcription = "No transcription for this segment's time range."
        print("DEBUG: Attached transcription slices to segments.")


    # --- ADDED: Method to Calculate Cost ---
    def _calculate_cost(self) -> Optional[float]:
        deployment_name = self.env.vision.deployment
        pricing = self.PRICING_INFO.get(deployment_name)

        if not pricing:
             # Try common aliases or base model names if specific deployment name not found
             if "gpt-4o-mini" in deployment_name: pricing = self.PRICING_INFO.get("gpt-4o-mini")
             elif "gpt-4o" in deployment_name: pricing = self.PRICING_INFO.get("gpt-4o")
             elif "gpt-4-turbo" in deployment_name: pricing = self.PRICING_INFO.get("gpt-4-turbo")
             # Add more fallback logic if needed

        if not pricing:
            print(f"Warning: Pricing information not found for deployment '{deployment_name}'. Cannot calculate cost.")
            return None

        input_text_price = pricing.get("input_per_million", 0)
        output_text_price = pricing.get("output_per_million", 0)

        # Azure's report_input_text_tokens already includes the image token equivalent cost
        api_prompt_tokens = self.token_usage.get("report_input_text_tokens", 0)
        api_completion_tokens = self.token_usage.get("report_output_text_tokens", 0)

        cost = 0.0
        cost += (api_prompt_tokens / 1_000_000 * input_text_price)
        cost += (api_completion_tokens / 1_000_000 * output_text_price)
        
        print(f"INFO: Calculated cost based on total Azure input tokens (including image equivalent) and output tokens.")
        return cost

    # --- ADDED: Method to Upload Action Summary ---
    def _upload_action_summary(self, local_summary_path: str) -> Optional[str]:
        """Uploads the ActionSummary.json to blob storage and returns its SAS URL."""
        if not self.env.blob_storage or not self.env.blob_storage.account_name or \
           not self.env.blob_storage.container_name or \
           (not self.env.blob_storage.connection_string and not self.env.blob_storage.sas_token):
            print("Blob storage not configured for writing outputs. Skipping ActionSummary upload.")
            return None

        # Check if the local file exists
        if not os.path.exists(local_summary_path):
            print(f"Warning: Local summary file not found at {local_summary_path}. Skipping upload.")
            return None

        # Determine blob name based on input video URL if possible
        summary_blob_name = f"{os.path.splitext(self.manifest.name)[0]}_ActionSummary.json" # Default name
        try:
            if self.video_blob_url:
                parsed_url = urlparse(self.video_blob_url)
                # Extract path, remove leading '/' if present
                input_path = parsed_url.path.lstrip('/')
                # Remove container name from the start of the path
                container_name = self.env.blob_storage.container_name
                if input_path.startswith(container_name + '/'):
                     input_path = input_path[len(container_name) + 1:]

                # Construct output path
                input_dir = os.path.dirname(input_path)
                input_filename = os.path.splitext(os.path.basename(input_path))[0]
                output_subpath = f"{input_dir}/{input_filename}_analysis" if input_dir else f"{input_filename}_analysis"
                summary_blob_name = f"{output_subpath}/{os.path.basename(local_summary_path)}"
                print(f"Derived summary blob name: {summary_blob_name}")
            else:
                # Fallback: Place in a general 'analysis_outputs' folder
                output_subpath = "analysis_outputs"
                summary_blob_name = f"{output_subpath}/{os.path.basename(local_summary_path)}"
                print(f"Using default summary blob name: {summary_blob_name}")

            print(f"Attempting to upload {local_summary_path} to blob: {summary_blob_name}")
            # Use a longer SAS validity for the output file if needed, e.g., 7 days
            output_sas_url = upload_blob(
                local_file_path=local_summary_path,
                blob_name=summary_blob_name,
                env=self.env,
                overwrite=True,
                read_permission_hours=24 * 7 # e.g., 7 days validity
            )

            if output_sas_url:
                print(f"ActionSummary uploaded successfully.")
                return output_sas_url
            else:
                print("Warning: Failed to upload ActionSummary.")
                return None
        except Exception as e:
            print(f"Error during ActionSummary upload: {e}")
            return None

    # --- ADDED: Method to Gather Report Data ---
    def _gather_report_data(self, final_results: Dict, analysis_config_name: str, elapsed_time: float, local_summary_path: str) -> Dict:
        """Gathers all data points needed for the summary report."""
        print("Gathering data for summary report...")
        report = {}

        # --- Basic Info ---
        report["Filename"] = self.manifest.name or "N/A"
        report["Blob URL for file"] = self.video_blob_url or "N/A" # Use stored URL
        report["Duration"] = f"{self.manifest.source_video.duration:.3f}s" if self.manifest.source_video.duration else "N/A"
        report["Model used"] = self.env.vision.deployment or "N/A"
        report["Schema Version"] = "1.0.0" # Placeholder

        # --- Category/Subcategory from ActionSummary ---
        action_summary = final_results.get("actionSummary", {})
        report["Category"] = action_summary.get("category", "N/A")
        report["Subcategory"] = action_summary.get("subCategory", "N/A")

        # --- Scribe Output URL ---
        report["Scribe Output URL (on blob)"] = self._upload_action_summary(local_summary_path) or "Upload Failed/Skipped"

        # --- Tokens ---
        report["Input Tokens"] = self.token_usage.get("report_input_text_tokens", 0)
        report["Output Tokens"] = self.token_usage.get("report_output_text_tokens", 0)
        report["Image Tokens"] = self.token_usage.get("report_input_image_tokens", 0)
        report["Total Tokens"] = self.token_usage.get("report_total_tokens", 0)

        # --- Price ---
        cost = self._calculate_cost()
        report["Price ($)"] = f"{cost:.6f}" if cost is not None else "Calculation Failed (Check Pricing Info/Model Name)"
        # --- REMOVE CONFUSING NOTE --- 
        # The cost calculation already includes image tokens via Azure's total prompt_tokens 
        # if a specific image_per_million rate isn't provided.
        # The old note was added if image_per_million was 0 or missing, which was misleading.
        # if cost is not None and self.token_usage.get("report_input_image_tokens", 0) > 0:
        #      # Add note if image tokens were used but not priced
        #      if self.PRICING_INFO.get(self.env.vision.deployment, {}).get("image_per_million", 0) == 0:
        #           report["Price ($)"] += " (Excludes Image Tokens)"
        # --- END REMOVAL ---

        # --- Video Resolution ---
        video_resolution = "N/A"
        # Prioritize actionSummary's downscaledResolution as it reflects what was likely analyzed
        downscaled_res_as = action_summary.get("downscaledResolution")
        if downscaled_res_as and isinstance(downscaled_res_as, list) and len(downscaled_res_as) == 2 and all(isinstance(dim, int) and dim > 0 for dim in downscaled_res_as):
            video_resolution = f"{downscaled_res_as[0]}x{downscaled_res_as[1]} (analyzed)"
        elif self.manifest.source_video and self.manifest.source_video.size:
            size = self.manifest.source_video.size
            if len(size) == 2 and all(isinstance(s, int) and s > 0 for s in size):
                video_resolution = f"{size[0]}x{size[1]} (original)"
        elif self.manifest.processing_params and self.manifest.processing_params.downscaled_resolution:
            proc_down_res = self.manifest.processing_params.downscaled_resolution
            if isinstance(proc_down_res, list) and len(proc_down_res) == 2 and all(isinstance(dim, int) and dim > 0 for dim in proc_down_res):
                 video_resolution = f"{proc_down_res[0]}x{proc_down_res[1]} (downscaled)"
        report["Video Resolution"] = video_resolution

        print("Report data gathered.")
        return report

    # --- ADDED: Method to Write Summary Report ---
    def _write_summary_report(self, report_data: Dict):
        """Writes the gathered report data to a JSON file."""
        output_dir = self.manifest.processing_params.output_directory
        if not output_dir or not os.path.isdir(output_dir):
            print("Error: Output directory not specified or found. Cannot write report.")
            return

        report_filename = f"{os.path.splitext(self.manifest.name)[0]}_analysis_report.json"
        report_filepath = os.path.join(output_dir, report_filename)

        output_json = {
            "Filename": report_data.get("Filename", "N/A"),
            "CategoryDetails": {
                "Category": report_data.get("Category", "N/A"),
                "Subcategory": report_data.get("Subcategory", "N/A")
            },
            "Duration": report_data.get("Duration", "N/A"),
            "ModelUsed": report_data.get("Model used", "N/A"),
            "SchemaVersion": report_data.get("Schema Version", "N/A"),
            "TokenUsage": {
                "Input": report_data.get("Input Tokens", 0),
                "Output": report_data.get("Output Tokens", 0),
                "Image": report_data.get("Image Tokens", 0),
                "Total": report_data.get("Total Tokens", 0)
            },
            "Price": report_data.get("Price ($)", "N/A"),
            "VideoResolution": report_data.get("Video Resolution", "N/A"),
            "Links": {
                "InputVideoBlobURL": report_data.get("Blob URL for file", "N/A"),
                "ActionSummaryJsonBlobURL": report_data.get("Scribe Output URL (on blob)", "N/A")
            }
        }

        try:
            with open(report_filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(output_json, jsonfile, indent=4, ensure_ascii=False)
            print(f"Analysis summary report saved to: {report_filepath}")

        except Exception as e:
            print(f"Error writing summary report to {report_filepath}: {e}")

# --- END ADDED METHODS ---