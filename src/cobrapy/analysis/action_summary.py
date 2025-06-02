from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar, Optional, TYPE_CHECKING
from collections import defaultdict
import os
from datetime import datetime, timezone
from ..cobra_utils import seconds_to_iso8601_duration
import json
import logging
from ..models.video import Segment

logger = logging.getLogger(__name__)

class ActionSummary(AnalysisConfig):
    """
    This AnalysisConfig enforces that the LLM must produce a JSON response with 
    top-level 'chapters' and 'globalTags' keys. The 'chapters' must have exactly
    one object describing the segment, and 'globalTags' must always include 
    'persons', 'actions', and 'objects' arrays, even if they are empty.
    
    The code below also includes a process_segment_results() method that merges
    any segment-level tags into continuous intervals at second-level accuracy.
    """

    name: str = "ActionSummary"
    analysis_sequence: str = "mapreduce"

    # --- ADDED: Configurable list of shot types ---
    SHOT_TYPES: ClassVar[List[str]] = [
    "Establishing Shot",
    "Extreme Wide Shot (EWS)",
    "Wide Shot (WS)",
    "Full Shot (FS)",
    "Medium Wide Shot (MWS)",
    "Medium Long Shot (MLS)",
    "Medium Shot (MS)",
    "Cowboy Shot",
    "Medium Close-Up (MCU)",
    "Close-Up (CU)",
    "Extreme Close-Up (ECU)",
    "Two-Shot",
    "Three-Shot",
    "Reverse Angle",
    "Over-the-Shoulder",
    "Point-of-View (POV) Shot",
    "Reaction Shot",
    "Insert Shot",
    "Cutaway",
    "Dutch Angle",
    "Tracking/Dolly Shot",
    "Crane/Jib Shot",
    "Handheld/Steadicam Shot",
    "Whip-Pan (Swish-Pan)",
    "Special / Other"
]

    # --- END ADDED ---

    # --- ADDED: Configurable list of Asset Categories ---
    ASSET_CATEGORIES: ClassVar[List[str]] = [
        "Sports",
        "Drama",
        "Comedy",
        "News",
        "Documentary",
        "Social Media",
        "Commercial/Advertisement",
        "Educational",
        "Music Video",
        "Gaming",
        "Lifestyle/Vlog",
        "Technology",
        "Travel",
        "Other"
    ]
    # --- END ADDED ---

    # -----------------------------------------------------------------------
    # System Prompt - Chapters Only (Revised to avoid segment-specific language)
    # -----------------------------------------------------------------------

    system_prompt_chapters: ClassVar[str] = (
        """You are VideoAnalyzerGPT, focused on describing video content within specific time ranges. Your goal is to generate a single, detailed description for the provided time range, using precise timestamps based *only* on the frame data provided.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    *   "chapters": An array containing EXACTLY ONE chapter object describing the content within the specified time range. **DO NOT include a "globalTags" key.**

2.  **SHOT ANALYSIS (NEW):** Within the chapter object, include:
    *   `shotType`: Classify the dominant camera shot types observed during this segment. Provide this as a **LIST of strings**. Choose **one or more** from the following list: {shot_types_list}. If the shot changes (e.g., zoom, dolly), include all relevant types observed.
    *   `shotDescription`: Describe the shot's composition (people, setting), visual style/grading (e.g., cinematic, noir, vibrant), and any noticeable camera movement (e.g., Dolly In/Out, Track In/Out, Zoom In/Out, Crane Up/Down).

3.  **EXACT OUTPUT FORMAT:** Use the *actual start and end times* for this segment (`{start_time}s` to `{end_time}s`) in the `start` and `end` fields of your JSON output.
    ```json
    {{
      "chapters": [
        {{
          "start": "{start_time}s",
          "end": "{end_time}s",
          "shotType": ["Medium Shot (MS)", "Close Up (CU)"],
          "shotDescription": "Starts as Medium Shot, zooms into Close Up on the character's reaction...",
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "theme": "short theme",
          "summary": "Detailed, descriptive summary of the video content..."
        }}
      ]
    }}
    ```

4.  **TIMESTAMP PRECISION & ACCURACY:**
    *   Use the absolute start ({start_time}s) and end ({end_time}s) times provided for the current time range.
    *   Format: "0.000s".

5.  **SUMMARY CONTENT:** Describe the setting, visuals, actions, and relevant audio context occurring *during this specific time range* ({start_time}s to {end_time}s). Focus entirely on *what is happening in the video content*. It is imperative that this information is accurate, detailed, and verbose. Interweave information from the images and transcription provided to give a full picture of the video during this time. **Avoid using phrases like "in this segment", "this clip", or referring to the analysis process itself.**
+ **IMPORTANT: Base your entire description *only* on the provided frame images and transcription text for this specific time range. DO NOT invent details, characters, settings, or events not directly observable in the input.**
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt - Tags Only (Unified - Handles Custom or Default Instructions)
    # -----------------------------------------------------------------------
    system_prompt_tags: ClassVar[str] =  (
    """You are VideoTaggerGPT, a specialist analysis agent. Your task is to identify and tag entities in video frames based on the provided criteria. It is imperative that you return all persons, objects, and actions in the given frames you are provided with as I have a disability that makes it impossible for me to view the video and only this JSON response.
Input (in the user message) will contain:
- segment_start, segment_end ("X.XXXs" format)
- frame_timestamps: A JSON map like {{ "1": "T1.XXXs", "2": "T2.XXXs", ... }}
- A sequence of Images referenced by ID (e.g., "Image #1", "Image #2").

Output MUST be **only** valid JSON matching this structure:
```json
{{
  "globalTags": {{
    "persons": [ {{ "name": "Concise label (e.g., 'Man in red shirt', 3-5 words max)", "timecodes": [{{"start": "...", "end": "..."}}] }} ],
    "objects": [ {{ "name": "Concise label (e.g., 'Red car on street', 3-5 words max)", "timecodes": [{{"start": "...", "end": "..."}}] }} ],
    "actions": [ {{ "name": "Concise action (e.g., 'Person walking', 'Car driving', 3-5 words max)", "timecodes": [{{"start": "...", "end": "..."}}] }} ]
  }}
}}
```

**CRITICAL TAGGING INSTRUCTIONS:**

**General Naming Convention:**
- All 'name' attributes for persons, objects, and actions MUST be very concise (3-5 words maximum), like a short descriptive label.
- For 'persons', focus on key visual characteristics (e.g., 'Woman in blue dress', 'Man with glasses and hat').
- For 'objects', use a short descriptive name (e.g., 'Red sports car', 'Wooden table with laptop').
- For 'actions', use a brief verb phrase describing the activity (e.g., 'Person opening door', 'Dog running in park').
- **Avoid full sentences or paragraphs for the 'name' attribute.**

**Persons:**
{person_instructions}

**Objects:**
{object_instructions}

**Actions:**
{action_instructions}

 **CRITICAL TAGGING INSTRUCTIONS:**
    …
    Example: If 'person A' is first seen in Image #2 (timestamp 3.500s) and last seen in Image #5 (timestamp 6.500s), the timecode is `{{"start": "3.500s", "end": "6.500s"}}`. If they reappear in Image #8 (9.500s), create a new entry.

**The following rules are critical to the success of the job:**
**Timestamp Rules:**
- Use the provided `frame_timestamps` map to determine `start`/`end` times (format "X.XXXs"). The `start` time is the timestamp of the **first frame** the item is clearly visible in the provided images, and the `end` time is the timestamp of the **last frame** it is clearly visible.
- **CRITICAL: DO NOT use the overall `segment_start` and `segment_end` values for the tag timecodes.** Use ONLY the timestamps corresponding to the frames where the entity is visible.
- If an item appears in consecutive frames, merge into one timecode interval spanning its appearance.
- If an item disappears and reappears later within the provided frames, create separate timecode entries for each appearance.
- All timecodes must be strictly within the range covered by the provided `frame_timestamps`.

FINAL CHECK: Ensure your output is **only** the valid JSON structure specified above, with `start` and `end` times accurately reflecting the **frame timestamps** where entities are visible. Use empty arrays if no items are identified for a category.
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt Lens - Chapters
    # -----------------------------------------------------------------------
    system_prompt_lens_chapters:  ClassVar[str] = (
        """Analyze the segment from {start_time}s to {end_time}s to generate a chapter summary. The segment contains:
- {number_of_frames} frames (with timestamps)
- Possibly an audio transcription: {transcription_context}

Remember, your output MUST be valid JSON containing ONLY the 'chapters' key, with exactly one chapter object inside the array, describing this segment.
"""
    )

    # -----------------------------------------------------------------------
    # Example Structures (Split)
    # -----------------------------------------------------------------------
    results_template_chapters: Dict[str, Any] = {
        "chapters": [
            {
                "start": "0.000s",
                "end": "30.000s",
                "shotType": [],
                "shotDescription": "",
                "sentiment": "neutral",
                "emotions": [],
                "theme": "",
                "summary": ""
            }
        ]
    }

    results_template_tags: Dict[str, Any] = {
        "globalTags": {
            "persons": [],
            "actions": [],
            "objects": []
        }
    }

    # Combine templates for code that might still reference the old one (e.g. sequential analysis)
    results_template: Dict[str, Any] = {
        **results_template_chapters,  # type: ignore
        **results_template_tags  # type: ignore
    }

    # Whether we run a final summary across all segments after everything finishes
    run_final_summary: bool = True

    # The prompt used for the final summary across the entire video
    summary_prompt:  ClassVar[str] = (
        """Analyze the complete video analysis results (chapters and tags) provided below to understand the video's content.
        Your task is to generate TWO summaries and return them ONLY as a single JSON object.

        Additionally, classify the overall video content:
        *   `category`: Choose the **single most appropriate** category for the entire video from this list: {asset_categories_list}
        *   `subCategory`: (Optional) If applicable and obvious, provide a specific sub-category (e.g., for 'Sports', maybe 'Basketball Highlights'; for 'News', maybe 'Political Report'). **DO NOT invent a sub-category if one isn't clear.** Leave it as `null` or omit the key if unsure.

        1.  **description**: A very concise, 1-2 sentence description summarizing the absolute core gist or main topic of the video content. Focus on what the video is primarily *about*. Avoid listing specific details unless essential for the core topic.
        2.  **summary**: A detailed summary that captures the key narrative, main participants, significant actions, and important objects across all segments. Focus on the overall flow and major themes rather than segment-by-segment details. **The summary should be {summary_length_instruction}**

        CRITICAL: You MUST return ONLY a valid JSON object containing `description`, `summary`, `category`, and optionally `subCategory`. Example format:
        ```json
        {{
          "category": "Sports",
          "subCategory": "Basketball Highlights",
          "description": "A 1-2 sentence description...",
          "summary": "The detailed summary..."
        }}
        ```
        Do not include any text outside of this JSON structure."""
    )

    # Track known tags at the class level for FINAL aggregation if needed
    # We will use instance-level tracking in VideoAnalyzer for prompting
    known_tags: ClassVar[Dict[str, set]] = {
        "persons": set(),
        "actions": set(),
        "objects": set(),
    }

    # Helper to initialize tags structure if not present
    def _ensure_tags_structure(self, result_dict: Dict[str, Any]) -> None:
        if "globalTags" not in result_dict:
            result_dict["globalTags"] = {}
        if "persons" not in result_dict["globalTags"]:
            result_dict["globalTags"]["persons"] = []
        if "actions" not in result_dict["globalTags"]:
            result_dict["globalTags"]["actions"] = []
        if "objects" not in result_dict["globalTags"]:
            result_dict["globalTags"]["objects"] = []

    def process_segment_results(self, enriched_segment_results: List[Dict[str, Any]], manifest: "VideoManifest", env: "CobraEnvironment", parsed_copyright_info: Optional[Dict] = None, runtime_seconds: Optional[float] = None, tokens: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Processes results from multiple segments, merging chapters and aggregating globalTags.
        Now includes dedicated handling for 'extracted_actions' from segments.
        """
        final_results: Dict[str, Any] = {
            "chapters": [],
            "globalTags": {
                "persons": [],
                "actions": [],
                "objects": [],
                # Potentially other global tag categories if defined
            },
            "processing_info": {
                "processed_segments": len(enriched_segment_results),
                "runtime_seconds": runtime_seconds,
                "tokens_used": tokens,
                "copyright_info_applied": bool(parsed_copyright_info)
            }
        }
        
        # Hold all tags before aggregation to handle potential duplicates from different sources/segments
        all_persons_intermediate = []
        all_objects_intermediate = []
        all_actions_intermediate = [] # For merging tags from LLM and dedicated extraction

        # Temporary sets to track unique tag names encountered during segment processing
        # This helps in adding to manifest.global_tags only unique names
        current_run_persons = set(manifest.global_tags.get("persons", []))
        current_run_actions = set(manifest.global_tags.get("actions", []))
        current_run_objects = set(manifest.global_tags.get("objects", []))

        for segment_result_envelope in enriched_segment_results:
            # Get the LLM output using the correct key
            segment_llm_output = segment_result_envelope.get("analysisResult") # CORRECTED KEY
            
            # Get the segment object
            segment_obj: Optional["Segment"] = segment_result_envelope.get("segment_object") # NEW - requires change in video_analyzer.py
            segment_name_for_log = segment_result_envelope.get("segmentName", "Unknown") # Fallback if segment_obj is not there yet

            if not segment_llm_output:
                logger.warning(f"Segment {segment_name_for_log} had no LLM output (analysisResult) to process.")
                continue
            
            # Ensure the basic structure exists in the LLM output
            self._ensure_tags_structure(segment_llm_output)

            # Process chapters
            if "chapters" in segment_llm_output and isinstance(segment_llm_output["chapters"], list):
                for chapter in segment_llm_output["chapters"]:
                    try:
                        # Attempt to convert "start" and "end" times to float if they are strings
                        if isinstance(chapter.get("start"), str):
                            chapter["start"] = float(chapter["start"].replace("s", ""))
                        if isinstance(chapter.get("end"), str):
                            chapter["end"] = float(chapter["end"].replace("s", ""))
                        
                        # Add dominant colors if available from segment object
                        if segment_obj and segment_obj.dominant_colors_hex: # Check if segment_obj is not None
                            chapter["dominantColorsHex"] = segment_obj.dominant_colors_hex

                        final_results["chapters"].append(chapter)
                    except ValueError as e:
                        logger.error(f"Could not convert chapter start/end time to float for segment {segment_name_for_log}: {chapter}. Error: {e}")
                    except Exception as e:
                        logger.error(f"Error processing chapter for segment {segment_name_for_log}: {chapter}. Error: {e}")
            
            # Process globalTags from LLM
            segment_tags = segment_llm_output.get("globalTags", {})
            if segment_tags: # Check if segment_tags is not None
                for tag_type in ["persons", "objects", "actions"]:
                    if tag_type in segment_tags and isinstance(segment_tags[tag_type], list):
                        for tag_entry in segment_tags[tag_type]:
                            if isinstance(tag_entry, dict) and "name" in tag_entry and "timecodes" in tag_entry:
                                if tag_type == "persons":
                                    all_persons_intermediate.append(tag_entry)
                                    current_run_persons.add(tag_entry["name"])
                                elif tag_type == "objects":
                                    all_objects_intermediate.append(tag_entry)
                                    current_run_objects.add(tag_entry["name"])
                                elif tag_type == "actions": # Actions from main LLM tagging
                                    all_actions_intermediate.append(tag_entry)
                                    current_run_actions.add(tag_entry["name"])
                            else:
                                logger.warning(f"Malformed tag entry in segment_tags for type {tag_type} in segment {segment_name_for_log}: {tag_entry}")
            
            # Process 'extracted_actions' from the segment object (new granular extraction)
            if segment_obj and hasattr(segment_obj, 'extracted_actions') and segment_obj.extracted_actions: # Check if segment_obj is not None
                logger.debug(f"Processing {len(segment_obj.extracted_actions)} extracted_actions for segment {segment_obj.segment_name}")
                for action_entry in segment_obj.extracted_actions:
                    if isinstance(action_entry, dict) and "name" in action_entry and "timecodes" in action_entry:
                        all_actions_intermediate.append(action_entry) # Add to the same intermediate list
                        current_run_actions.add(action_entry["name"])
                    else:
                        logger.warning(f"Malformed extracted_action entry for segment {segment_obj.segment_name}: {action_entry}")

        # Sort chapters by start time
        final_results["chapters"].sort(key=lambda x: x.get("start", float('inf')))

        # Aggregate and merge timecodes for all tag types
        # The _aggregate_tag_timestamps method should handle string timecodes like "0.000s"
        # and convert them internally or expect them to be numeric.
        # Let's ensure they are strings as per its original design.
        
        def convert_timecodes_to_str_format(tag_list):
            for tag in tag_list:
                if "timecodes" in tag and isinstance(tag["timecodes"], list):
                    new_timecodes = []
                    for tc in tag["timecodes"]:
                        try:
                            start_val = tc['start']
                            end_val = tc['end']
                            # Ensure they are strings ending with 's'
                            start_str = f"{float(str(start_val).replace('s','')):.3f}s" if isinstance(start_val, (str, float, int)) else start_val
                            end_str = f"{float(str(end_val).replace('s','')):.3f}s" if isinstance(end_val, (str, float, int)) else end_val
                            new_timecodes.append({"start": start_str, "end": end_str})
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Skipping malformed timecode {tc} during string conversion: {e}")
                    tag["timecodes"] = new_timecodes
            return tag_list

        all_persons_intermediate = convert_timecodes_to_str_format(all_persons_intermediate)
        all_objects_intermediate = convert_timecodes_to_str_format(all_objects_intermediate)
        all_actions_intermediate = convert_timecodes_to_str_format(all_actions_intermediate)

        aggregated_persons = self._aggregate_tag_timestamps({"persons": all_persons_intermediate})
        aggregated_objects = self._aggregate_tag_timestamps({"objects": all_objects_intermediate})
        aggregated_actions = self._aggregate_tag_timestamps({"actions": all_actions_intermediate}) # Now includes dedicated extracted actions

        final_results["globalTags"]["persons"] = aggregated_persons.get("persons", [])
        final_results["globalTags"]["objects"] = aggregated_objects.get("objects", [])
        final_results["globalTags"]["actions"] = aggregated_actions.get("actions", [])


        # Update manifest's global_tags with any newly discovered tag names (unique)
        # These are just the names, not the timecodes, used for subsequent prompting if needed.
        if not manifest.global_tags: # Initialize if it's None
            manifest.global_tags = {"persons": [], "actions": [], "objects": []}
        
        # Ensure each key exists before trying to extend
        for key in ["persons", "actions", "objects"]:
            if key not in manifest.global_tags:
                manifest.global_tags[key] = []

        # Add new unique names to the manifest's global_tags lists
        # These are used for providing known entities to the LLM in subsequent calls.
        # The `current_run_persons/actions/objects` sets now contain all unique names from this run.
        
        # Example of updating manifest.global_tags:
        # For persons:
        # existing_manifest_persons = set(manifest.global_tags.get("persons", []))
        # new_persons_to_add = list(current_run_persons - existing_manifest_persons)
        # if new_persons_to_add:
        # manifest.global_tags["persons"].extend(new_persons_to_add)
        # manifest.global_tags["persons"] = sorted(list(set(manifest.global_tags["persons"]))) # Keep unique and sorted

        # Consolidate unique tag names for the manifest (not the final_results JSON)
        # The manifest.global_tags are simple lists of names.
        manifest.global_tags["persons"] = sorted(list(current_run_persons))
        manifest.global_tags["actions"] = sorted(list(current_run_actions))
        manifest.global_tags["objects"] = sorted(list(current_run_objects))

        logger.info(f"Final aggregated persons: {len(final_results['globalTags']['persons'])}")
        logger.info(f"Final aggregated objects: {len(final_results['globalTags']['objects'])}")
        logger.info(f"Final aggregated actions: {len(final_results['globalTags']['actions'])}") # Log count of aggregated actions

        # Update the class-level known_tags for potential use in other parts of the system or future runs,
        # though instance-level VideoAnalyzer._current_known_persons etc. are primary for per-segment prompting
        ActionSummary.known_tags["persons"].update(current_run_persons)
        ActionSummary.known_tags["actions"].update(current_run_actions)
        ActionSummary.known_tags["objects"].update(current_run_objects)

        return final_results

    @staticmethod
    def _timecodes_to_intervals_action_summary(timecodes: List[Any]) -> List[List[float]]:
        """
        Normalises various timecode formats into [start, end] float pairs.
        Accepts:
            - {"start": "...s","end":"...s"} dicts
            - {"start": float/int,"end": float/int} dicts
            - plain floats / ints (treated as single frame, extended by 0.001s)
            - "12.345s" strings (treated as single frame, extended by 0.001s)
        Returns a list of [start, end] pairs (floats, 3 dp), sorted by start time.
        Invalid formats are logged and skipped.
        """
        intervals = []
        for tc in timecodes:
            try:
                if isinstance(tc, dict) and "start" in tc and "end" in tc:
                    start_val = tc["start"]
                    end_val = tc["end"]

                    if isinstance(start_val, str): s = float(start_val.replace("s", ""))
                    elif isinstance(start_val, (int, float)): s = float(start_val)
                    else: raise ValueError("Invalid start time format")

                    if isinstance(end_val, str): e = float(end_val.replace("s", ""))
                    elif isinstance(end_val, (int, float)): e = float(end_val)
                    else: raise ValueError("Invalid end time format")

                    if s > e: # Swap if start is after end
                        logger.warning(f"Correcting inverted timecode: start ({s}) > end ({e}). Swapping.")
                        s, e = e, s
                    intervals.append([round(s, 3), round(e, 3)])

                elif isinstance(tc, (int, float)):
                    s = round(float(tc), 3)
                    intervals.append([s, s + 0.001]) # Represent single point as 1ms interval
                elif isinstance(tc, str):
                    s = round(float(tc.replace("s", "")), 3)
                    intervals.append([s, s + 0.001]) # Represent single point as 1ms interval
                else:
                    logger.warning(f"Unsupported timecode format skipped: {tc}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error processing timecode {tc}: {e}. Skipping.")
                continue
        return sorted(intervals, key=lambda x: x[0])

    @staticmethod
    def _merge_overlapping_intervals_action_summary(intervals: List[List[float]], max_gap: float = 0.5) -> List[List[float]]:
        """
        Merges touching, overlapping, or nearly-touching time intervals.
        max_gap: Maximum gap in seconds to bridge between intervals.
        Assumes intervals are already sorted by start time.
        """
        if not intervals:
            return []
        
        merged = []
        current_start, current_end = intervals[0]

        for i in range(1, len(intervals)):
            next_start, next_end = intervals[i]
            # If the next interval starts before or slightly after the current one ends (within max_gap)
            if next_start <= current_end + max_gap:
                current_end = max(current_end, next_end) # Extend the current interval
            else:
                merged.append([current_start, current_end]) # Finalize current interval
                current_start, current_end = next_start, next_end # Start a new one
        
        merged.append([current_start, current_end]) # Add the last processed interval
        return merged

    def _aggregate_tag_timestamps(self, tag_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Aggregates and merges timecodes for each tag type.
        For 'persons', groups by 'id', merges timecodes, and uses the first encountered description and metadata.
        For 'objects' and 'actions', groups by 'name' and merges timecodes.
        """
        aggregated_results: Dict[str, List[Dict[str, Any]]] = {}

        for tag_type, entries in tag_dict.items():
            if not entries:
                aggregated_results[tag_type] = []
                continue

            if tag_type == "persons":
                # Group persons by 'id'
                persons_by_id: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
                for entry in entries:
                    # Ensure 'id' exists, otherwise skip or assign a default
                    person_id = entry.get("id")
                    if person_id is None:
                        logger.warning(f"Person entry missing 'id', cannot aggregate: {entry.get('name', 'Unknown')}")
                        continue
                    persons_by_id[person_id].append(entry)
                
                processed_persons = []
                for person_id, person_entries in persons_by_id.items():
                    if not person_entries:
                        continue
                    
                    # Use metadata from the first entry for this ID
                    representative_entry = person_entries[0]
                    all_timecodes_for_id = []
                    for p_entry in person_entries:
                        all_timecodes_for_id.extend(p_entry.get("timecodes", []))
                    
                    numerical_intervals = self._timecodes_to_intervals_action_summary(all_timecodes_for_id)
                    merged_intervals_numeric = self._merge_overlapping_intervals_action_summary(numerical_intervals)
                    
                    final_timecodes_str_format = [
                        {"start": f"{s:.3f}s", "end": f"{e:.3f}s"} for s, e in merged_intervals_numeric
                    ]
                    
                    # Construct the final person entry
                    # Preserve relevant fields like yoloClass, thumb, original/refined IDs
                    final_person = {
                        "name": representative_entry.get("name"),
                        "id": person_id, # This is the grouping key
                        "yoloClass": representative_entry.get("yoloClass", "person"), # Default to person
                        "timecodes": final_timecodes_str_format
                    }
                    # Carry over thumb and tracking IDs if present in the representative entry
                    for key_to_carry in ["thumb", "representative_thumb_path", "refined_track_id_str", "original_yolo_ids_ref", "original_yolo_id_ref"]:
                        if key_to_carry in representative_entry and representative_entry[key_to_carry] is not None:
                            final_person[key_to_carry] = representative_entry[key_to_carry]
                    
                    processed_persons.append(final_person)
                aggregated_results[tag_type] = processed_persons
            
            else: # For 'objects' and 'actions', group by 'name'
                tags_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for entry in entries:
                    name = entry.get("name")
                    if not name:
                        logger.warning(f"Tag entry of type '{tag_type}' missing 'name': {entry}")
                        continue
                    tags_by_name[name].append(entry)

                processed_tags = []
                for name, tag_entries_for_name in tags_by_name.items():
                    if not tag_entries_for_name:
                        continue
                    
                    # Use metadata from the first entry for this name (e.g. yoloClass for objects)
                    representative_entry = tag_entries_for_name[0]
                    all_timecodes_for_name = []
                    for t_entry in tag_entries_for_name:
                        all_timecodes_for_name.extend(t_entry.get("timecodes", []))
                        
                    numerical_intervals = self._timecodes_to_intervals_action_summary(all_timecodes_for_name)
                    merged_intervals_numeric = self._merge_overlapping_intervals_action_summary(numerical_intervals)
                    
                    final_timecodes_str_format = [
                        {"start": f"{s:.3f}s", "end": f"{e:.3f}s"} for s, e in merged_intervals_numeric
                    ]
                    
                    final_tag_entry = {
                        "name": name, # This is the grouping key
                        "timecodes": final_timecodes_str_format
                    }
                    # For objects, carry over yoloClass, id, thumb, and tracking IDs if present
                    if tag_type == "objects":
                        final_tag_entry["yoloClass"] = representative_entry.get("yoloClass")
                        final_tag_entry["id"] = representative_entry.get("id") # Object ID might not be as persistent as person ID
                        for key_to_carry in ["thumb", "representative_thumb_path", "refined_track_id_str", "original_yolo_ids_ref", "original_yolo_id_ref"]:
                             if key_to_carry in representative_entry and representative_entry[key_to_carry] is not None:
                                final_tag_entry[key_to_carry] = representative_entry[key_to_carry]
                                
                    processed_tags.append(final_tag_entry)
                aggregated_results[tag_type] = processed_tags
                
        return aggregated_results

# --- UPDATED Helper Function for Robust Timestamp Conversion (Keep existing logic) ---
# Note: This will apply to chapters and tags within actionSummary
def convert_string_timestamps_to_numeric(data):
    """Recursively converts 'start'/'end' string values like 'X.XXXs' to numeric floats.
    If a timecode dictionary is missing a key, logs a warning and sets the missing key to None."""
    if isinstance(data, dict):
        is_timecode_dict = 'start' in data or 'end' in data; new_dict = {}
        for k, v in data.items():
            if is_timecode_dict and k in ("start", "end"):
                if isinstance(v, str) and v.endswith('s'):
                    try: new_dict[k] = round(float(v[:-1]), 3)
                    except (ValueError, TypeError): new_dict[k] = v
                elif isinstance(v, (int, float)): new_dict[k] = round(float(v), 3)
                else: new_dict[k] = v
            # --- ADDED: Skip conversion for new shot fields ---
            elif k in ("shotType", "shotDescription"):
                 new_dict[k] = v # Keep as string
            # --- END ADDED ---
            else: new_dict[k] = convert_string_timestamps_to_numeric(v)
        if is_timecode_dict:
            if 'start' not in new_dict: new_dict['start'] = None
            if 'end' not in new_dict: new_dict['end'] = None
        return new_dict
    elif isinstance(data, list):
        processed_list = [convert_string_timestamps_to_numeric(item) for item in data]; return [item for item in processed_list if item is not None]
    else: return data
# --- End UPDATED Helper Function ---

# Add explicit type hint for VideoManifest to satisfy the forward reference
if TYPE_CHECKING:
    from ..models.video import VideoManifest
    from ..models.environment import CobraEnvironment