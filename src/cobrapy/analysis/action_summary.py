from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar, Optional
from collections import defaultdict
import os
from datetime import datetime, timezone
from ..cobra_utils import seconds_to_iso8601_duration
import json

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
    "persons": [ {{ "name": "...", "timecodes": [{{"start": "...", "end": "..."}}] }} ],
    "objects": [ {{ "name": "...", "timecodes": [{{"start": "...", "end": "..."}}] }} ],
    "actions": [ {{ "name": "...", "timecodes": [{{"start": "...", "end": "..."}}] }} ]
  }}
}}
```

**CRITICAL TAGGING INSTRUCTIONS:**

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
        "objects": set(),
        "actions": set()
    }

    # -----------------------------------------------------------------------
    # MERGING / POST-PROCESSING LOGIC
    # -----------------------------------------------------------------------
    def process_segment_results(self, enriched_segment_results: List[Dict[str, Any]], manifest: "VideoManifest", env: "CobraEnvironment", parsed_copyright_info: Optional[Dict] = None, runtime_seconds: Optional[float] = None, tokens: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Process raw segment results, creates a detailed transcription output object,
        and merges tags/chapters into the actionSummary object.
        Leverages pre-merged intervals from segment analysis and performs a final merge pass.
        Returns a dictionary with 'transcriptionDetails' and 'actionSummary'.
        """
        process_start_time_utc = datetime.now(timezone.utc) # Record processing time
        transcription_details = {}
        action_summary_content = {}
        full_transcription_obj = None
        full_transcript_text = "Transcription data unavailable." # Default text

        # --- 1. Find and Process Full Transcription Object ---
        for container in enriched_segment_results:
            if (
                container
                and isinstance(container.get("fullTranscriptionObject"), dict)
                and "recognizedPhrases" in container["fullTranscriptionObject"]
            ):
                full_transcription_obj = container["fullTranscriptionObject"]
                print("DEBUG: Found Batch Transcription result object for detailed processing.")

        if full_transcription_obj and isinstance(full_transcription_obj.get("recognizedPhrases"), list):
            phrases = full_transcription_obj.get("recognizedPhrases", [])
            
            # --- 1a. Extract Root Transcription Details ---
            source_file = "Unknown"
            source_duration = None
            created_at = None
            if manifest and manifest.source_video:
                 source_path = manifest.source_video.path
                 source_duration = manifest.source_video.duration
                 source_file = os.path.basename(source_path) if source_path else "Unknown"
                 try:
                      if source_path and os.path.exists(source_path):
                           mod_time = os.path.getmtime(source_path)
                           created_at_dt = datetime.fromtimestamp(mod_time, timezone.utc)
                           created_at = created_at_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                 except Exception as e:
                      print(f"Warning: Could not get source created_at time: {e}")

            # Extract language (copied from previous logic)
            primary_language_code = None
            if "language" in full_transcription_obj: primary_language_code = full_transcription_obj.get("language")
            elif "properties" in full_transcription_obj:
                props = full_transcription_obj.get("properties", {})
                lang_id = (props.get("languageIdentification") or props.get("languageDetectionResult") or props.get("recognitionMetadata", {}).get("language"))
                if lang_id: primary_language_code = lang_id
            if not primary_language_code and phrases and isinstance(phrases, list) and len(phrases) > 0:
                 first_phrase = phrases[0];
                 if isinstance(first_phrase, dict) and first_phrase.get("language"): primary_language_code = first_phrase.get("language")
            if not primary_language_code or primary_language_code == "unk": primary_language_code = "en-US" # Fallback
            language_short_code = primary_language_code.split('-')[0] if primary_language_code else "en" # Use short code like 'en'

            transcript_id_base = os.path.splitext(source_file)[0] if source_file != "Unknown" else "transcript"
            transcript_id = f"{transcript_id_base}-{process_start_time_utc.strftime('%Y%m%d%H%M%S')}"

            total_confidence = 0
            valid_segments_count = 0
            processed_segments = []

            # Reconstruct full transcript text
            full_transcript_text = " ".join(
                phrase.get("nBest", [{}])[0].get("display", "")
                for phrase in phrases
                if phrase.get("nBest")
            ).strip()

            # --- 1b. Process Each Segment (Phrase) ---
            for idx, phrase in enumerate(phrases):
                if not isinstance(phrase, dict): continue
                try:
                    best_recognition = phrase.get("nBest", [{}])[0]
                    confidence = best_recognition.get("confidence")
                    if confidence is not None:
                        total_confidence += confidence
                        valid_segments_count += 1

                    offset_ticks = phrase.get("offsetInTicks", 0)
                    duration_ticks = phrase.get("durationInTicks", 0)
                    start_sec = offset_ticks / 10_000_000.0
                    end_sec = start_sec + (duration_ticks / 10_000_000.0)
                    duration_sec_segment = duration_ticks / 10_000_000.0

                    processed_segments.append({
                        "segmentId": idx + 1,
                        "speaker": f"Speaker_{phrase.get('speaker')}" if phrase.get('speaker') is not None else "Unknown Speaker",
                        "startSec": round(start_sec, 3),
                        "endSec": round(end_sec, 3),
                        "confidence": confidence,
                        "textVariants": {
                            "display": best_recognition.get("display"),
                            "lexical": best_recognition.get("lexical"),
                            "maskedITN": best_recognition.get("maskedITN")
                        },
                        "offsetInTicks": offset_ticks,
                        "durationInTicks": duration_ticks,
                        "duration": seconds_to_iso8601_duration(duration_sec_segment)
                    })
                except (ValueError, TypeError, IndexError, KeyError) as e:
                    print(f"WARNING: Skipping transcription phrase due to processing error: {e}. Data: {phrase}")

            avg_confidence = round(total_confidence / valid_segments_count, 3) if valid_segments_count > 0 else None

            # --- 1c. Assemble transcriptionDetails ---
            transcription_details = {
                "transcriptId": transcript_id,
                "language": language_short_code,
                "aiProcessing": {
                    "transcription": {
                        "createdBy": "transcriber-service-v2",
                        "model": f"Azure STT - Region: {env.speech.region if env and env.speech else 'N/A'}",
                        "confidenceAvg": avg_confidence,
                        "processedAt": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
                    }
                },
                "segments": processed_segments
            }
            print(f"DEBUG: Assembled transcriptionDetails with {len(processed_segments)} segments.")
        else:
            print("DEBUG: Batch Transcription result object not found or invalid. Cannot create detailed transcription output.")
            transcription_details = None

        # --- 2. Collect Pre-merged Tags and Chapters for actionSummary ---
        # Use defaultdict(list) to directly collect interval dicts
        collected_intervals_by_tag = {
            "persons": defaultdict(list),
            "objects": defaultdict(list),
            "actions": defaultdict(list)
        }
        chapters = []
        all_frame_paths = set()

        # Build a map for quick segment lookup by name (or start/end if name unreliable)
        segment_map = {seg.segment_name: seg for seg in manifest.segments if seg.segment_name}

        for i, result_container in enumerate(enriched_segment_results):
            if not isinstance(result_container, dict): continue
            analysis_result = result_container.get("analysisResult")
            segment_name = result_container.get("segmentName", f"segment_{i}")
            segment_start = result_container.get("startTime")
            segment_end = result_container.get("endTime")

            # Find the corresponding original segment for color data
            original_segment = segment_map.get(segment_name)
            # Fallback lookup if name isn't unique/present (less reliable)
            if not original_segment and segment_start is not None and segment_end is not None:
                for seg in manifest.segments:
                    # Use a small tolerance for float comparison
                    if abs(seg.start_time - segment_start) < 0.001 and abs(seg.end_time - segment_end) < 0.001:
                        original_segment = seg
                        break

            segment_colors = original_segment.dominant_colors_hex if original_segment and hasattr(original_segment, 'dominant_colors_hex') else []

            segment_frame_paths = result_container.get("framePaths", [])
            if isinstance(segment_frame_paths, list): all_frame_paths.update(segment_frame_paths)
            if analysis_result is None or segment_start is None or segment_end is None: continue
            if not isinstance(analysis_result, dict): continue

            # --- Process Chapters (validation logic remains) ---
            raw_chapters = analysis_result.get("chapters", [])
            if isinstance(raw_chapters, dict): raw_chapters = [raw_chapters]
            if isinstance(raw_chapters, list):
                 for chapter in raw_chapters:
                    if not isinstance(chapter, dict): continue
                    # --- ADDED: Ensure new shot keys exist, default to empty/null if not ---
                    if not isinstance(chapter.get("shotType"), list):
                        chapter["shotType"] = []
                    if "shotDescription" not in chapter: chapter["shotDescription"] = ""
                    # --- ADDED: Add dominant colors ---
                    chapter["dominantColorsHex"] = segment_colors if segment_colors else []
                    # --- END ADDED ---
                    is_chapter_valid = True
                    for time_field in ["start", "end"]:
                         original_time_str = chapter.get(time_field)
                         if original_time_str is None: continue
                         try:
                             time_str = str(original_time_str).rstrip('s'); time_val = float(time_str); original_val = time_val
                             # Clamp chapter times to segment boundaries if they exceed them
                             clamped_time_val = max(segment_start, min(segment_end, time_val))
                             # Only rewrite if clamping occurred OR format is wrong
                             if abs(clamped_time_val - original_val) > 0.0001 or not str(original_time_str).endswith('s') or len(time_str.split('.')[-1]) != 3:
                                 chapter[time_field] = f"{clamped_time_val:.3f}s"
                             else: # Preserve original formatting if valid and within bounds
                                 chapter[time_field] = f"{original_val:.3f}s"
                         except (ValueError, TypeError) as e: is_chapter_valid = False; print(f"WARN: Invalid chapter time {time_field}={original_time_str}"); break
                    if is_chapter_valid and "start" in chapter and "end" in chapter:
                         try:
                             start_f = float(str(chapter["start"]).rstrip('s')); end_f = float(str(chapter["end"]).rstrip('s'))
                             if start_f > end_f: chapter["start"] = chapter["end"] # Ensure start <= end
                         except (ValueError, TypeError): is_chapter_valid = False; print(f"WARN: Could not compare chapter start/end")
                    if is_chapter_valid: chapters.append(chapter)


            # --- Collect Pre-merged Tags ---
            global_tags_segment = analysis_result.get("globalTags", {})
            if not isinstance(global_tags_segment, dict): continue

            for tag_type in ["persons", "actions", "objects"]:
                 if tag_type not in global_tags_segment or not isinstance(global_tags_segment[tag_type], list): continue
                 for item in global_tags_segment[tag_type]:
                     if not isinstance(item, dict): continue
                     item_name = item.get("name"); timecodes = item.get("timecodes", []) # timecodes is List[Dict]
                     # Basic validation
                     if not item_name or not isinstance(item_name, str) or not item_name.strip() or not isinstance(timecodes, list): continue

                     tag_name_to_aggregate = item_name.strip()
                     if not tag_name_to_aggregate: continue

                     # Directly extend the list with the interval dicts
                     valid_timecode_dicts = [tc for tc in timecodes if isinstance(tc, dict) and "start" in tc and "end" in tc]
                     if valid_timecode_dicts:
                         collected_intervals_by_tag[tag_type][tag_name_to_aggregate].extend(valid_timecode_dicts)
                     # No need to handle floats here anymore

        # --- 3. Perform Final Merge Across All Segments ---
        final_tags_structure = {"persons": [], "actions": [], "objects": []}
        # Access the VideoAnalyzer instance's merge helper via self if ActionSummary is used within it
        # Or assume merge functions are available in scope
        from ..video_analyzer import VideoAnalyzer # Or adjust import as needed

        for tag_type, names_dict in collected_intervals_by_tag.items():
             for name, intervals_list in names_dict.items():
                 # `intervals_list` is now List[Dict[str, float]]
                 # Normalize to List[List[float]] for merging helper
                 numeric_intervals = VideoAnalyzer._timecodes_to_intervals(None, intervals_list) # Pass self=None if calling static helper
                 # Perform the final merge across all collected intervals for this tag name
                 merged_numeric_intervals = VideoAnalyzer._merge_overlapping(None, numeric_intervals, max_gap=2.0) # Use desired gap
                 # Convert back to list of dicts
                 final_timecode_dicts = [
                     {"start": round(s, 3), "end": round(e, 3)} for s, e in merged_numeric_intervals
                 ]
                 if final_timecode_dicts:
                      final_tags_structure[tag_type].append({
                          "name": name,
                          "timecodes": final_timecode_dicts
                      })

        # Assign the results from the final merge
        final_person_tags = final_tags_structure["persons"]
        final_action_tags = final_tags_structure["actions"]
        final_object_tags = final_tags_structure["objects"]

        # --- 4. Convert Timestamps to Numeric (Helper function handles 'start'/'end' keys) ---
        # This step remains useful for ensuring final output has numeric values if needed downstream
        chapters = convert_string_timestamps_to_numeric(chapters)
        final_person_tags = convert_string_timestamps_to_numeric(final_person_tags)
        final_action_tags = convert_string_timestamps_to_numeric(final_action_tags)
        final_object_tags = convert_string_timestamps_to_numeric(final_object_tags)

        # --- 5. Assemble actionSummary Content ---
        sorted_frame_paths = sorted(list(all_frame_paths))
        all_tag_names = set()
        for tag_list in [final_person_tags, final_action_tags, final_object_tags]:
            for tag_item in tag_list:
                if isinstance(tag_item, dict) and "name" in tag_item:
                    tag_name = tag_item["name"]
                    if isinstance(tag_name, str) and tag_name.strip(): all_tag_names.add(tag_name.strip())
        tags_list = sorted(list(all_tag_names))
        tags_string = ", ".join(tags_list)

        # --- Prepare top-level metadata ---
        top_level_meta = {} # Keep the dictionary for copyright and AI processing

        # --- Add Copyright Info ---
        if parsed_copyright_info:
            top_level_meta["copyright"] = parsed_copyright_info
            print("DEBUG: Added copyright info to actionSummary metadata.")

        # --- Prepare AI Processing block ---
        ai_processing = {
             "summarization": {"createdBy": f"llm-summary-{self.name}", "model": env.vision.deployment if env and env.vision else 'N/A', "processedAt": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')},
             "chapterGeneration": {"createdBy": f"llm-chapters-{self.name}", "model": env.vision.deployment if env and env.vision else 'N/A', "processedAt": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}
        }
        top_level_meta["aiProcessing"] = ai_processing

        # --- Assemble final actionSummary content dictionary ---
        action_summary_output = {
            **top_level_meta,
            "transcript": full_transcript_text,
            "tagsString": tags_string,
            "tags": tags_list,
            "chapter": chapters,
            "person": final_person_tags,
            "action": final_action_tags,
            "object": final_object_tags,
            "downscaledResolution": manifest.processing_params.downscaled_resolution if manifest and hasattr(manifest, 'processing_params') and hasattr(manifest.processing_params, 'downscaled_resolution') else None,
        }

        # Add runtime and tokens if provided (moved after dict construction)
        if runtime_seconds is not None:
            action_summary_output["runtime"] = seconds_to_iso8601_duration(runtime_seconds)
            action_summary_output["runtimeSeconds"] = round(runtime_seconds, 3)
        if tokens is not None:
            action_summary_output["tokens"] = tokens

        # --- 6. Assemble Final Output Dictionary ---
        final_output = {
            "actionSummary": action_summary_output
        }
        if transcription_details:
            final_output["transcriptionDetails"] = transcription_details

        print("DEBUG: process_segment_results finished. Returning dict with actionSummary and potentially transcriptionDetails.")
        return final_output

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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.video import VideoManifest
    from ..models.environment import CobraEnvironment