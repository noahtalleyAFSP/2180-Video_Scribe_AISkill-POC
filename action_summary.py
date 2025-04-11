from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar, Optional
from collections import defaultdict
import os
from datetime import datetime, timezone
from ..cobra_utils import seconds_to_iso8601_duration

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

    # -----------------------------------------------------------------------
    # System Prompt - Chapters Only (Corrected - No personNames/peoples)
    # -----------------------------------------------------------------------
    system_prompt_chapters: str = (
        """You are VideoAnalyzerGPT, focused on summarizing video segments. Your goal is to generate a single, detailed chapter summary for the provided segment, using precise timestamps based *only* on the frame data provided.

**CRITICAL INSTRUCTIONS - CHAPTERS ONLY - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    *   "chapters": An array containing EXACTLY ONE chapter object describing the current segment. **DO NOT include a "globalTags" key.**

2.  **EXACT OUTPUT FORMAT (Chapters Only):**
    ```json
    {{
      "chapters": [
        {{
          "start": "45.123s",
          "end": "75.456s",
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "theme": "short theme",
          "summary": "Detailed, descriptive summary..."
        }}
      ]
    }}
    ```

3.  **TIMESTAMP PRECISION & ACCURACY (Chapters):**
    *   Use segment absolute start ({start_time}s) and end ({end_time}s) times.
    *   Format: "0.000s".

4.  **SUMMARY CONTENT:** Describe setting, visuals, actions, audio context for *this specific segment*.
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt - Tags Only (Standard - No Custom People List)
    # -----------------------------------------------------------------------
    system_prompt_tags: str = (
        """You are VideoAnalyzerGPT, specialized in identifying and tagging entities in video frames. Your task has THREE DISTINCT PARTS that MUST ALL be completed:

1) Identify and tag PERSONS using descriptive names based on appearance.
2) Identify and tag relevant OBJECTS visible in the frames.
3) Identify and tag significant ACTIONS being performed.

**IMPORTANT: You MUST complete ALL THREE parts of this task!**

**PERSONS TAGGING TASK**
- Identify clearly visible individuals.
- **Naming Rule:** The 'name' MUST describe the person's **appearance and static features ONLY**. DO NOT include actions (walking, sitting) in the person's name. Tag actions separately.
- Use descriptive, consistent identifiers based *only* on appearance (e.g., "Woman in red dress", "Man in grey suit", "Child with blue backpack"). Do NOT add arbitrary IDs.

**OBJECTS TAGGING TASK**
- Identify relevant physical items (e.g., "Laptop", "Desk", "Window", "Briefcase", "Chair").
- **Consistency is Key:** Use specific, consistent names. Check 'Known Objects' list below and reuse names where appropriate. Avoid minor variations for the same object (e.g., prefer "Telephone" over "Phone", "Handset").

**ACTIONS TAGGING TASK**
- Identify significant activities (e.g., "Walking", "Sitting", "Typing", "Talking", "Holding object").
- **Consistency is Key:** Use clear, concise action verbs. Check 'Known Actions' list below and reuse names where appropriate. **Crucially, avoid creating synonyms for the same core action.** For example, if someone is interacting with a telephone, prefer a single consistent tag like "Using telephone" instead of multiple variations like "Holding telephone", "Talking on phone", "Picking up telephone" unless the specific phase of the action is critical and distinct. Choose the most representative and consistent term.

**TIMESTAMP REQUIREMENTS:**
- Base ALL timecodes (`start`, `end`) **STRICTLY** on the absolute timestamps of the {number_of_frames} frames provided for THIS chunk ({start_time}s to {end_time}s).
- Timecodes must represent the interval the item/person/action is *visibly present* or *actively occurring* AS OBSERVED IN THE **PROVIDED FRAMES**.
- Do NOT automatically extend to chunk boundaries unless the frames clearly show continuous presence.
- Format: "0.000s".

**OUTPUT FORMAT:**
    ```json
    {{
      "globalTags": {{
        "persons": [ /* {{"name": "Descriptive Name", "timecodes": [{{"start": "X.XXXs", "end": "Y.YYYs"}}]}} */ ],
        "objects": [ /* {{"name": "Object Name", "timecodes": [{{"start": "X.XXXs", "end": "Y.YYYs"}}]}} */ ],
        "actions": [ /* {{"name": "Action Name", "timecodes": [{{"start": "X.XXXs", "end": "Y.YYYs"}}]}} */ ]
      }}
    }}
    ```
**Important**: For the custom persons list, it is important that if it is determined that any of the names are within the image, call each and every single one out in your response. 

**CRITICAL:** All three tag categories MUST be populated.** If nothing is found for a category, include an empty array `[]`. Ensure timestamp accuracy based ONLY on provided frames for this chunk. Prioritize consistent naming for objects and actions.
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt - Tags Only (Custom People Focus - Stronger Enforcement)
    # -----------------------------------------------------------------------
    system_prompt_tags_custom_people: str = (
        """You are VideoAnalyzerGPT, a specialist analysis agent. Your task is to identify and tag entities in video frames based on clear, generic definitions. You MUST complete THREE distinct tasks:

1) **PERSONS TAGGING (CUSTOM CRITERIA):** Identify and label persons strictly using one of the following options. The allowed person labels with descriptions are provided below. Use only the exact labels as provided. Do not invent any additional labels.
--- Allowed Person Options ---
{people_definitions}
--- End Allowed Person Options ---
  - You can have up to four persons responses per image, as that is how many options there are. Thanks! Make sure to tag every single description for each person that matches the rules above. One image can have many tags. 

2) **OBJECTS TAGGING:** Identify and tag physically discernible items in the frames. Provide descriptions of each object using consistent and descriptive terminology. For example, "Laptop", "Desk", "Window", "Briefcase", "Chair". These are examples; include others if clearly present.

3) **ACTIONS TAGGING:** Identify and tag noticeable activities or behaviors occurring in the frames. Use brief, generic action verbs (e.g., "moving", "interacting") and ensure the naming remains consistent across frames. If custom action definitions are provided, follow them; otherwise, describe actions in general terms.

TIMESTAMP REQUIREMENTS:
- Base ALL timecodes ("start", "end") strictly on the absolute timestamps of the {number_of_frames} frames provided for this chunk (from {start_time}s to {end_time}s).
- Format timecodes as "X.XXXs".

OUTPUT FORMAT:
```json
{{
  "globalTags": {{
    "persons": [ {{"name": "An option from the list of allowed person options", "timecodes": [...] }} ],
    "objects": [ {{"name": "Object Description", "timecodes": [...] }} ],
    "actions": [ {{"name": "Action Description", "timecodes": [...] }} ]
  }}
}}
```
Ensure you find all the people, objects, and actions in the video and tag them accordingly. It is imperative that the descriptions of each tag are entirely accurate as I have a disability that makes it impossible for me to view the video and only this JSON response.
FINAL CHECK: Ensure that the "persons" array contains only entries with names from the allowed options above. For objects and actions, the names should be generally descriptive and consistent, without relying on domain-specific terminology. Use empty arrays if no items are identified.
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt Lens - Chapters
    # -----------------------------------------------------------------------
    system_prompt_lens_chapters: str = (
        """Analyze the segment from {start_time}s to {end_time}s to generate a chapter summary. The segment contains:
- {number_of_frames} frames (with timestamps)
- Possibly an audio transcription: {transcription_context}

Remember, your output MUST be valid JSON containing ONLY the 'chapters' key, with exactly one chapter object inside the array, describing this segment.
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt Lens - Tags (Updated to handle default reminders)
    # -----------------------------------------------------------------------
    system_prompt_lens_tags: str = (
        """Analyze the time chunk from {start_time}s to {end_time}s to identify tags. The chunk contains:
- {number_of_frames} frames (with timestamps)
- Overall Segment Transcription Context: {transcription_context}

{explicit_object_reminder}
{explicit_action_reminder}

Remember, your output MUST be valid JSON containing ONLY the 'globalTags' key, with 'persons', 'actions', and 'objects' arrays (populate all three!). Ensure all tag timecodes are strictly within {start_time}s and {end_time}s and based ONLY on the provided frames.

--- Custom Definitions & Instructions ---
**Persons:**
{people_definitions}

**Objects:**
{object_definitions}

**Actions:**
{action_definitions}
--- End Custom Definitions ---

Known Tags (Reuse these names if applicable for Actions and Objects NOT covered by custom definitions):
- Known Actions: {known_actions}
- Known Objects: {known_objects}
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
    summary_prompt: str = (
        """Analyze the complete video analysis results (chapters and tags) provided below.
        Your task is to generate TWO summaries and return them ONLY as a single JSON object.

        1.  **description**: A very concise, 1-2 sentence description summarizing the absolute core gist or main topic of the video content. Focus on what the video is primarily *about*. Avoid listing specific details unless essential for the core topic.
        2.  **summary**: A detailed, concise summary that captures the key narrative, main participants, significant actions, and important objects across all segments. Focus on the overall flow and major themes rather than segment-by-segment details.

        CRITICAL: You MUST return ONLY a valid JSON object with exactly two keys: "description" and "summary". Example format:
        ```json
        {
          "description": "A 1-2 sentence description...",
          "summary": "The detailed summary..."
        }
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
    def process_segment_results(self, enriched_segment_results: List[Dict[str, Any]], manifest: "VideoManifest", env: "CobraEnvironment", parsed_copyright_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process raw segment results, creates a detailed transcription output object,
        and merges tags/chapters into the actionSummary object.
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
                break

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

        # --- 2. Process Tags and Chapters for actionSummary ---
        tag_timestamps = {
            "persons": defaultdict(list),
            "objects": defaultdict(list),
            "actions": defaultdict(list)
        }
        chapters = []
        all_frame_paths = set()
        for i, result_container in enumerate(enriched_segment_results):
            if not isinstance(result_container, dict): continue
            analysis_result = result_container.get("analysisResult")
            segment_name = result_container.get("segmentName", f"segment_{i}")
            segment_start = result_container.get("startTime")
            segment_end = result_container.get("endTime")
            segment_frame_paths = result_container.get("framePaths", [])
            if isinstance(segment_frame_paths, list): all_frame_paths.update(segment_frame_paths)
            if analysis_result is None or segment_start is None or segment_end is None: continue
            if not isinstance(analysis_result, dict): continue

            raw_chapters = analysis_result.get("chapters", [])
            if isinstance(raw_chapters, dict): raw_chapters = [raw_chapters]
            if isinstance(raw_chapters, list):
                for chapter in raw_chapters:
                    if not isinstance(chapter, dict): continue
                    is_chapter_valid = True
                    for time_field in ["start", "end"]:
                         original_time_str = chapter.get(time_field)
                         if original_time_str is None: continue
                         try:
                             time_str = str(original_time_str).rstrip('s'); time_val = float(time_str); original_val = time_val
                             clamped_time_val = max(segment_start, min(segment_end, time_val))
                             if abs(clamped_time_val - original_val) > 0.0001 or not str(original_time_str).endswith('s') or len(time_str.split('.')[-1]) != 3: chapter[time_field] = f"{clamped_time_val:.3f}s"
                             else: chapter[time_field] = f"{original_val:.3f}s"
                         except (ValueError, TypeError) as e: is_chapter_valid = False; break
                    if is_chapter_valid and "start" in chapter and "end" in chapter:
                         try:
                             start_f = float(str(chapter["start"]).rstrip('s')); end_f = float(str(chapter["end"]).rstrip('s'))
                             if start_f > end_f: chapter["start"] = chapter["end"]
                         except (ValueError, TypeError): is_chapter_valid = False
                    if is_chapter_valid: chapters.append(chapter)

            global_tags_segment = analysis_result.get("globalTags", {})
            if not isinstance(global_tags_segment, dict): continue
            for tag_type in ["persons", "actions", "objects"]:
                 if tag_type not in global_tags_segment or not isinstance(global_tags_segment[tag_type], list): continue
                 for item in global_tags_segment[tag_type]:
                     if not isinstance(item, dict): continue
                     item_name = item.get("name"); timecodes = item.get("timecodes", [])
                     if not item_name or not isinstance(item_name, str) or not item_name.strip() or not isinstance(timecodes, list): continue
                     tag_name_to_aggregate = item_name.strip()
                     if not tag_name_to_aggregate: continue
                     for timestamp in timecodes:
                         if isinstance(timestamp, (float, int)):
                             tag_timestamps[tag_type][tag_name_to_aggregate].append(float(timestamp))
                         else:
                             print(f"Warning: Unexpected timestamp format encountered in process_segment_results for tag '{tag_name_to_aggregate}': {timestamp} ({type(timestamp)})")

        # --- 3. Aggregate Tags into Intervals ---
        aggregated_intervals_by_type = {
            "persons": self._aggregate_tag_timestamps(tag_timestamps["persons"]),
            "objects": self._aggregate_tag_timestamps(tag_timestamps["objects"]),
            "actions": self._aggregate_tag_timestamps(tag_timestamps["actions"])
        }
        final_person_tags = [{"name": n, "timecodes": i} for n, i in aggregated_intervals_by_type.get("persons", {}).items()]
        final_action_tags = [{"name": n, "timecodes": i} for n, i in aggregated_intervals_by_type.get("actions", {}).items()]
        final_object_tags = [{"name": n, "timecodes": i} for n, i in aggregated_intervals_by_type.get("objects", {}).items()]

        # --- 4. Convert Timestamps to Numeric (Helper function handles 'start'/'end' keys) ---
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
        top_level_meta = {}; upload_date_timestamp = None; source_details_action_summary = None
        if manifest and manifest.source_video:
             top_level_meta["title"] = manifest.name or "Untitled Video"
             top_level_meta["thumbnailUrl"] = sorted_frame_paths[0] if sorted_frame_paths else None
             top_level_meta["durationSec"] = round(manifest.source_video.duration, 3) if manifest.source_video.duration is not None else None
             top_level_meta["durationIso"] = manifest.source_video.duration_iso
             top_level_meta["contentUrl"] = None
             top_level_meta["format"] = manifest.source_video.format_name
             top_level_meta["transcriptId"] = None
             source_path = manifest.source_video.path
             try:
                  if source_path and os.path.exists(source_path): mod_time = os.path.getmtime(source_path); upload_date_dt = datetime.fromtimestamp(mod_time, timezone.utc); upload_date_timestamp = upload_date_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
             except Exception as e: print(f"Warning: Could not determine upload_date: {e}")
             top_level_meta["uploadDate"] = upload_date_timestamp

             # Create source details for action summary
             source_file_as = os.path.basename(source_path) if source_path else "Unknown"
             created_at_timestamp_as = None
             try:
                  if source_path and os.path.exists(source_path): mod_time = os.path.getmtime(source_path); created_at_dt = datetime.fromtimestamp(mod_time, timezone.utc); created_at_timestamp_as = created_at_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
             except Exception as e: print(f"Warning: Could not determine source created_at: {e}")
             source_details_action_summary = {"file": source_file_as, "durationSec": top_level_meta["durationSec"], "createdAt": created_at_timestamp_as}

        else: # Placeholders if no manifest
             top_level_meta["title"] = "Unknown Video"; top_level_meta["thumbnailUrl"] = sorted_frame_paths[0] if sorted_frame_paths else None; top_level_meta["durationSec"] = None; top_level_meta["durationIso"] = None; top_level_meta["contentUrl"] = None; top_level_meta["format"] = None; top_level_meta["transcriptId"] = None; top_level_meta["uploadDate"] = None

        # --- Add Copyright Info ---
        if parsed_copyright_info:
            top_level_meta["copyright"] = parsed_copyright_info
            print("DEBUG: Added copyright info to actionSummary metadata.")

        # --- Prepare AI Processing block ---
        ai_processing = {
             "summarization": {"createdBy": f"llm-summary-{self.name}", "model": env.vision.deployment if env and env.vision else 'N/A', "processedAt": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')},
             "chapterGeneration": {"createdBy": f"llm-chapters-{self.name}", "method": "llm-based", "processedAt": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}
        }
        top_level_meta["aiProcessing"] = ai_processing

        # --- Assemble final actionSummary content dictionary ---
        action_summary_content = {
             **top_level_meta,
             "source": source_details_action_summary,
             "transcript": full_transcript_text,
             "tagsString": tags_string,
             "tags": tags_list,
             "chapter": chapters,
             "person": final_person_tags,
             "action": final_action_tags,
             "object": final_object_tags,
             "thumbnail": sorted_frame_paths,
        }
        if action_summary_content.get("source") is None:
            action_summary_content.pop("source", None)
        if transcription_details and transcription_details.get("transcriptId"):
             action_summary_content["transcriptId"] = transcription_details.get("transcriptId")

        # --- 6. Assemble Final Output Dictionary ---
        final_output = {
            "actionSummary": action_summary_content
        }
        if transcription_details:
            final_output["transcriptionDetails"] = transcription_details

        print("DEBUG: process_segment_results finished. Returning dict with actionSummary and potentially transcriptionDetails.")
        return final_output

    # -----------------------------------------------------------------------
    # Helper function to convert timestamp arrays into intervals
    # e.g. [0.0, 0.1, 0.2, 1.0, 1.1] -> intervals [0.0-0.2, 1.0-1.1]
    # -----------------------------------------------------------------------
    def _aggregate_tag_timestamps(
        self, tag_data: Dict[str, List[float]], max_gap_initial: float = 2.0, merge_threshold: float = 2.0
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Convert lists of timestamps (floats) into [start..end] intervals with advanced merging.
        1. Initial grouping based on max_gap_initial (0.5s)
        2. Aggressive merging of adjacent intervals with gaps <= merge_threshold (3.0s)
        3. Special handling for 'Compliant' and 'Non-Compliant' tags to ensure maximum continuity
        """
        aggregated_intervals_merged = {}

        for tag_name, timestamps in tag_data.items():
            if not timestamps:
                continue

            # Remove duplicates, sort ascending
            unique_sorted_times = sorted(list(set(round(t, 3) for t in timestamps)))
            if not unique_sorted_times:
                continue

            # --- Pass 1: Create initial intervals based on max_gap_initial ---
            initial_intervals_numeric = []
            start_interval = unique_sorted_times[0]
            last_time = unique_sorted_times[0]

            for current_time in unique_sorted_times[1:]:
                if current_time - last_time > max_gap_initial + 0.001:
                    end_interval_time = max(start_interval, last_time)
                    if end_interval_time > start_interval:
                        initial_intervals_numeric.append([start_interval, end_interval_time])
                    elif end_interval_time == start_interval:
                        initial_intervals_numeric.append([start_interval, end_interval_time + 0.001])
                    start_interval = current_time
                last_time = current_time

            # Add final interval
            final_end_time = max(start_interval, last_time)
            if final_end_time > start_interval:
                initial_intervals_numeric.append([start_interval, final_end_time])
            elif final_end_time == start_interval:
                initial_intervals_numeric.append([start_interval, final_end_time + 0.001])

            # --- Pass 2: Enhanced merging for compliance tags ---
            # Use more aggressive merging for Compliant/Non-Compliant tags
            if "Compliant Construction Worker" in tag_name:
                # For compliance tags, use a more aggressive merge threshold
                special_merge_threshold = merge_threshold + 1.0  # 4.0 seconds
            else:
                special_merge_threshold = merge_threshold  # Normal 3.0 seconds
            
            # (Continue with merging logic using special_merge_threshold)
            # Sort intervals by start time
            initial_intervals_numeric.sort(key=lambda x: x[0])
            
            # Start merging
            merged_intervals_numeric = []
            if not initial_intervals_numeric:
                continue
            
            current_start, current_end = initial_intervals_numeric[0]
            
            for next_start, next_end in initial_intervals_numeric[1:]:
                gap = next_start - current_end
                if gap <= special_merge_threshold + 0.001:
                    # Merge this interval
                    current_end = max(current_end, next_end)
                else:
                    # Gap too large, finalize current interval and start a new one
                    merged_intervals_numeric.append([current_start, current_end])
                    current_start, current_end = next_start, next_end
                
            # Add the final interval
            merged_intervals_numeric.append([current_start, current_end])
            
            # --- EDIT: Format to numeric format ---
            final_numeric_intervals = [
                {"start": round(start, 3), "end": round(end, 3)} # Output numbers, rounded
                for start, end in merged_intervals_numeric
                # Ensure duration is non-negative after rounding, though merge logic should handle this
                if round(start, 3) <= round(end, 3)
            ]

            if final_numeric_intervals:
                # Store the list of numeric intervals
                aggregated_intervals_merged[tag_name] = final_numeric_intervals
            # --- END EDIT ---

        return aggregated_intervals_merged

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
