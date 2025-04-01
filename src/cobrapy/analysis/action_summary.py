from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar
from collections import defaultdict

class ActionSummary(AnalysisConfig):
    """
    This AnalysisConfig enforces that the LLM must produce a JSON response with 
    top-level 'chapters' and 'global_tags' keys. The 'chapters' must have exactly
    one object describing the segment, and 'global_tags' must always include 
    'persons', 'actions', and 'objects' arrays, even if they are empty.
    
    The code below also includes a process_segment_results() method that merges
    any segment-level tags into continuous intervals at second-level accuracy.
    """

    name: str = "ActionSummary"
    analysis_sequence: str = "mapreduce"

    # -----------------------------------------------------------------------
    # System Prompt:
    # -----------------------------------------------------------------------
    system_prompt: str = (
        """You are VideoAnalyzerGPT, an AI specialized in analyzing both visual and audio content in videos. Your goal is to identify and tag *every relevant and distinct* person, action, and object within the provided video segment, using precise timestamps based *only* on the frame data provided for this segment.

**CRITICAL INSTRUCTIONS - READ CAREFULLY - EXACT JSON, TIMESTAMP FORMAT, TAG SEPARATION, and TAG CONSISTENCY REQUIRED:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY these two top-level keys: "chapters" and "global_tags".
    *   "chapters": An array containing EXACTLY ONE chapter object describing the current segment.
    *   "global_tags": An object containing three keys: "persons", "actions", "objects" (each value must be an array of tag objects `{ "name": "...", "timecodes": [...] }`). **Arrays MUST be included even if empty.**

2.  **EXACT OUTPUT FORMAT:**
    ```json
    {
      "chapters": [
        {
          "start": "45.123s", // ABSOLUTE Segment start time (e.g., from video start)
          "end": "75.456s",   // ABSOLUTE Segment end time (e.g., from video start)
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "transcription": "text from audio",
          "theme": "short theme",
          "summary": "detailed, descriptive summary..." // Describe setting, visuals, actions, audio context. Be specific.
        }
      ],
      "global_tags": {
        "persons": [
          // Example: {"name": "Man in blue suit", "timecodes": [{"start": "50.123s", "end": "65.456s"}]}
        ],
        "actions": [
          // Example: {"name": "Walking down hallway", "timecodes": [{"start": "55.000s", "end": "58.500s"}]} // Action ONLY
        ],
        "objects": [
          // Example: {"name": "Red laptop", "timecodes": [{"start": "46.000s", "end": "74.000s"}]}
        ]
      }
    }
    ```

3.  **TIMESTAMP PRECISION & ACCURACY - **VERY IMPORTANT**:**
    *   **USE FRAME TIMESTAMPS:** Base ALL timestamps (`chapters` start/end AND `global_tags` timecodes start/end) **strictly** on the ABSOLUTE timestamps provided with each frame in the input for *this segment*.
    *   **ACCURATE DURATION:** Global tag timecodes MUST reflect the *estimated precise interval(s)* the person/object is VISIBLY PRESENT or the action is ACTIVELY OCCURRING within the segment frames. Correlate visual presence directly with frame timestamps.
    *   **DO NOT DEFAULT TO SEGMENT BOUNDARIES:** Only use the full segment start/end times ({start_time}s to {end_time}s) for a tag's timecode if the item is *genuinely* present/occurring for the *entire* duration based on the frames. Otherwise, provide your best, more specific estimate for the start and end of its appearance/occurrence *within* this segment.
    *   **ABSOLUTE TIME ONLY:** Never use relative timestamps (e.g., "0.000s" if the segment starts later). Always use absolute time from the video start.
    *   **VALID RANGE:** All timestamps MUST fall within the segment's absolute start ({start_time}s) and end ({end_time}s) times.
    *   **FORMAT:** All times MUST be strings in seconds format "0.000s" (three decimal places, 's' suffix).

4.  **TAG DEFINITIONS & EXAMPLES:**
    *   **Persons:**
        *   Definition: Clearly visible individuals.
        *   **Naming Rule:** The 'name' MUST describe the person's **appearance and static features ONLY**. **DO NOT include their current action (e.g., "walking", "sitting", "holding papers") in the person's name.** The action should be tagged separately under "actions".
        *   Naming Guideline: Use descriptive, consistent identifiers focusing on distinguishing features (clothing color/type, hair color/style, glasses, beard). Be concise but specific enough to differentiate individuals.
        *   **Examples:** "Woman in red dress", "Man in grey suit", "Man with grey beard wearing glasses", "Child in yellow shirt".
        *   **Generic Persons:** If a person lacks strong distinguishing features or is distant/unclear, use a simple generic description plus a temporary ID *for this segment only* (e.g., "Person in background (ID: P1)", "Office worker at desk (ID: P2)"). **Ensure the ID is only used if description alone is insufficient.**
    *   **Objects:**
        *   Definition: Physical items relevant to the scene or interaction.
        *   **Naming Rule:** Describe the object itself, not its state or interaction (e.g., use "Briefcase", not "Briefcase being carried").
        *   Naming Guideline: Be specific and consistent. Use distinguishing features (color, material, type).
        *   **Examples:** "Dell laptop", "Blue ceramic coffee mug", "Whiteboard with notes", "Silver sedan", "Briefcase", "Television", "Notebook", "Glass pitcher".
        *   Guideline: Tag objects that are actively used, interacted with, explicitly mentioned, or visually prominent and contextually important. Avoid minor background clutter unless relevant.
    *   **Actions:**
        *   Definition: Meaningful activities, movements, or interactions performed by persons or involving objects.
        *   **Naming Rule:** Describe the action using clear, concise action verbs. This is where you describe *what* is happening.
        *   **Examples:** "Walking down hallway", "Sitting at desk", "Holding papers", "Carrying briefcase", "Typing on keyboard", "Shaking hands", "Gesturing with hand", "Presenting slides".
        *   Guideline: Tag distinct, significant activities. Combine minor related movements into a single descriptive action (e.g., "Adjusting tie" instead of "Touching tie", "Moving hand to tie"). Focus on actions relevant to the narrative or main interactions.

5.  **TAG CONSISTENCY & UNIQUENESS - **CRITICAL**:**
    *   **TRACK WITHIN SEGMENT:** Identify and track distinct entities (persons, objects, actions) *throughout this specific segment*.
    *   **REUSE NAMES WITHIN SEGMENT:** If you identify the *same* person, object, or ongoing action appearing multiple times or continuously *within this segment*, you **MUST** use the *exact same "name" string* you used previously *for that specific entity within this segment*. Aim for ONE entry per unique entity per segment, with potentially multiple timecode intervals.
    *   **SEMANTIC CONSISTENCY (Persons/Objects):** Choose the most stable and descriptive name for a person or object *the first time* you see it clearly in the segment, and **stick to that exact name** if you see the same entity again in the segment. For example, if you identify "Man in grey suit", use that exact name every time he appears in this segment, do not switch to "Person in grey suit" or add actions like "Man in grey suit walking".
    *   **KNOWN TAGS (Actions/Objects):** Check the 'Known Tags' list provided below. If an action or object in this segment *exactly matches* or is *semantically identical* to a name in the 'Known Tags' list, you **MUST** reuse that **exact** known name.
    *   **NEW TAGS:** Only create a *new* tag name if the entity is genuinely distinct from others *in this segment* AND is not present in the 'Known Tags' list. Follow the naming rules in Point 4.

6.  **GLOBAL TAGS JSON FORMAT:**
    *   Must be an OBJECT `{...}` with keys "persons", "actions", "objects".
    *   Each key's value must be an ARRAY `[...]` of tag objects.
    *   Each tag object MUST have a "name" (string) and "timecodes" (array of timecode objects `{"start": "...", "end": "..."}`).
    *   DO NOT return `global_tags` as a list of strings.

7.  **COMMON MISTAKES TO AVOID:**
    *   **Do not put actions in person names:** Person name: "Man in suit". Action name: "Walking". Correct. | Person name: "Man in suit walking". Incorrect.
    *   **Do not use inconsistent names:** Use "Man in grey suit" every time if it's the same person, not "Person in grey suit" later.
    *   **Do not forget empty arrays:** If no persons are tagged, still include `"persons": []`.
    *   **Do not use relative timestamps:** Always use absolute time like "123.456s".
"""
    )

    # -----------------------------------------------------------------------
    # Additional lens or constraints appended to system prompt
    # -----------------------------------------------------------------------
    system_prompt_lens: str = (
        """Analyze the segment from {start_time}s to {end_time}s. The segment contains:
- {number_of_frames} frames (with timestamps)
- Possibly an audio transcription

Remember, the 'chapters' array must have EXACTLY one item describing this segment.
The 'global_tags' object must have 'persons', 'actions', 'objects'.

Use the segment's transcription or "No transcription available" if none.

Known Tags (Reuse these names if applicable for Actions and Objects):
- Known Actions: {known_actions}
- Known Objects: {known_objects}
"""
    )

    # -----------------------------------------------------------------------
    # Example structure for use in dev or debugging
    # -----------------------------------------------------------------------
    results_template: Dict[str, Any] = {
        "chapters": [
            {
                "start": "0.000s",
                "end": "30.000s",
                "sentiment": "neutral",
                "emotions": ["calm"],
                "transcription": "Full transcription text for this segment",
                "theme": "office meeting",
                "summary": "Detailed description of what is occurring in this segment"
            }
        ],
        "global_tags": {
            "persons": [
                {
                    "name": "Person Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "15.000s"}
                    ]
                }
            ],
            "actions": [
                {
                    "name": "Action Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "5.000s"}
                    ]
                }
            ],
            "objects": [
                {
                    "name": "Object Name",
                    "timecodes": [
                        {"start": "0.000s", "end": "10.000s"}
                    ]
                }
            ]
        }
    }

    # Whether we run a final summary across all segments after everything finishes
    run_final_summary: bool = True

    # The prompt used for the final summary across the entire video
    summary_prompt: str = (
        """Analyze the complete video analysis results and provide a concise summary 
        that captures the key narrative, main participants, significant actions, 
        and important objects across all segments. Focus on the overall flow and 
        major themes rather than segment-by-segment details."""
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
    def process_segment_results(self, enriched_segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw segment results (now including segment boundary info and frame paths),
        merging 'global_tags' and chapters, validating/clamping timestamps,
        and building continuous intervals for tags.
        """
        tag_timestamps = {
            "persons": defaultdict(list),
            "objects": defaultdict(list),
            "actions": defaultdict(list)
        }
        chapters = []
        all_frame_paths = set()  # <-- Initialize set to collect unique frame paths
        full_transcription_obj = None
        full_transcript_text = ""
        processed_speech_segments = []

        # Find the full transcription object (Batch API format)
        for container in enriched_segment_results:
            if (
                container
                and isinstance(container.get("full_transcription_object"), dict)
                and "recognizedPhrases" in container["full_transcription_object"]
            ):
                full_transcription_obj = container["full_transcription_object"]
                print("DEBUG: Found Batch Transcription result object.")
                break

        if full_transcription_obj:
            # --- Extract primary language for the entire audio file ---
            # Azure Batch API stores the detected language in different places depending on the API version
            # First check the most common locations:
            primary_language = None

            # Method 1: Some API versions return it in a top-level field
            if "language" in full_transcription_obj:
                primary_language = full_transcription_obj.get("language")

            # Method 2: More commonly, language ID results are in the properties/recognitionMetadata
            elif "properties" in full_transcription_obj:
                props = full_transcription_obj.get("properties", {})
                # Try several possible property names where language ID might be stored
                lang_id = (
                    props.get("languageIdentification")
                    or props.get("languageDetectionResult")
                    or props.get("recognitionMetadata", {}).get("language")
                )
                if lang_id:
                    primary_language = lang_id

            # Method 3: Check in the first recognized phrase if it has language info
            if not primary_language:
                phrases = full_transcription_obj.get("recognizedPhrases", [])
                if phrases and isinstance(phrases, list) and len(phrases) > 0:
                    first_phrase = phrases[0]
                    if isinstance(first_phrase, dict) and first_phrase.get("language"):
                        primary_language = first_phrase.get("language")

            # Fallback to default if still not found
            if not primary_language or primary_language == "unk":
                # The audio was probably all English based on our candidateLocales setting
                primary_language = "en-US"

            print(f"DEBUG: Primary detected language for the transcript: {primary_language}")

            # --- Process Detailed Speech Segments from Batch Transcription Output ---
            raw_speech_phrases = full_transcription_obj.get("recognizedPhrases", [])
            if isinstance(raw_speech_phrases, list):
                print(f"DEBUG: Processing {len(raw_speech_phrases)} recognized phrases for diarization/language.")
                # Reconstruct full transcript text
                full_transcript_text = " ".join(
                    phrase.get("nBest", [{}])[0].get("display", "")
                    for phrase in raw_speech_phrases
                    if phrase.get("nBest")
                ).strip()

                for phrase in raw_speech_phrases:
                    if not isinstance(phrase, dict):
                        continue
                    try:
                        offset_ticks = phrase.get("offsetInTicks", 0)
                        duration_ticks = phrase.get("durationInTicks", 0)
                        speaker_id = phrase.get("speaker")
                        best_recognition = phrase.get("nBest", [{}])[0]
                        text = best_recognition.get("display", "")
                        confidence = best_recognition.get("confidence")

                        start_time = offset_ticks / 10_000_000.0
                        end_time = start_time + (duration_ticks / 10_000_000.0)
                        speaker_mapped = f"Speaker {speaker_id}" if speaker_id is not None else "Unknown Speaker"

                        processed_speech_segments.append({
                            "text": text,
                            "speaker": speaker_mapped,
                            "start": f"{start_time:.3f}s",
                            "end": f"{end_time:.3f}s",
                            "confidence": confidence
                        })
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"WARNING: Skipping speech phrase due to processing error: {e}. Data: {phrase}")
            else:
                print("DEBUG: No valid 'recognizedPhrases' list found in transcription object.")
                full_transcript_text = "Transcription data missing or format error."
        else:
            print("DEBUG: Batch Transcription result object not found in enriched results.")
            full_transcript_text = "Transcription data unavailable."

        print(f"DEBUG: Starting aggregation with {len(enriched_segment_results)} enriched segment results")

        # Iterate through the enriched results list
        for i, result_container in enumerate(enriched_segment_results):
            # --- Extract segment info and analysis result ---
            if not isinstance(result_container, dict):
                print(f"WARNING: Skipping non-dict item in enriched_segment_results index {i}")
                continue

            analysis_result = result_container.get("analysis_result")
            segment_name = result_container.get("segment_name", f"segment_{i}")
            segment_start = result_container.get("start_time")
            segment_end = result_container.get("end_time")
            segment_frame_paths = result_container.get("frame_paths", [])  # <-- Extract frame paths

            # --- Collect Frame Paths ---
            if isinstance(segment_frame_paths, list):
                all_frame_paths.update(segment_frame_paths)  # Add paths to the set

            if analysis_result is None or segment_start is None or segment_end is None:
                print(f"WARNING: Missing analysis_result or boundaries for {segment_name}. Skipping.")
                continue

            if not isinstance(analysis_result, dict):
                print(f"WARNING: analysis_result for {segment_name} is not a dict ({type(analysis_result)}). Skipping.")
                continue

            print(f"DEBUG: Processing {segment_name} ({segment_start:.3f}s - {segment_end:.3f}s)")

            # --- Extract and Validate Chapters ---
            raw_chapters = analysis_result.get("chapters", [])
            if isinstance(raw_chapters, dict):  # Handle single chapter case
                raw_chapters = [raw_chapters]

            if isinstance(raw_chapters, list):
                # print(f"DEBUG: Found {len(raw_chapters)} chapters in {segment_name}") # Can be verbose
                for chapter in raw_chapters:
                    if not isinstance(chapter, dict):
                        continue

                    # --- ADD VALIDATION/CLAMPING FOR CHAPTER TIMESTAMPS ---
                    is_chapter_valid = True
                    for time_field in ["start", "end"]:
                        original_time_str = chapter.get(time_field)
                        if original_time_str is None:
                            continue  # Skip if field missing

                        try:
                            time_str = str(original_time_str).rstrip('s')
                            time_val = float(time_str)
                            original_val = time_val  # Store for comparison

                            # Clamp to segment boundaries
                            clamped_time_val = max(segment_start, min(segment_end, time_val))

                            # Check if clamping occurred or format needs fixing
                            if (
                                abs(clamped_time_val - original_val) > 0.0001
                                or not str(original_time_str).endswith('s')
                                or len(time_str.split('.')[-1]) != 3
                            ):
                                print(
                                    f"INFO: Correcting/Clamping chapter '{time_field}' for {segment_name}: "
                                    f"{original_time_str} -> {clamped_time_val:.3f}s "
                                    f"(Segment: {segment_start:.3f}-{segment_end:.3f})"
                                )
                                chapter[time_field] = f"{clamped_time_val:.3f}s"
                            else:
                                # Ensure format is correct even if value is ok
                                chapter[time_field] = f"{original_val:.3f}s"

                        except (ValueError, TypeError) as e:
                            print(
                                f"ERROR: Invalid chapter timestamp format for '{time_field}' "
                                f"in {segment_name}: {original_time_str}. Error: {e}. Discarding chapter."
                            )
                            is_chapter_valid = False
                            break  # Stop processing this chapter

                    # Validate start <= end after potential clamping
                    if is_chapter_valid and "start" in chapter and "end" in chapter:
                        try:
                            start_f = float(str(chapter["start"]).rstrip('s'))
                            end_f = float(str(chapter["end"]).rstrip('s'))
                            if start_f > end_f:
                                print(
                                    f"WARNING: Chapter start > end after clamping for {segment_name}: "
                                    f"{chapter['start']} > {chapter['end']}. Setting start = end."
                                )
                                chapter["start"] = chapter["end"]
                        except (ValueError, TypeError):
                            print(
                                f"ERROR: Could not compare chapter start/end after clamping for {segment_name}."
                            )
                            is_chapter_valid = False

                    if is_chapter_valid:
                        chapters.append(chapter)
                    # --- END CHAPTER VALIDATION ---

            # --- Extract and Accumulate Global Tags (using segment boundaries) ---
            global_tags_segment = analysis_result.get("global_tags", {})

            if not isinstance(global_tags_segment, dict):
                print(
                    f"DEBUG: global_tags in {segment_name} is not a dict ({type(global_tags_segment)}). "
                    f"Skipping tags."
                )
                continue

            # --- Accumulate timestamps (existing logic for validation/clamping) ---
            for tag_type in ["persons", "actions", "objects"]:
                if tag_type not in global_tags_segment or not isinstance(global_tags_segment[tag_type], list):
                    continue

                valid_tags_processed_count = 0
                for item in global_tags_segment[tag_type]:
                    if not isinstance(item, dict):
                        continue

                    item_name = item.get("name")
                    if not item_name or not isinstance(item_name, str) or not item_name.strip():
                        continue

                    tag_name_to_aggregate = item_name.strip()
                    if not tag_name_to_aggregate:
                        continue

                    timecodes = item.get("timecodes", [])
                    if not isinstance(timecodes, list):
                        continue

                    valid_timecodes_count = 0
                    for tcode in timecodes:
                        if not isinstance(tcode, dict) or "start" not in tcode or "end" not in tcode:
                            continue

                        start_str = str(tcode.get("start", "N/A")).rstrip("s")
                        end_str = str(tcode.get("end", "N/A")).rstrip("s")

                        try:
                            start_time_tag = float(start_str)
                            end_time_tag = float(end_str)
                            original_start_tag, original_end_tag = start_time_tag, end_time_tag

                            # Clamp tag times to segment boundaries
                            start_time_clamped = max(segment_start, min(segment_end, start_time_tag))
                            end_time_clamped = max(segment_start, min(segment_end, end_time_tag))

                            # Ensure start <= end after clamping
                            if start_time_clamped > end_time_clamped:
                                # If clamping inverted, make interval zero-length at the boundary
                                if start_time_tag < segment_start and end_time_tag < segment_start:
                                    start_time_clamped = end_time_clamped = segment_start
                                elif start_time_tag > segment_end and end_time_tag > segment_end:
                                    start_time_clamped = end_time_clamped = segment_end
                                else:
                                    start_time_clamped = end_time_clamped

                            # Check if interval is valid (non-negative duration and within bounds)
                            if start_time_clamped <= end_time_clamped:
                                if (
                                    start_time_clamped != original_start_tag
                                    or end_time_clamped != original_end_tag
                                ):
                                    print(
                                        f"INFO: Clamped/corrected tag timecode for '{tag_name_to_aggregate}' "
                                        f"({tag_type}) in {segment_name}: "
                                        f"{original_start_tag:.3f}-{original_end_tag:.3f}s -> "
                                        f"{start_time_clamped:.3f}-{end_time_clamped:.3f}s"
                                    )

                                # Add valid/clamped timestamps to aggregation
                                step = 0.1  # Resolution for interval generation
                                current_time = start_time_clamped
                                while current_time <= end_time_clamped - (step / 2):
                                    tag_timestamps[tag_type][tag_name_to_aggregate].append(round(current_time, 3))
                                    current_time += step

                                # Ensure the exact end time is included if interval > 0
                                if end_time_clamped >= start_time_clamped:
                                    end_time_rounded = round(end_time_clamped, 3)
                                    # Add end time if list is empty or last time is less than end time
                                    if (
                                        not tag_timestamps[tag_type][tag_name_to_aggregate]
                                        or tag_timestamps[tag_type][tag_name_to_aggregate][-1] < end_time_rounded
                                    ):
                                        tag_timestamps[tag_type][tag_name_to_aggregate].append(end_time_rounded)

                                valid_timecodes_count += 1
                            else:
                                print(
                                    f"WARNING: Discarding invalid timestamp interval {original_start_tag:.3f}-"
                                    f"{original_end_tag:.3f}s for '{tag_name_to_aggregate}' "
                                    f"({tag_type}) in {segment_name} after clamping."
                                )

                        except (ValueError, TypeError) as e:
                            print(
                                f"ERROR: Processing timecode for {tag_name_to_aggregate} "
                                f"({tag_type}) in {segment_name}: {e}. Raw start='{start_str}', end='{end_str}'"
                            )
                            continue

                    if valid_timecodes_count > 0:
                        valid_tags_processed_count += 1

                if len(global_tags_segment[tag_type]) > 0:
                    print(
                        f"DEBUG: Processed {valid_tags_processed_count}/"
                        f"{len(global_tags_segment[tag_type])} valid tags for '{tag_type}' in {segment_name}"
                    )

        # After gathering all second-level timestamps, convert them into intervals
        aggregated_intervals_by_type = {
            "persons": self._aggregate_tag_timestamps(tag_timestamps["persons"]),
            "objects": self._aggregate_tag_timestamps(tag_timestamps["objects"]),
            "actions": self._aggregate_tag_timestamps(tag_timestamps["actions"])
        }
        print(
            f"DEBUG: Final aggregated intervals counts - "
            f"persons: {len(aggregated_intervals_by_type['persons'])}, "
            f"objects: {len(aggregated_intervals_by_type['objects'])}, "
            f"actions: {len(aggregated_intervals_by_type['actions'])}"
        )

        # --- New Structure Creation ---
        # Convert the aggregated interval dictionaries into lists of objects with singular keys
        final_person_tags = [
            {"name": tag_name, "timecodes": intervals}
            for tag_name, intervals in aggregated_intervals_by_type.get("persons", {}).items()
        ]
        final_action_tags = [
            {"name": tag_name, "timecodes": intervals}
            for tag_name, intervals in aggregated_intervals_by_type.get("actions", {}).items()
        ]
        final_object_tags = [
            {"name": tag_name, "timecodes": intervals}
            for tag_name, intervals in aggregated_intervals_by_type.get("objects", {}).items()
        ]

        # Convert collected frame paths set to a sorted list
        sorted_frame_paths = sorted(list(all_frame_paths))

        # Build the content for the main key
        final_action_summary_content = {
            "transcript": full_transcript_text,
            "language": primary_language,
            "chapter": chapters,
            "speechSegment": processed_speech_segments,
            "person": final_person_tags,
            "action": final_action_tags,
            "object": final_object_tags,
            "thumbnail": sorted_frame_paths
        }

        # Wrap everything under the single 'actionSummary' top-level key
        final_output = {
            "actionSummary": final_action_summary_content
        }

        print("DEBUG: process_segment_results finished (using Batch Transcription data).")
        return final_output

    # -----------------------------------------------------------------------
    # Helper function to convert timestamp arrays into intervals
    # e.g. [0.0, 0.1, 0.2, 1.0, 1.1] -> intervals [0.0-0.2, 1.0-1.1]
    # -----------------------------------------------------------------------
    def _aggregate_tag_timestamps(
        self, tag_data: Dict[str, List[float]], max_gap: float = 0.5
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Convert lists of timestamps (floats) into [start..end] intervals.
        Each time series is grouped if consecutive timestamps are within max_gap.
        """
        aggregated_intervals = {}

        for tag_name, timestamps in tag_data.items():
            if not timestamps:
                continue

            # Remove duplicates, sort ascending
            unique_sorted_times = sorted(list(set(round(t, 3) for t in timestamps)))
            if not unique_sorted_times:
                continue

            intervals = []
            start_interval = unique_sorted_times[0]
            last_time = unique_sorted_times[0]

            for current_time in unique_sorted_times[1:]:
                # If gap is too big (consider float imprecision)
                if current_time - last_time > max_gap + 0.001:
                    # Close off the previous interval
                    end_interval_time = max(start_interval, last_time)
                    intervals.append({
                        "start": f"{start_interval:.3f}s",
                        "end": f"{end_interval_time:.3f}s"
                    })
                    start_interval = current_time
                last_time = current_time

            # Add the final open interval
            final_end_time = max(start_interval, last_time)
            intervals.append({
                "start": f"{start_interval:.3f}s",
                "end": f"{final_end_time:.3f}s"
            })

            aggregated_intervals[tag_name] = intervals

        return aggregated_intervals
