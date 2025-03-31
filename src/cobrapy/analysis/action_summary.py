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
        """You are VideoAnalyzerGPT, an AI specialized in analyzing both visual and audio content in videos.

<CURRENT_CURSOR_POSITION>
**CRITICAL INSTRUCTIONS - READ CAREFULLY - EXACT JSON & TIMESTAMP FORMAT REQUIRED:**

1. **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY these two top-level keys: "chapters" and "global_tags".
   - "chapters": An array containing EXACTLY ONE chapter object describing the current segment.
   - "global_tags": An object containing three keys: "persons", "actions", "objects" (each value must be an array of tag objects).

2. **EXACT OUTPUT FORMAT:**
{{
  "chapters": [
    {{
      "start": "45.123s", // ABSOLUTE Segment start time (e.g., from video start)
      "end": "75.456s",   // ABSOLUTE Segment end time (e.g., from video start)
      "sentiment": "neutral",
      "emotions": ["emotion1", "emotion2"],
      "transcription": "text from audio",
      "theme": "short theme",
      "summary": "detailed, descriptive summary..." // Describe setting, visuals, actions, audio context. Be specific.
    }}
  ],
  "global_tags": {{
    "persons": [
      // Example: {{"name": "person name", "timecodes": [{{"start": "50.123s", "end": "65.456s"}}]}} // PRECISE timecodes WITHIN segment boundaries
    ],
    "actions": [
      // Example: {{"name": "action name", "timecodes": [{{"start": "55.000s", "end": "58.500s"}}]}} // PRECISE timecodes WITHIN segment boundaries
    ],
    "objects": [
      // Example: {{"name": "object name", "timecodes": [{{"start": "46.000s", "end": "74.000s"}}]}} // PRECISE timecodes WITHIN segment boundaries
    ]
  }}
}}

3. **TIMESTAMP PRECISION & ABSOLUTE TIME - **VERY IMPORTANT**:**
   - **USE FRAME TIMESTAMPS:** Base ALL timestamps (`chapters` start/end AND `global_tags` timecodes start/end) on the ABSOLUTE timestamps provided with each frame in the input.
   - **BE SPECIFIC:** Global tag timecodes MUST reflect the *estimated precise time* the person/object is VISIBLY PRESENT or the action is ACTIVELY OCCURRING within the segment frames provided.
   - **DO NOT DEFAULT TO SEGMENT BOUNDARIES:** Only use the full segment start/end times for a tag's timecode if the item is genuinely present/occurring for the *entire* duration based on the frames. Otherwise, provide your best, more specific estimate.
   - **ABSOLUTE TIME ONLY:** Never use timestamps relative to the start of the segment (e.g., starting from "0.000s" if the segment begins later in the video). Always use the absolute time from the start of the video as shown in the frame timestamps.
   - **VALID RANGE:** All timestamps MUST be within the overall segment's absolute start ({start_time}) and end ({end_time}) times provided in the analysis section below.
   - **FORMAT:** All times MUST be strings in seconds with format "0.000s" (three decimal places and the 's' suffix).

4. **GLOBAL TAGS FORMAT:**
   - Must be an OBJECT `{...}` with keys "persons", "actions", "objects".
   - Each key must have a VALUE that is an ARRAY `[...]` of tag objects.
   - Each tag object MUST have a "name" (string) and "timecodes" (array of timecode objects `{{"start": "...", "end": "..."}}`).
   - DO NOT return `global_tags` as a simple list/array of strings.

5. **TAG CONSISTENCY:**
   - Check the 'Known Tags' list provided below. If a similar tag exists, REUSE THE EXACT KNOWN NAME.
   - Only add new tags if genuinely distinct.
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
        Process raw segment results (now including segment boundary info),
        merging 'global_tags' and chapters, validating/clamping timestamps,
        and building continuous intervals for tags.
        """
        tag_timestamps = {
            "persons": defaultdict(list),
            "objects": defaultdict(list),
            "actions": defaultdict(list)
        }
        chapters = []
        # segment_boundaries dictionary is no longer needed as info is in enriched_segment_results

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

            if analysis_result is None or segment_start is None or segment_end is None:
                 print(f"WARNING: Missing analysis_result or boundaries for {segment_name}. Skipping.")
                 continue

            if not isinstance(analysis_result, dict):
                print(f"WARNING: analysis_result for {segment_name} is not a dict ({type(analysis_result)}). Skipping.")
                continue

            print(f"DEBUG: Processing {segment_name} ({segment_start:.3f}s - {segment_end:.3f}s)")


            # --- Extract and Validate Chapters ---
            raw_chapters = analysis_result.get("chapters", [])
            if isinstance(raw_chapters, dict): # Handle single chapter case
                raw_chapters = [raw_chapters]

            if isinstance(raw_chapters, list):
                # print(f"DEBUG: Found {len(raw_chapters)} chapters in {segment_name}") # Can be verbose
                for chapter in raw_chapters:
                    if not isinstance(chapter, dict): continue

                    # --- ADD VALIDATION/CLAMPING FOR CHAPTER TIMESTAMPS ---
                    is_chapter_valid = True
                    for time_field in ["start", "end"]:
                        original_time_str = chapter.get(time_field)
                        if original_time_str is None: continue # Skip if field missing

                        try:
                             time_str = str(original_time_str).rstrip('s')
                             time_val = float(time_str)
                             original_val = time_val # Store for comparison

                             # Clamp to segment boundaries
                             clamped_time_val = max(segment_start, min(segment_end, time_val))

                             # Check if clamping occurred or format needs fixing
                             if abs(clamped_time_val - original_val) > 0.0001 or not str(original_time_str).endswith('s') or len(time_str.split('.')[-1]) != 3:
                                  print(f"INFO: Correcting/Clamping chapter '{time_field}' for {segment_name}: {original_time_str} -> {clamped_time_val:.3f}s (Segment: {segment_start:.3f}-{segment_end:.3f})")
                                  chapter[time_field] = f"{clamped_time_val:.3f}s"
                             else:
                                 # Ensure format is correct even if value is ok
                                 chapter[time_field] = f"{original_val:.3f}s"

                        except (ValueError, TypeError) as e:
                             print(f"ERROR: Invalid chapter timestamp format for '{time_field}' in {segment_name}: {original_time_str}. Error: {e}. Discarding chapter.")
                             is_chapter_valid = False
                             break # Stop processing this chapter

                    # Validate start <= end after potential clamping
                    if is_chapter_valid and "start" in chapter and "end" in chapter:
                        try:
                             start_f = float(str(chapter["start"]).rstrip('s'))
                             end_f = float(str(chapter["end"]).rstrip('s'))
                             if start_f > end_f:
                                 print(f"WARNING: Chapter start > end after clamping for {segment_name}: {chapter['start']} > {chapter['end']}. Setting start = end.")
                                 chapter["start"] = chapter["end"]
                        except (ValueError, TypeError):
                             print(f"ERROR: Could not compare chapter start/end after clamping for {segment_name}.")
                             is_chapter_valid = False

                    if is_chapter_valid:
                        chapters.append(chapter)
                    # --- END CHAPTER VALIDATION ---

            # --- Extract and Accumulate Global Tags (using segment boundaries) ---
            global_tags_segment = analysis_result.get("global_tags", {})

            if not isinstance(global_tags_segment, dict):
                print(f"DEBUG: global_tags in {segment_name} is not a dict ({type(global_tags_segment)}). Skipping tags.")
                continue

            # --- Accumulate timestamps (existing logic for validation/clamping is good) ---
            for tag_type in ["persons", "actions", "objects"]:
                if tag_type not in global_tags_segment or not isinstance(global_tags_segment[tag_type], list):
                    continue

                # print(f"DEBUG: Found {len(global_tags_segment[tag_type])} raw '{tag_type}' tags for {segment_name}") # Verbose
                valid_tags_processed_count = 0
                for item in global_tags_segment[tag_type]:
                    if not isinstance(item, dict): continue

                    item_name = item.get("name")
                    if not item_name or not isinstance(item_name, str) or not item_name.strip(): continue

                    tag_name_to_aggregate = item_name.strip()
                    if not tag_name_to_aggregate: continue

                    timecodes = item.get("timecodes", [])
                    if not isinstance(timecodes, list): continue

                    valid_timecodes_count = 0
                    for tcode in timecodes:
                        if not isinstance(tcode, dict) or "start" not in tcode or "end" not in tcode: continue

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
                                      start_time_clamped = end_time_clamped = segment_start # Clamp both to start
                                 elif start_time_tag > segment_end and end_time_tag > segment_end:
                                      start_time_clamped = end_time_clamped = segment_end # Clamp both to end
                                 else: # Overlap happened during clamping
                                      start_time_clamped = end_time_clamped # Make zero length at clamped end

                            # Check if interval is valid (non-negative duration and within bounds)
                            if start_time_clamped <= end_time_clamped:
                                if start_time_clamped != original_start_tag or end_time_clamped != original_end_tag:
                                    print(f"INFO: Clamped/corrected tag timecode for '{tag_name_to_aggregate}' ({tag_type}) in {segment_name}: {original_start_tag:.3f}-{original_end_tag:.3f}s -> {start_time_clamped:.3f}-{end_time_clamped:.3f}s")

                                # Add valid/clamped timestamps to aggregation
                                step = 0.1 # Resolution for interval generation
                                current_time = start_time_clamped
                                while current_time <= end_time_clamped - (step / 2): # Use epsilon for float comparison
                                    tag_timestamps[tag_type][tag_name_to_aggregate].append(round(current_time, 3))
                                    current_time += step

                                # Ensure the exact end time is included if interval > 0
                                if end_time_clamped >= start_time_clamped:
                                    end_time_rounded = round(end_time_clamped, 3)
                                    # Add end time if list is empty or last time is less than end time
                                    if not tag_timestamps[tag_type][tag_name_to_aggregate] or tag_timestamps[tag_type][tag_name_to_aggregate][-1] < end_time_rounded:
                                         tag_timestamps[tag_type][tag_name_to_aggregate].append(end_time_rounded)

                                valid_timecodes_count += 1
                            else:
                                 print(f"WARNING: Discarding invalid timestamp interval {original_start_tag:.3f}-{original_end_tag:.3f}s for '{tag_name_to_aggregate}' ({tag_type}) in {segment_name} after clamping.")


                        except (ValueError, TypeError) as e:
                            print(f"ERROR: Processing timecode for {tag_name_to_aggregate} ({tag_type}) in {segment_name}: {e}. Raw start='{start_str}', end='{end_str}'")
                            continue
                    if valid_timecodes_count > 0:
                         valid_tags_processed_count += 1
                    # print(f"DEBUG: Processed {valid_timecodes_count}/{len(timecodes)} timecodes for '{tag_name_to_aggregate}' ({tag_type}) in {segment_name}") # Verbose
                if len(global_tags_segment[tag_type]) > 0 :
                     print(f"DEBUG: Processed {valid_tags_processed_count}/{len(global_tags_segment[tag_type])} valid tags for '{tag_type}' in {segment_name}")


        # After gathering all second-level timestamps, convert them into intervals
        aggregated_tags = {
            "persons": self._aggregate_tag_timestamps(tag_timestamps["persons"]),
            "objects": self._aggregate_tag_timestamps(tag_timestamps["objects"]),
            "actions": self._aggregate_tag_timestamps(tag_timestamps["actions"])
        }
        print(f"DEBUG: Final aggregated tags counts - persons: {len(aggregated_tags['persons'])}, objects: {len(aggregated_tags['objects'])}, actions: {len(aggregated_tags['actions'])}")


        # Build final merged data
        final_results = {
            "chapters": chapters, # Use the validated and potentially clamped chapters
            "global_tags": {
                tag_type: [
                    {"name": tag_name, "timecodes": intervals}
                    for tag_name, intervals in tag_data.items()
                ]
                for tag_type, tag_data in aggregated_tags.items()
            }
        }

        return final_results

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
            # Round timestamps before sorting to handle potential float inaccuracies from linspace/rounding
            unique_sorted_times = sorted(list(set(round(t, 3) for t in timestamps)))
            if not unique_sorted_times:
                continue

            intervals = []
            if not unique_sorted_times: continue # Should be redundant, but safe

            start_interval = unique_sorted_times[0]
            last_time = unique_sorted_times[0]

            for current_time in unique_sorted_times[1:]:
                # If gap is too big OR float precision makes gap slightly larger than step
                if current_time - last_time > max_gap + 0.001: # Add tolerance for float math
                    # Ensure interval end is not before start
                    end_interval_time = max(start_interval, last_time)
                    intervals.append({
                        "start": f"{start_interval:.3f}s",
                        "end": f"{end_interval_time:.3f}s" # Use max
                    })
                    start_interval = current_time # Start new interval
                # Update last_time regardless of whether a new interval started
                last_time = current_time


            # Add the final open interval
            final_end_time = max(start_interval, last_time)
            intervals.append({
                "start": f"{start_interval:.3f}s",
                "end": f"{final_end_time:.3f}s" # Use max
            })

            aggregated_intervals[tag_name] = intervals

        return aggregated_intervals
