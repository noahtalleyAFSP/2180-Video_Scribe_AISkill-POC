from .base_analysis_config import AnalysisConfig
from typing import Dict, Any, List, ClassVar, Optional
from collections import defaultdict
import os
from datetime import datetime, timezone
from ..cobra_utils import seconds_to_iso8601_duration, merge_overlapping, convert_string_to_seconds
import json
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from time import perf_counter
from openai.types import Completion
from ..models.video import VideoManifest, Segment
from ..models.environment import CobraEnvironment

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
        """You are VideoAnalyzerGPT, analyzing basketball game footage like a **sports commentator**. Your goal is to generate a single, detailed description for the provided time range, using precise timestamps based *only* on the frame data provided. Focus on the play-by-play action, player movements, and game situation.

**GAME CONTEXT:**
*   **Teams:** Cleveland Cavaliers (Cavs) vs. Miami Heat.
*   **Jersey Colors:** Cavs are wearing **RED/MAROON** jerseys. Heat are wearing **WHITE** jerseys.
*   **IMPORTANT:** When describing players, use their team affiliation (e.g., "Heat player", "Cavaliers player") or their specific name if known, rather than just the jersey color.

**CONTEXT - AVAILABLE TAGS:**
*   **Recognized Persons/Roles:** {person_labels_list}
*   **Recognized Actions:** {action_labels_list}
*   **Refer to these specific player and action labels** whenever possible for accuracy and consistency (e.g., use 'Tyler Hero' if identified, not just 'Heat player').

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    *   "chapters": An array containing EXACTLY ONE chapter object describing the content within the specified time range. **DO NOT include a "globalTags" key.**

2.  **SHOT ANALYSIS:** Within the chapter object, include:
    *   `shotType`: Classify the dominant camera shot types observed during this segment (e.g., Wide Shot, Close-Up on player). Provide this as a **LIST of strings**. Choose **one or more** from the following list: {shot_types_list}. If the shot changes, include all relevant types observed.
    *   `shotDescription`: Describe the shot's composition (players, court position, background), visual style, and any noticeable camera movement (e.g., Tracking player, Zoom on basket).

3.  **EXACT OUTPUT FORMAT:** Use the *actual start and end times* for this segment (`{start_time}s` to `{end_time}s`) in the `start` and `end` fields of your JSON output.
    ```json
    {{
      "chapters": [
        {{
          "start": "{start_time}s",
          "end": "{end_time}s",
          "shotType": ["Medium Shot (MS)", "Close Up (CU)"],
          "shotDescription": "Medium shot follows the Heat player dribbling up court, zooms in for a close-up as #14 Tyler Herro attempts a layup...", // Use team/specific names
          "sentiment": "neutral", // e.g., exciting, tense, routine
          "emotions": [], // Observed player emotions (e.g., frustration, celebration)
          "theme": "e.g., Fast break, Defensive stand, Timeout discussion",
          "summary": "Detailed, play-by-play summary using commentator language and specific labels (e.g., 'Cavaliers' #3 Caris LeVert passes to #31 Jarrett Allen who goes for a slam dunk')..." // Use team/specific names
        }}
      ]
    }}
    ```

4.  **TIMESTAMP PRECISION & ACCURACY:**
    *   Use the absolute start ({start_time}s) and end ({end_time}s) times provided for the current time range.
    *   Format: "0.000s".

5.  **COMMENTARY SUMMARY CONTENT:** Describe the setting (arena, court position), visuals, key player actions (using specific labels like 'dribbling', 'passing', 'shooting', 'slam dunk' from the list above), and identify players using their specific labels (e.g., '#14 Tyler Herro', 'Referee') or team affiliation (Heat/Cavaliers) whenever possible. Mention relevant audio context (crowd noise, whistles). **Speak like a sports commentator.** Focus entirely on *what is happening in the game*. Interweave information from the images and transcription. **Base your entire description *only* on the provided frame images and transcription text. DO NOT invent details.** Avoid meta-commentary like "in this segment".
"""
    )

    # -----------------------------------------------------------------------
    # System Prompt - Combined Tags (Persons & Actions)
    # -----------------------------------------------------------------------
    system_prompt_tags: ClassVar[str] = (
"""You are VideoAnalyzerGPT, tagging basketball footage. **Use ONLY the labels provided.**
**If no jersey number is 100 % readable in a frame, return an empty `persons` array.**Output MUST be a JSON object with ONLY "persons" and "actions" top-level keys:

```json
{
  "persons": [ { "name": str, "timecodes": [ { "start": "X.XXXs", "end": "Y.YYYs" } ] } ],
  "actions": [ { "name": str, "timecodes": [ { "start": "X.XXXs", "end": "Y.YYYs" } ] } ]
}
```

**TIMING SOURCE:** Use ONLY the `frame_timestamps` array (provided in the Frame data JSON) for your `start` / `end` fields.
- Adhere strictly to the allowed labels listed in the user message.
- Use timestamps EXACTLY as provided in the frame data (`"X.XXXs"`).
- Group consecutive frames (gap ≤ 0.25 s) into one timecode entry; otherwise
- output distinct intervals even if they are only a single frame long.
**VISUAL vs AUDIO CUES for SCORING PLAYS**
* **Never infer time from the score graphic.**
- For "persons": Only tag players if Rules 1-3 (see user JSON) all pass in a frame.
- If visible in just one frame, set `"start"` = `"end"` = that frame’s timestamp.
- For "actions": Only tag actions clearly visible AND listed in the allowed list.
- **DO NOT GUESS. DO NOT USE LABELS NOT PROVIDED.**
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
        """Analyze the complete basketball video analysis results (chapters and tags) provided below.
        Your task is to generate TWO summaries and classify the video content, returning them ONLY as a single JSON object. **Write like a sports commentator.**

        Classification:
        *   `category`: Choose the **single most appropriate** category for the entire video from this list: {asset_categories_list}. (Likely 'Sports' for basketball).
        *   `subCategory`: Provide a specific basketball-related sub-category (e.g., 'NBA Highlights', 'College Game Analysis', 'Player Profile', 'Skills Tutorial'). **DO NOT invent a sub-category if one isn't clear.** Leave it as `null` or omit the key if unsure.

        Summaries (Commentator Perspective):
        1.  **description**: A concise, 1-2 sentence **headline** summarizing the core action or outcome of the video content (e.g., "Highlights of the Cavaliers' narrow victory over the Heat, sealed by a last-second shot.").
        2.  **summary**: A detailed **game recap** or analysis that captures the key narrative, main players involved, significant plays (shots, passes, dunks, blocks), and overall flow. Mention key stats or moments if available in the tags/chapters. **The summary should be {summary_length_instruction}**

        CRITICAL: You MUST return ONLY a valid JSON object containing `description`, `summary`, `category`, and optionally `subCategory`. Example format:
        ```json
        {{
          "category": "Sports",
          "subCategory": "NBA Highlights",
          "description": "Exciting highlights from the Lakers vs Celtics game, featuring dominant performances...",
          "summary": "The detailed game recap from a commentator's view..."
        }}
        ```
        Do not include any text outside of this JSON structure."""
    )

    # Track known tags at the class level for FINAL aggregation if needed
    # We will use instance-level tracking in VideoAnalyzer for prompting
    known_tags: ClassVar[Dict[str, set]] = {
        "persons": set(),
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
            full_transcript_text = get_full_transcript_text(phrases)

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

                    # --- FIX: Convert transcript segment tuple to dict --- 
                    segment_dict = {
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
                    }
                    processed_segments.append(segment_dict)
                    # --- END FIX ---
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
        collected_intervals_by_tag: Dict[str, Dict[str, List[Dict]]] = {
            "persons": {},
            "actions": {},
        }
        all_chapters = []
        scoreboard_event_count = 0 # Initialize counter

        for i, container in enumerate(enriched_segment_results): # Use enumerate to get index 'i'
             # --- Get analysis result (handle both keys) ---
             analysis_result = (container.get("analysisResult")
                                or container.get("analysis_result")
                                or {})

             # --- ADDED DEBUG LOG ---
             print(f"DEBUG Process Seg {i}: analysis_result keys: {list(analysis_result.keys())}")
             print(f"DEBUG Process Seg {i}: globalTags content: {analysis_result.get('globalTags')}")
             # --- END ADDED DEBUG LOG ---

             if not isinstance(container, dict) or not isinstance(analysis_result, dict):
                 print(f"DEBUG: Skipping container {i} due to missing/invalid structure or analysisResult.")
                 continue

             # --- Try to get the corresponding segment for start time ---
             current_segment: Optional["Segment"] = None
             segment_start_time = 0.0
             try:
                 # Assuming index 'i' corresponds to the segment index in the manifest
                 if manifest and manifest.segments and i < len(manifest.segments):
                      current_segment = manifest.segments[i]
                      segment_start_time = current_segment.start_time if current_segment.start_time is not None else 0.0
                 else:
                      print(f"Warning: Could not reliably get segment start time for container {i}. Using 0.0s fallback.")
             except (IndexError, TypeError, AttributeError) as e:
                 print(f"Warning: Error getting segment start time for container {i}: {e}. Using 0.0s fallback.")


             # --- Collect chapters ---
             chapters = analysis_result.get("chapters", [])
             if isinstance(chapters, list):
                 all_chapters.extend(chap for chap in chapters if isinstance(chap, dict))
             elif isinstance(chapters, dict):
                 all_chapters.append(chapters)

             # --- Collect Global Tags (Persons, Actions from LLM) ---
             global_tags_segment = analysis_result.get("globalTags", {})
             if isinstance(global_tags_segment, dict):
                 for tag_type in ["persons", "actions"]:
                     if tag_type not in global_tags_segment or not isinstance(global_tags_segment[tag_type], list):
                         # Ensure the key exists even if empty, prevents key errors later
                         if tag_type not in collected_intervals_by_tag:
                             collected_intervals_by_tag[tag_type] = {}
                         continue

                     for tag_entry in global_tags_segment[tag_type]:
                         if isinstance(tag_entry, dict) and "name" in tag_entry and isinstance(tag_entry.get("timecodes"), list):
                             tag_name = tag_entry["name"]
                             # Initialize list for this tag_name if not present
                             if tag_name not in collected_intervals_by_tag[tag_type]:
                                 collected_intervals_by_tag[tag_type][tag_name] = []

                             for interval in tag_entry["timecodes"]:
                                 if isinstance(interval, dict) and "start" in interval and "end" in interval:
                                     try:
                                         # --- MODIFIED: Ensure using cobra_utils function ---
                                         # The import `from ..cobra_utils import ... convert_string_to_seconds` handles this.
                                         # No change needed here if import is correct.
                                         start_sec = convert_string_to_seconds(interval["start"])
                                         end_sec = convert_string_to_seconds(interval["end"])
                                         # --- END MODIFICATION ---
                                         collected_intervals_by_tag[tag_type][tag_name].append({"start": start_sec, "end": end_sec})
                                         # print(f"DEBUG: Collected {tag_type} interval for '{tag_name}': {start_sec:.3f}s - {end_sec:.3f}s")
                                     except ValueError as e:
                                         # This warning should now only appear for truly unparseable strings
                                         print(f"Warning: Could not convert timestamps for {tag_type} '{tag_name}': {interval}, Error: {e}")
                         else:
                             # Optional: Log if a tag entry is malformed
                             # print(f"Warning: Malformed tag entry skipped in {tag_type}: {tag_entry}")
                             pass # Skip malformed entries

        # --- ADDED DEBUG LOG ---
        try:
            # Use json.dumps for potentially large/nested structures
            collected_intervals_dump = json.dumps(collected_intervals_by_tag, indent=2)
        except TypeError:
            collected_intervals_dump = str(collected_intervals_by_tag) # Fallback if not serializable
        print(f"DEBUG Process Seg - After Collection Loop: collected_intervals_by_tag = {collected_intervals_dump}")
        # --- END ADDED DEBUG LOG ---

        # --- 3. Perform Final Merge and Format Output ---
        print("DEBUG: Starting final merge and formatting...")
        merged_action_summary_tags = {
            "persons": [],
            "actions": [],
            # "objects": [] # Assuming objects might be added later or aren't used currently
        }

        # --- ADDED: Determine clip end time for filtering ---
        clip_end = 0.0 # Default
        if manifest and manifest.segments:
             try:
                  # Use the end time of the last segment
                  clip_end = manifest.segments[-1].end_time if manifest.segments[-1].end_time is not None else 0.0
                  print(f"DEBUG: Determined clip end time for filtering: {clip_end:.3f}s")
             except (IndexError, AttributeError, TypeError):
                  print("Warning: Could not determine clip end time from manifest segments. Using 0.0s.")
        elif manifest and manifest.source_video and manifest.source_video.duration:
             clip_end = manifest.source_video.duration
             print(f"DEBUG: Using source video duration as clip end time for filtering: {clip_end:.3f}s")
        else:
             print("Warning: Cannot determine clip end time. Timestamp filtering might be inaccurate.")
        MAX_DRIFT = 0.5 # seconds
        # --- END ADDED ---

        for tag_type, tags_dict in collected_intervals_by_tag.items():
            if tag_type not in merged_action_summary_tags: continue # Skip if tag type isn't expected in final output

            for tag_name, intervals in tags_dict.items():
                if not intervals: continue # Skip if no intervals collected

                # Sort intervals by start time before merging
                sorted_intervals = sorted(intervals, key=lambda x: x["start"])

                # Merge overlapping/adjacent intervals
                merged_intervals = merge_overlapping(sorted_intervals, max_gap=0.1) # Use correct argument name 'max_gap'

                # --- ADDED DEBUG LOG ---
                print(f"DEBUG Process Seg - Merging '{tag_name}': sorted_intervals = {sorted_intervals}")
                print(f"DEBUG Process Seg - Merging '{tag_name}': merged_intervals (after merge_overlapping) = {merged_intervals}")
                # --- END ADDED DEBUG LOG ---

                # --- ADDED: Filter out intervals outside the clip duration ---
                original_count = len(merged_intervals)
                merged_intervals = [
                    iv for iv in merged_intervals
                    # Check both start and end are within bounds (allow for small drift)
                    if 0.0 <= iv["start"] <= clip_end + MAX_DRIFT
                       and 0.0 <= iv["end"] <= clip_end + MAX_DRIFT
                       # Also ensure start is not nonsensically after end (though merge_overlapping should prevent this)
                       and iv["start"] <= iv["end"] + MAX_DRIFT
                ]
                # --- ADDED DEBUG LOG ---
                print(f"DEBUG Process Seg - Merging '{tag_name}': merged_intervals (after filter) = {merged_intervals}")
                # --- END ADDED DEBUG LOG ---
                filtered_count = original_count - len(merged_intervals)
                if filtered_count > 0:
                     print(f"DEBUG: Filtered out {filtered_count} intervals for tag '{tag_name}' exceeding clip end ({clip_end:.3f}s + {MAX_DRIFT}s drift).")
                # --- END ADDED ---

                # Format for final output ("X.XXXs")
                formatted_timecodes = [
                    {"start": f"{iv['start']:.3f}s", "end": f"{iv['end']:.3f}s"}
                    for iv in merged_intervals
                ]
                # --- ADDED DEBUG LOG ---
                print(f"DEBUG Process Seg - Merging '{tag_name}': formatted_timecodes = {formatted_timecodes}")
                # --- END ADDED DEBUG LOG ---

                if formatted_timecodes: # Only add if there are valid timecodes after merging
                     merged_action_summary_tags[tag_type].append({
                         "name": tag_name,
                         "timecodes": formatted_timecodes
                     })
                     # print(f"DEBUG: Merged {tag_type} '{tag_name}' - {len(formatted_timecodes)} final intervals.")

        # --- ADDED DEBUG LOG ---
        try:
            # Use json.dumps for potentially large/nested structures
            merged_tags_dump = json.dumps(merged_action_summary_tags, indent=2)
        except TypeError:
            merged_tags_dump = str(merged_action_summary_tags) # Fallback if not serializable
        print(f"DEBUG Process Seg - After Merging Loop: merged_action_summary_tags = {merged_tags_dump}")
        # --- END ADDED DEBUG LOG ---

        # --- 4. Generate Final Summary (Optional) ---
        final_summary_content = {} # Stores category, subCategory, description, summary from LLM
        # --- ADDED: Re-extract chapters here to ensure they are included --- 
        all_chapters = []
        for container in enriched_segment_results:
            # --- Modified check for analysisResult key --- 
            analysis_result = container.get("analysis_result") or container.get("analysisResult") or {}
            if analysis_result:
                chapters = analysis_result.get("chapters", [])
                if isinstance(chapters, list):
                    all_chapters.extend(chap for chap in chapters if isinstance(chap, dict)) # Ensure only dicts are added
                elif isinstance(chapters, dict): # Handle case where only one chapter object might be returned directly
                    all_chapters.append(chapters)
        # --- END ADDED --- 
        
        summary_start_time = perf_counter()
        if self.run_final_summary:
            print("DEBUG: Generating final summary...")
            # Prepare context for final summary prompt
            # chapters_context was moved outside this block
            chapters_context_str = json.dumps(all_chapters, indent=2) # Use the collected chapters

            # Extract transcription text if available
            if full_transcription_obj:
                full_transcript_text = get_full_transcript_text(full_transcription_obj.get("recognizedPhrases", []))
            else:
                full_transcript_text = "Transcription data unavailable."

            # Prepare tags context
            tags_context = json.dumps(merged_action_summary_tags, indent=2)

            # Format the final prompt
            summary_prompt_formatted = self.summary_prompt.format(
                asset_categories_list=", ".join(self.ASSET_CATEGORIES),
                summary_length_instruction=env.summary_length_instruction # Use env setting
            )

            # Construct messages for the final summary LLM call
            final_summary_messages = [
                SystemMessage(content=summary_prompt_formatted),
                HumanMessage(content=f"Video Analysis Context:\n\nChapters:\n{chapters_context_str}\n\nTags:\n{tags_context}\n\nFull Transcription:\n{full_transcript_text}\n\nPlease generate the final JSON summary based on all this information.")
            ]

            try:
                # Make the LLM call for the final summary (SYNCHRONOUS)
                # --- Create SYNC LLM client instance dynamically ---
                if env.get_llm_provider() == "azure":
                     # Ensure the correct AzureOpenAI (sync) class is imported
                     from openai import AzureOpenAI # Add sync import if not already present globally
                     llm_client = AzureOpenAI(
                         azure_endpoint=env.vision.endpoint,
                         api_key=env.vision.api_key.get_secret_value(),
                         api_version=env.vision.api_version
                     )
                else: # Assuming default is OpenAI
                     # Ensure the correct OpenAI (sync) class is imported
                     from openai import OpenAI # Add sync import if not already present globally
                     llm_client = OpenAI(api_key=env.vision.api_key.get_secret_value()) # Need base_url if not default OpenAI
                # --- End Client Creation ---

                # --- Use SYNC call (no await) --- 
                response = llm_client.chat.completions.create(
                    model=env.get_llm_model_name(purpose="summary"), # Use appropriate model
                    messages=[m.to_dict() for m in final_summary_messages],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=1024
                )
                # --- Synchronous clients typically don't need an explicit close --- 
                # await llm_client.close() # REMOVE await and close call
                
                summary_result_raw = response.choices[0].message.content or "" # Ensure it's a string
                if not summary_result_raw.strip():
                    print("Warning: Model returned empty content (possibly content-filtered). Using fallback.")
                    # Provide fallback even if empty
                    final_summary_content = {
                        "category": "Unknown",
                        "subCategory": None,
                        "description": "Summary unavailable – model returned no content.",
                        "summary": "-" # Simplified fallback summary
                    }
                else:
                    # Try parsing only if content is not empty
                    final_summary_content = json.loads(summary_result_raw)
                    print(f"DEBUG: Final summary generated successfully. ({perf_counter() - summary_start_time:.2f}s)")

            except json.JSONDecodeError as e: # Keep existing JSON error handling
                print(f"ERROR: Failed to parse JSON from LLM summary response: {e}")
                print(f"LLM Raw Response (first 500 chars): {summary_result_raw[:500]}")
                # Provide a fallback empty summary structure
                final_summary_content = {
                    "category": "Unknown",
                    "subCategory": None,
                    "description": "Summary generation failed (JSON parse error).",
                    "summary": "Could not generate a detailed summary due to a JSON parsing error."
                }
            except Exception as e: # Catch other potential errors during LLM call/processing
                print(f"ERROR: Failed to generate final summary: {e}")
                # Provide a fallback empty summary structure
                final_summary_content = {
                    "category": "Unknown",
                    "subCategory": None,
                    "description": "Summary generation failed.",
                    "summary": "Could not generate a detailed summary due to an error."
                }
        else: # If final summary is skipped
             final_summary_content = {
                 "category": "Not Applicable",
                 "subCategory": None,
                 "description": "Final summary generation was skipped.",
                 "summary": "Final summary generation was skipped."
             }

        # --- 5. Assemble Final Output Structure --- 
        print("DEBUG: Assembling final output structure...")
        # Combine merged tags and final summary into action_summary_content
        action_summary_content = {
            **final_summary_content, # Add category, subCategory, description, summary
            "globalTags": merged_action_summary_tags, # Add merged persons, actions
            "chapters": all_chapters # <-- Ensure chapters are included here
        }

        # --- 6. Add Metadata --- 
        process_end_time_utc = datetime.now(timezone.utc)
        processing_duration_seconds = (process_end_time_utc - process_start_time_utc).total_seconds()

        metadata = {
            "processingStartTimeUTC": process_start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "processingEndTimeUTC": process_end_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "processingDurationSeconds": round(processing_duration_seconds, 3),
            "runtimeSeconds": round(runtime_seconds, 3) if runtime_seconds is not None else None,
            "totalTokens": tokens["total"] if tokens else None,
            "promptTokens": tokens["prompt"] if tokens else None,
            "completionTokens": tokens["completion"] if tokens else None,
            "llmProvider": env.get_llm_provider(),
            "llmModelUsed": env.get_llm_model_name(), # Get default model used across segments
            "llmSummaryModelUsed": env.get_llm_model_name(purpose="summary") if self.run_final_summary else None,
            "asrProvider": env.get_asr_provider(),
            "asrModelUsed": env.get_asr_model_name(),
            "copyrightInfo": parsed_copyright_info, # Add parsed copyright info
            "version": "cobrapy-vX.Y.Z" # Replace with actual version later
        }
        action_summary_content["metadata"] = metadata
        # Optional: Add manifest details if needed
        # action_summary_content["sourceManifest"] = manifest.to_dict() if manifest else None

        print(f"DEBUG: action_summary processing complete. ({processing_duration_seconds:.2f}s)")

        # Return both the detailed transcription and the action summary
        return {
            "transcriptionDetails": transcription_details,
            "actionSummary": action_summary_content,
        }

# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------
def get_full_transcript_text(recognized_phrases: List[Dict]) -> str:
    """Extracts and joins the display text from recognized phrases."""
    return " ".join(
        phrase.get("nBest", [{}])[0].get("display", "")
        for phrase in recognized_phrases
        if phrase and isinstance(phrase, dict) and phrase.get("nBest")
    ).strip()

# Add explicit type hint for VideoManifest to satisfy the forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.video import VideoManifest, Segment
    from ..models.environment import CobraEnvironment

# Make sure SystemMessage and HumanMessage are defined or imported if used in the summary generation
# For standard OpenAI library v1.0+
class SystemMessage:
    def __init__(self, content: str):
        self.role = "system"
        self.content = content
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class HumanMessage:
     def __init__(self, content: str):
        self.role = "user"
        self.content = content
     def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
