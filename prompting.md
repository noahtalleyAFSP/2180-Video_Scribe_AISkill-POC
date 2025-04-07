# Prompting Strategy (`ActionSummary`)



## Overview

The `ActionSummary` configuration employs a multi-stage approach within the `VideoAnalyzer` (`src/cobrapy/video_analyzer.py`) to analyze video segments. For each segment defined during preprocessing, the analysis typically involves:

1.  **Chapter Generation:** A single call to the LLM to generate a summary chapter for the entire segment.
2.  **Tag Generation (Chunked):** Multiple calls to the LLM, processing the segment in smaller, potentially overlapping time chunks, to identify persons, objects, and actions with more precise timecodes.
3.  **Aggregation:** Results from the chapter and tag generation steps are combined and processed by `ActionSummary.process_segment_results` to create the final structured output (`_ActionSummary.json`) and update the video manifest.

Distinct system prompts and formatting "lenses" are used for chapter and tag generation to clearly define the expected task and output format for the LLM. The `_generate_segment_prompt` method in `VideoAnalyzer` dynamically selects and formats these prompts based on the task and available context (like custom definition lists).

## 1. Chapter Generation

**Goal:** Generate a single, concise summary object describing the key aspects of an entire video segment.

**Prompts Used:**

*   **Base System Prompt:** `ActionSummary.system_prompt_chapters`
*   **Formatting Lens:** `ActionSummary.system_prompt_lens_chapters`

**Base Prompt (`system_prompt_chapters`):**

This prompt sets the stage for the LLM, defining its role (VideoAnalyzerGPT focused on summarizing) and outlining the core requirements.

```python
# Located in src/cobrapy/analysis/action_summary.py
system_prompt_chapters: str = (
    """You are VideoAnalyzerGPT, focused on summarizing video segments. Your goal is to generate a single, detailed chapter summary for the provided segment, using precise timestamps based *only* on the frame data provided.

**CRITICAL INSTRUCTIONS - CHAPTERS ONLY - READ CAREFULLY:**

1.  **JSON STRUCTURE:** You MUST return a valid JSON object with ONLY the top-level key: "chapters".
    *   "chapters": An array containing EXACTLY ONE chapter object describing the current segment. **DO NOT include a "global_tags" key.**

2.  **EXACT OUTPUT FORMAT (Chapters Only):**
    ```json
    {
      "chapters": [
        {
          "start": "45.123s",
          "end": "75.456s",
          "sentiment": "neutral",
          "emotions": ["emotion1", "emotion2"],
          "theme": "short theme",
          "summary": "Detailed, descriptive summary..."
        }
      ]
    }
    ```

3.  **TIMESTAMP PRECISION & ACCURACY (Chapters):**
    *   Use segment absolute start ({start_time}s) and end ({end_time}s) times.
    *   Format: "0.000s".

4.  **SUMMARY CONTENT:** Describe setting, visuals, actions, audio context for *this specific segment*.
"""
)
```

**Key Instructions:**

*   **Role:** VideoAnalyzerGPT, summarizing segments.
*   **Output:** Must be JSON, containing only a `chapters` key with a single chapter object.
*   **Fields:** Specifies the required fields within the chapter object (`start`, `end`, `sentiment`, `emotions`, `theme`, `summary`). Note that `transcription` was intentionally removed from the chapter object to avoid redundancy, as full transcription data is available elsewhere.
*   **Timestamps:** Must be based on the provided segment start/end times and formatted as "X.XXXs".
*   **Content:** The summary should cover visuals, actions, and audio context for the specific segment.

**Formatting Lens (`system_prompt_lens_chapters`):**

This template is formatted with segment-specific information before being added to the user message sent to the LLM.

```python
# Located in src/cobrapy/analysis/action_summary.py
system_prompt_lens_chapters: str = (
    """Analyze the segment from {start_time}s to {end_time}s to generate a chapter summary. The segment contains:
- {number_of_frames} frames (with timestamps)
- Possibly an audio transcription: {transcription_context}

Remember, your output MUST be valid JSON containing ONLY the 'chapters' key, with exactly one chapter object inside the array, describing this segment.
"""
)
```

**Formatted Variables:**

*   `{start_time}`: Start time of the segment (e.g., "45.000").
*   `{end_time}`: End time of the segment (e.g., "75.000").
*   `{number_of_frames}`: Number of frames being provided for analysis.
*   `{transcription_context}`: A snippet of the full video's transcription for context.

**Process:** The `_generate_segment_prompt` function formats the lens with actual values and combines it with the base prompt and frame image data to create the final message list sent to the LLM for chapter generation.

## 2. Tag Generation (Chunked)

**Goal:** Identify persons, objects, and actions occurring within smaller time chunks of a segment, providing detailed timecodes based *only* on the frames within that chunk.

**Why Chunking?**

*   **Context Limits:** LLMs have limits on the number of images/tokens they can process at once. Chunking keeps the input manageable.
*   **Timestamp Accuracy:** Analyzing smaller chunks allows the LLM to provide more granular and accurate start/end times for tags based on when they appear/disappear within that specific chunk's frames.

**Prompts Used:**

*   **Base System Prompts (Dynamically Chosen):**
    *   `ActionSummary.system_prompt_tags` (Standard)
    *   `ActionSummary.system_prompt_tags_custom_people` (Used if a custom people list is provided)
*   **Formatting Lens:** `ActionSummary.system_prompt_lens_tags`

**Dynamic Prompt Selection:** The `_generate_segment_prompt` method checks if a custom people list (`self.peoples_list`) was loaded by the `VideoAnalyzer`.
*   If **YES**, it uses `system_prompt_tags_custom_people`.
*   If **NO**, it uses the standard `system_prompt_tags`.

### 2.1 Standard Tagging (`system_prompt_tags`)

This is used when no specific custom people definitions are provided.

```python
# Located in src/cobrapy/analysis/action_summary.py
system_prompt_tags: str = (
    """You are VideoAnalyzerGPT, specialized in identifying and tagging entities in video frames. Your task has THREE DISTINCT PARTS that MUST ALL be completed:

1) Identify and tag PERSONS using descriptive names based on appearance.
2) Identify and tag relevant OBJECTS visible in the frames.
3) Identify and tag significant ACTIONS being performed.

**IMPORTANT: You MUST complete ALL THREE parts of this task!**

**PERSONS TAGGING TASK (Standard):**
- Identify clearly visible individuals.
- **Naming Rule:** The 'name' MUST describe the person's **appearance and static features ONLY**. DO NOT include actions (walking, sitting) in the person's name. Tag actions separately.
- Use descriptive, consistent identifiers based *only* on appearance (e.g., "Woman in red dress", "Man in grey suit", "Child with blue backpack"). Do NOT add arbitrary IDs.

**OBJECTS TAGGING TASK: (MANDATORY - DO NOT SKIP)**
- Identify relevant physical items (e.g., "Laptop", "Desk", "Window", "Briefcase", "Chair").
- **Consistency is Key:** Use specific, consistent names. Check 'Known Objects' list below and reuse names where appropriate. Avoid minor variations for the same object (e.g., prefer "Telephone" over "Phone", "Handset").

**ACTIONS TAGGING TASK: (MANDATORY - DO NOT SKIP)**
- Identify significant activities (e.g., "Walking", "Sitting", "Typing", "Talking", "Holding object").
- **Consistency is Key:** Use clear, concise action verbs. Check 'Known Actions' list below and reuse names where appropriate. **Crucially, avoid creating synonyms for the same core action.** For example, if someone is interacting with a telephone, prefer a single consistent tag like "Using telephone" instead of multiple variations like "Holding telephone", "Talking on phone", "Picking up telephone" unless the specific phase of the action is critical and distinct. Choose the most representative and consistent term.

**TIMESTAMP REQUIREMENTS:**
- Base ALL timecodes (`start`, `end`) **STRICTLY** on the absolute timestamps of the {number_of_frames} frames provided for THIS chunk ({start_time}s to {end_time}s).
- Timecodes must represent the interval the item/person/action is *visibly present* or *actively occurring* AS OBSERVED IN THE **PROVIDED FRAMES**.
- Do NOT automatically extend to chunk boundaries unless the frames clearly show continuous presence.
- Format: "0.000s".

**OUTPUT FORMAT:**
    ```json
    {
      "global_tags": {
        "persons": [ /* {"name": "Descriptive Name", "timecodes": [{"start": "X.XXXs", "end": "Y.YYYs"}]} */ ],
        "objects": [ /* {"name": "Object Name", "timecodes": [{"start": "X.XXXs", "end": "Y.YYYs"}]} */ ],
        "actions": [ /* {"name": "Action Name", "timecodes": [{"start": "X.XXXs", "end": "Y.YYYs"}]} */ ]
      }
    }
    ```

**CRITICAL:** All three tag categories MUST be populated.** If nothing is found for a category, include an empty array `[]`. Ensure timestamp accuracy based ONLY on provided frames for this chunk. Prioritize consistent naming for objects and actions.
"""
)
```

**Key Instructions:**

*   **Role:** VideoAnalyzerGPT, tagging entities.
*   **Mandatory Tasks:** MUST identify Persons, Objects, AND Actions.
*   **Person Naming:** Describe appearance ONLY (e.g., "Man in grey suit"), not actions.
*   **Object/Action Naming:** Emphasizes consistency, avoiding synonyms, and potentially reusing names from known lists provided via the lens.
*   **Timestamps:** CRITICAL - must be based *only* on the frames provided for the specific chunk, within the chunk's start/end times. Do not extrapolate.
*   **Output:** JSON with a single `global_tags` key, containing `persons`, `objects`, and `actions` arrays (even if empty).

### 2.2 Custom People Tagging (`system_prompt_tags_custom_people`)

This prompt is activated when a `peoples_list_path` is provided to the `VideoAnalyzer`, indicating custom definitions and rules should be used, particularly for person tagging (e.g., compliance checks).

```python
# Located in src/cobrapy/analysis/action_summary.py
system_prompt_tags_custom_people: str = (
    """You are VideoAnalyzerGPT, a specialist in frame-by-frame analysis using STRICT custom definitions. Your task has THREE DISTINCT PARTS that MUST ALL be completed:

1) Identify and tag PERSONS based on compliance with safety requirements using ONLY the labels provided below.
2) Identify and tag SAFETY EQUIPMENT and OBJECTS visible in the frames (MANDATORY).
3) Identify and tag significant ACTIONS being performed (MANDATORY).

**IMPORTANT: You MUST complete ALL THREE parts of this task!**

**PERSONS TAGGING TASK (Custom Compliance):**
- Examine EACH individual in EVERY frame provided for this chunk.
- **EXCLUSIVE NAMING:** Use ONLY the labels from 'Custom Person Definitions' below ('Compliant Construction Worker', 'Non-Compliant Construction Worker'). DO NOT invent descriptions.
- **RIGOROUS COMPLIANCE CHECK:**
    - **Default to Non-Compliant:** Assume Non-Compliant unless proven otherwise *in these specific frames*.
    - **Condition for 'Compliant':** MUST visually confirm BOTH safety helmet AND high-visibility jacket/vest in **all provided frames** where the person is visible in this chunk.
    - **Condition for 'Non-Compliant':** Tag as Non-Compliant IF they lack EITHER item in **any provided frame** where visible in this chunk.
- **Timestamping Compliance:** Timecode covers interval person is visible *in these frames* AND maintains that status.

**OBJECTS TAGGING TASK: (MANDATORY - DO NOT SKIP)**
- You MUST identify visible safety equipment: "Safety Helmet", "High-Visibility Jacket".
- Also tag other relevant objects using 'Custom Object Definitions' if provided, otherwise generic names.

**ACTIONS TAGGING TASK: (MANDATORY - DO NOT SKIP)**
- Identify significant activities using 'Custom Action Definitions' if provided, otherwise generic verbs.

**TIMESTAMP REQUIREMENTS:**
- Base ALL timecodes (`start`, `end`) **STRICTLY** on the absolute timestamps of the {number_of_frames} frames provided for THIS chunk ({start_time}s to {end_time}s).
- Timecodes represent observed duration *in these frames*. Do not guess.
- Format: "0.000s".

**OUTPUT FORMAT:**
```json
{
  "global_tags": {
    "persons": [ /* {"name": "Compliant/Non-Compliant...", "timecodes": [...] } */ ],
    "objects": [ /* {"name": "Safety Helmet", ...}, {"name": "Object Name", ...} */ ],
    "actions": [ /* {"name": "Action Name", ...} */ ]
  }
}
```
**CRITICAL: All three tag categories MUST be populated.** Ensure timestamp accuracy based ONLY on provided frames. Include empty arrays `[]` if nothing found for a category.
"""
)

```

**Key Differences & Instructions:**

*   **Role:** Specialist using STRICT custom definitions.
*   **Person Naming:** EXCLUSIVE - uses ONLY labels provided via the lens (e.g., "Compliant Construction Worker").
*   **Compliance Logic:** Detailed, rigorous rules for determining compliance based on visual evidence (helmet, vest) *within the provided frames*. Assumes non-compliant by default.
*   **Object Tagging:** Explicitly mandates tagging specific safety equipment ("Safety Helmet", "High-Visibility Jacket").
*   Other rules (Mandatory Tasks, Timestamps, Output Format) remain largely the same.

### 2.3 Tagging Lens (`system_prompt_lens_tags`)

This template provides crucial context to the LLM for the tagging task, regardless of whether the standard or custom people prompt is used.

```python
# Located in src/cobrapy/analysis/action_summary.py
system_prompt_lens_tags: str = (
    """Analyze the time chunk from {start_time}s to {end_time}s to identify tags. The chunk contains:
- {number_of_frames} frames (with timestamps)
- Overall Segment Transcription Context: {transcription_context}

{explicit_object_reminder}
{explicit_action_reminder}

Remember, your output MUST be valid JSON containing ONLY the 'global_tags' key, with 'persons', 'actions', and 'objects' arrays (populate all three!). Ensure all tag timecodes are strictly within {start_time}s and {end_time}s and based ONLY on the provided frames.

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
```

**Formatted Variables:**

*   `{start_time}`, `{end_time}`, `{number_of_frames}`, `{transcription_context}`: Same as chapter lens, but specific to the current *chunk*.
*   `{people_definitions}`, `{object_definitions}`, `{action_definitions}`: These are dynamically generated strings formatted from the JSON files provided via `peoples_list_path`, `objects_list_path`, `actions_list_path`. The `_generate_segment_prompt` method uses a helper (`format_definitions`) to create a readable list of labels and descriptions for the LLM. If no list is provided for a category, it defaults to "No custom definitions provided."
*   `{known_objects}`, `{known_actions}`: JSON strings representing lists of object/action names extracted from the *custom definition lists* (if provided). This helps the LLM maintain consistency by suggesting preferred names.
*   `{explicit_object_reminder}`, `{explicit_action_reminder}`: These placeholders can inject specific, urgent reminders, often used in conjunction with custom definitions (e.g., to emphasize the need to tag safety equipment). Currently hardcoded within `_generate_segment_prompt` when custom people lists are active.

**Process:** `_generate_segment_prompt` selects the appropriate base system prompt (`system_prompt_tags` or `system_prompt_tags_custom_people`), formats the `system_prompt_lens_tags` with context (including formatted definitions and known tags), and combines them with the chunk's frame data for the LLM call.

## 3. Aggregation (`process_segment_results`)

After the LLM returns results for chapters and all tag chunks, the `process_segment_results` method within `ActionSummary` takes over. Its key responsibilities include:

*   Collecting all chapter objects.
*   Collecting all tags (persons, objects, actions) from all chunks.
*   **Validating and Clamping Timestamps:** Ensuring all timestamps (from both chapters and tags) fall within their respective segment boundaries and correcting/clamping them if necessary.
*   **Merging Tag Timestamps:** Aggregating the potentially fragmented time points for each unique tag name (gathered across multiple chunks) into continuous intervals using the `_aggregate_tag_timestamps` helper function. This function handles merging gaps based on defined thresholds.
*   **Structuring Final Output:** Creating the final `actionSummary` JSON structure containing the transcript, language, chapters, detailed speech segments, aggregated tags (person, action, object with numeric timestamps), and thumbnail paths.


### Sending the images base64 encoding.
  # This loop processes the selected frames
            for frame_idx in sorted(list(selected_frames_indices)):
                # This block is indented under the 'for' loop
                frame_path = frames_to_use[frame_idx]
                frame_time = times_to_use[frame_idx]
                if os.path.exists(frame_path):
                    try:
                        # Indented under 'try'
                        base64_image = encode_image_base64(frame_path)
                        user_content_list.append({"type": "text", "text": f"\nFrame at {frame_time:.3f}s:"})
                        user_content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                        })
                        encoded_frame_count += 1
                    except Exception as e:
                        # Indented under 'except'
                        print(f"Warning: Could not encode frame {frame_path}. Error: {e}")
                        user_content_list.append({"type": "text", "text": f"\nFrame at {frame_time:.3f}s: [Error encoding image]"})
                else:
                    # Indented under 'else' associated with 'if os.path.exists...'
                    user_content_list.append({"type": "text", "text": f"\nFrame at {frame_time:.3f}s: [Image file missing]"})

            # This 'if' aligns with the 'for' loop above it
            if encoded_frame_count == 0 and number_of_frames_available > 0:
                print(f"Warning: No frames encoded for chunk despite {number_of_frames_available} available.")



## Conclusion

This prompting strategy separates concerns by using distinct prompts for chapter summarization and detailed tag generation. It leverages chunking for manageable input and timestamp accuracy during tagging. The dynamic selection of prompts based on custom lists allows for flexibility in analysis tasks (e.g., standard description vs. compliance checking). Finally, post-processing merges and refines the LLM outputs into a structured and coherent final result.
