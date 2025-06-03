# Recent Changes to Video Scribe

This document is an outline of the most recent iterations to Video Scribe.

## Key Feature Updates:

### 1. YOLO-Based Object and Person Tracking

-   **Initial Detection & Tracking:** The pipeline now integrates YOLO (e.g., `yolo11x.pt` with ByteTrack via `src/cobrapy/pipeline/step0_yolo_track.py`) for robust initial detection and tracking of objects and persons throughout the video.
    -   This process generates `raw_yolo_tags` which include frame-level bounding boxes, timestamps, and preliminary tracker IDs for each detected entity.
-   **Output:** Raw tracking data is stored in `{video_name}_manifest_raw_yolo_tags.json` and within the main `_video_manifest.json`.

### 2. Thumbnail Generation and Conditional Storage

-   **Representative Thumbnails:**
    -   **Raw Tracks:** For each raw track identified by YOLO, a representative thumbnail (a cropped image of the object/person) is generated (e.g., from its first appearance). These are stored in the `yolo_thumbs_raw/` directory.
    -   **Split/Refined Tracks:** When tracks are split due to scene cuts and re-identification (`step0b_reid_split_tracks.py`) or further refined across scenes (`track_refiner.py`), new thumbnails are generated for these (potentially new or modified) track segments. These are stored in `yolo_thumbs_reid_split/` and `refined_track_thumbs/` respectively.
    -   The paths to these thumbnails (`thumb` or `representative_thumb_path`) are included in the corresponding track data within the JSON outputs.
-   **Conditional Saving (New):**
    -   A new command-line argument `--skip-thumbnail-saving` has been introduced.
    -   When this flag is set, the generation and saving of all YOLO-related thumbnails (raw, split, and refined) are skipped. This can be useful for reducing disk space usage or speeding up processing if thumbnails are not required.
    -   By default (if the flag is not used), thumbnails are generated and saved as before.

### 3. Person Re-Identification (Re-ID) Across Scene Cuts

-   **Verification Across Cuts (`src/cobrapy/pipeline/step0b_reid_split_tracks.py`):**
    -   When a tracked person's appearance spans a detected scene cut, the system now performs a verification step.
    -   Image crops of the person are taken from just before the cut and just after the cut.
    -   These crops are sent to a Vision LLM (e.g., GPT-4 Vision) which is asked to determine if they depict the same individual.
    -   **Logic:**
        -   If the LLM confirms it's the **same person**, the track ID is maintained across the cut.
        -   If the LLM determines it's a **different person** (or is not confident enough), the original track is split at the cut. The portion of the track after the cut is assigned a new, unique ID. A new description may be generated for this new ID if necessary.
    -   This process helps in maintaining more accurate and continuous person tracking, even when individuals cross sharp scene transitions.
    -   The system previously included a face embedding comparison as a primary Re-ID method, but this has been updated to prioritize the LLM-based visual comparison for greater robustness, especially when faces are not clearly visible.
-   **Track Refinement (`src/cobrapy/track_refiner.py`):**
    -   This (optional but recommended) step further refines tracks, especially for persons, across multiple scenes. It uses an LLM to link track segments that might belong to the same individual but were broken by scene changes or occlusions, assigning a more persistent `refined_track_id`.

These updates aim to improve the accuracy of object and person tracking, provide more robust handling of identities across scene changes, and offer greater control over the generated artifacts.

### Granular Action Extraction (`video_analyzer.py`)

To provide a more detailed understanding of activities within each segment, a dedicated action extraction step has been implemented within `VideoAnalyzer._extract_actions_per_segment_async`. This process works as follows:

1.  **Chunking Frames:** The frames within a given segment are divided into smaller, manageable chunks (e.g., 10 frames per chunk, defined by `ACTION_EXTRACTION_FRAME_CHUNK_SIZE`).
2.  **Dedicated LLM Prompt:** Each chunk of frames is sent to the LLM with a specialized system prompt (`VideoAnalyzer.SYSTEM_PROMPT_ACTION_EXTRACTION`). This prompt instructs the model to identify all distinct actions performed by persons or objects within that small set of frames and their corresponding start/end timestamps *within that chunk*.
3.  **JSON Output:** The LLM is expected to return a JSON object containing a list of actions, each with a `classDescription` (e.g., "Man opening door") and precise `timecodes` relative to the video's start.
4.  **Aggregation:** The actions extracted from all chunks within a segment are collected.
5.  **Integration:** These fine-grained actions are then stored in the `segment.extracted_actions` list for each `Segment` object and are also aggregated into the final `_ActionSummary.json` under `globalTags.actions` by `ActionSummary.process_segment_results`. This means the final actions list combines actions identified by the main segment analysis LLM call and this more granular, chunk-based extraction.

This approach allows for the detection of shorter, more specific actions that might be missed or generalized by a single LLM call analyzing an entire segment's worth of frames at once. The token usage for these calls is tracked under the `action_extraction_..._tokens` category.

### Prompt Engineering for YOLO Descriptions and Re-ID

To enhance the understanding and linking of entities detected by YOLO and Re-ID processes, the system leverages Large Language Models (LLMs) with carefully constructed prompts. These prompts guide the LLM to provide detailed descriptions or make comparisons.

#### 1. YOLO Tag Description Prompts

**Purpose:** To obtain a concise, human-readable textual description for an object or person detected by YOLO, based on its cropped image. This description is more informative than a generic class label (e.g., "car" becomes "blue sedan"). This is primarily handled in the `_describe_yolo_tags_with_gpt` method within `src/cobrapy/video_analyzer.py`.

**Example Prompt Structure:**

The LLM is typically provided with a system message and a user message containing the image and instructions.

*   **System Message:**
    ```json
    {
      "role": "system",
      "content": "You are a vision assistant skilled at providing concise object descriptions."
    }
    ```

*   **User Message:**
    The user message is a multi-part message including text instructions and the image to analyze.
    ```json
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Object class: '{tag_class}'. Provide a concise, specific, descriptive label for this item. Focus on its material, type, color, or key distinguishing features. Do not use introductory phrases or refer to 'the image'. For example, if the class is 'bag', a good label might be 'red leather handbag' or 'canvas backpack'. If class is 'car', 'blue sedan' or 'vintage sports car'. Output only the descriptive label itself, nothing else."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,{crop_base64}",
            "detail": "low"
          }
        }
      ]
    }
    ```
    *   `{tag_class}`: This is replaced with the generic class detected by YOLO (e.g., "car", "person").
    *   `{crop_base64}`: This is replaced with the Base64 encoded string of the cropped image showing the detected entity.

#### 2. Person/Object Re-Identification (Re-ID) Prompts

**Purpose:** To determine if two different tracklets (sequences of detections over time, potentially separated by cuts or occlusions) belong to the same person or object. This is crucial for maintaining consistent identities throughout the video. Re-ID prompting occurs in modules like `src/cobrapy/pipeline/step0b_reid_split_tracks.py` and the `track_refiner.py` module.

**Conceptual Example:**

The actual prompts for Re-ID can be complex and vary based on the specific strategy (e.g., comparing two tracklets, describing a single tracklet for later feature matching). Below is a *conceptual example* of a prompt designed to ask an LLM to compare two tracklets:

*   **System Message (Conceptual):**
    ```json
    {
      "role": "system",
      "content": "You are an expert visual analyst specializing in comparing entities across different image sets to determine identity."
    }
    ```

*   **User Message (Conceptual):**
    ```json
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "You are given two sets of images: Tracklet A and Tracklet B. Each tracklet represents an entity detected in a video. Tracklet A contains images from time [start_A]s to [end_A]s. Tracklet B contains images from time [start_B]s to [end_B]s. \n\nBased on the visual evidence in ALL provided images for both tracklets, determine if Tracklet A and Tracklet B represent the same entity. Consider appearance, clothing (if applicable), and any persistent unique features. Pay attention to subtle details. \n\nRespond with ONLY a JSON object with the following keys: \n- \"is_match\": boolean (true if they are the same entity, false otherwise). \n- \"confidence\": string (your confidence in the decision: \"high\", \"medium\", or \"low\"). \n- \"reasoning\": string (a brief explanation for your decision, highlighting key visual similarities or differences)."
        },
        {
          "type": "text",
          "text": "Images for Tracklet A:"
        },
        // Multiple image_url parts for Tracklet A crops
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,{tracklet_A_image1_base64}", "detail": "high" }
        },
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,{tracklet_A_image2_base64}", "detail": "high" }
        },
        // ... more images for Tracklet A
        {
          "type": "text",
          "text": "Images for Tracklet B:"
        },
        // Multiple image_url parts for Tracklet B crops
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,{tracklet_B_image1_base64}", "detail": "high" }
        },
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,{tracklet_B_image2_base64}", "detail": "high" }
        }
        // ... more images for Tracklet B
      ]
    }
    ```
    *   Placeholders like `{tracklet_A_image1_base64}` would be replaced with the Base64 encoded images from the respective tracklets.
    *   The instruction to use `high` detail for images is important for Re-ID to catch subtle visual cues.

This approach allows the system to leverage the nuanced understanding of LLMs for tasks that go beyond simple object detection, leading to richer and more accurate video analysis.

