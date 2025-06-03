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
