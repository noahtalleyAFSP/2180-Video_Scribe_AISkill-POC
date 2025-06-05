import base64
import cv2
import json
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple
import logging

from .models.environment import CobraEnvironment
from .models.video import Segment # Assuming Segment objects are passed
# If manifest_segments is just a list of dicts, we might not need the Segment import directly
from .cobra_utils import _pad_bbox, get_frame_crop_base64 # IMPORT FROM UTILS

# --- Configuration Defaults (can be made configurable later) ---
REID_LLM_MODEL = "gpt-4-vision-preview" # Or whatever model is appropriate from env
REID_LLM_MAX_TOKENS = 300
DEFAULT_THUMB_DIR_NAME = "refined_track_thumbs"
CROP_PADDING_PERCENTAGE = 0.10 # 10% padding for crops - Keep for this module if used directly or pass to new util

# --- Helper Functions ---

def _ensure_dir(directory_path: Path):
    """Create a directory path safely – if a regular file of the same
    name already exists we fall back to its parent and warn."""
    if directory_path.exists() and directory_path.is_file():
        logging.warning("%s is a file; using its parent for dir ops", directory_path)
        directory_path = directory_path.parent
    directory_path.mkdir(parents=True, exist_ok=True)


def get_person_crop_at_boundary(
    video_path: str,
    track_frame_bboxes: List[Dict[str, Any]], # List of {"frame_number", "timestamp", "bbox"}
    boundary_time: float,
    position: str # "before" or "after"
) -> Tuple[Optional[str], Optional[float], Optional[int]]: # (base64_string, frame_timestamp, frame_number)
    """
    Finds the frame_bbox in track_frame_bboxes closest to boundary_time based on position.
    Returns: (base64_crop, actual_frame_timestamp, actual_frame_number) or (None, None, None).
    """
    candidate_frame_info = None
    min_time_diff = float('inf')

    for fb in track_frame_bboxes:
        timestamp = fb["timestamp"]
        time_diff = abs(timestamp - boundary_time)

        if position == "before":
            if timestamp <= boundary_time:
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    candidate_frame_info = fb
                elif time_diff == min_time_diff: # Prefer later frame if equally close before boundary
                    if candidate_frame_info is None or fb["timestamp"] > candidate_frame_info["timestamp"]:
                         candidate_frame_info = fb
        elif position == "after":
            if timestamp >= boundary_time:
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    candidate_frame_info = fb
                elif time_diff == min_time_diff: # Prefer earlier frame if equally close after boundary
                     if candidate_frame_info is None or fb["timestamp"] < candidate_frame_info["timestamp"]:
                        candidate_frame_info = fb
        
    if not candidate_frame_info:
        # print(f"[DEBUG] No suitable frame found for boundary_time {boundary_time:.3f}s (position: {position})")
        return None, None, None

    # print(f"[DEBUG] Chosen frame for boundary {boundary_time:.3f}s (pos: {position}): ts={candidate_frame_info['timestamp']:.3f}s, frame_num={candidate_frame_info['frame_number']}")
    
    base64_crop = get_frame_crop_base64(
        video_path,
        candidate_frame_info["frame_number"],
        candidate_frame_info["bbox"],
        crop_padding_percentage=CROP_PADDING_PERCENTAGE # Pass the constant
    )
    if base64_crop:
        return base64_crop, candidate_frame_info["timestamp"], candidate_frame_info["frame_number"]
    return None, None, None


def generate_thumbnail_for_refined_track(
    video_path: str,
    segment_frame_bboxes: List[Dict[str, Any]], # bboxes for this refined segment
    output_dir_refined_thumbs: Path,
    refined_track_id: str,
    skip_thumbnail_saving: bool = False
) -> Optional[str]:
    """
    Generates and saves a thumbnail for a refined track segment.
    Selects a representative frame (e.g., middle one), gets crop, saves it.
    Returns path to saved thumbnail or None.
    """
    if skip_thumbnail_saving:
        # print(f"[DEBUG] Skipping thumbnail generation for refined track {refined_track_id} due to flag.")
        return None
        
    if not segment_frame_bboxes:
        return None

    # Ensure output directory exists (caller should ideally manage this, but good for robustness)
    if not output_dir_refined_thumbs.exists():
        try:
            output_dir_refined_thumbs.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir_thumb:
            print(f"[ERROR] generate_thumbnail_for_refined_track: Failed to create thumb dir {output_dir_refined_thumbs}: {e_mkdir_thumb}")
            return None

    # Select middle frame of the segment for the thumbnail
    middle_idx = len(segment_frame_bboxes) // 2
    thumb_frame_info = segment_frame_bboxes[middle_idx]

    base64_crop = get_frame_crop_base64(
        video_path,
        thumb_frame_info["frame_number"],
        thumb_frame_info["bbox"],
        crop_padding_percentage=CROP_PADDING_PERCENTAGE # Pass the constant
    )

    if base64_crop:
        try:
            thumb_path = output_dir_refined_thumbs / f"{refined_track_id}.jpg"
            img_data = base64.b64decode(base64_crop)
            with open(thumb_path, "wb") as f:
                f.write(img_data)
            return str(thumb_path.resolve())
        except Exception as e:
            print(f"[ERROR] generate_thumbnail_for_refined_track: Failed to save thumb {refined_track_id}: {e}")
            return None
    return None


async def _call_llm_for_reid(env: CobraEnvironment, crop_before_base64: str, crop_after_base64: str, client, prompt_log: Optional[list] = None) -> Tuple[bool, int, int, int]:
    """
    Calls the LLM for visual comparison.
    Returns: (is_same_person, prompt_tokens, completion_tokens, image_tokens)
    is_same_person defaults to False on error. Tokens default to 0 on error.
    client: Pre-initialized AsyncAzureOpenAI client.
    """
    system_prompt = '''You are an AI assistant specializing in visual comparison. You will be given two images, each a cropped image of a person. Your task is to determine if the two images show the SAME person or DIFFERENT people.
Respond ONLY in JSON format with a single key "same_person" which is a boolean (true if the same person, false otherwise). Example: {"same_person": true}'''

    user_messages_content = [
        {"type": "text", "text": "Image 1 (person before potential scene cut):"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_before_base64}", "detail": "low"}},
        {"type": "text", "text": "Image 2 (person after potential scene cut):"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_after_base64}", "detail": "low"}},
        {"type": "text", "text": "Are these two images showing the same person? Consider facial features, clothing, and context if visible. Respond ONLY with JSON: {\"same_person\": boolean}"}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_messages_content}
    ]
    if prompt_log is not None:
        prompt_log.append({
            "type": "reid_compare",
            "context": {"crop_before": bool(crop_before_base64), "crop_after": bool(crop_after_base64)},
            "messages": messages
        })
    prompt_tokens, completion_tokens, image_tokens = 0, 0, 0 # Initialize token counts

    try:
        chat_completion = await client.chat.completions.create(
            model=env.vision.deployment or REID_LLM_MODEL,
            messages=messages,
            max_tokens=REID_LLM_MAX_TOKENS,
            temperature=0.0
        )
        response_content = chat_completion.choices[0].message.content

        if chat_completion.usage:
            prompt_tokens = chat_completion.usage.prompt_tokens or 0
            completion_tokens = chat_completion.usage.completion_tokens or 0
        
        # Estimate image tokens (simplified for this specific call as VideoAnalyzer's estimate_image_tokens is not directly available here)
        # This is a basic estimate. A more precise one would replicate estimate_image_tokens from VideoAnalyzer.
        # Assuming GPT4O_BASE = 85 tokens for a low-detail image.
        image_tokens = 85 * 2 # Two low-detail images

        json_response_str = response_content
        if response_content.strip().startswith("```json"):
            json_response_str = response_content.strip()[7:-3].strip()
        elif response_content.strip().startswith("```"):
             json_response_str = response_content.strip()[3:-3].strip()

        parsed_response = json.loads(json_response_str)
        is_same = parsed_response.get("same_person", False)
        return bool(is_same), prompt_tokens, completion_tokens, image_tokens
    except json.JSONDecodeError as e:
        # print(f"[ERROR] _call_llm_for_reid: JSONDecodeError: {e}. Response: {response_content}")
        # If parsing fails, we still got tokens from the API call.
        return False, prompt_tokens, completion_tokens, image_tokens
    except Exception as e:
        print(f"[ERROR] _call_llm_for_reid: LLM call failed: {e}")
        return False, 0, 0, 0 # Default to 0 tokens on full call failure


def _get_overlapping_scene_ids(track_start: float, track_end: float, manifest_segments: List[Segment]) -> List[str]:
    """Identifies scene IDs that overlap with the given track timespan."""
    overlapping_ids = []
    for seg in manifest_segments:
        # Check for overlap: max(start1, start2) < min(end1, end2)
        # Assuming seg.id is available and is the scene ID. If not, seg.segment_name or similar.
        scene_id_to_add = getattr(seg, 'id', None) or getattr(seg, 'segment_name', None)
        if scene_id_to_add is None: # Fallback if no suitable ID found
             scene_id_to_add = f"scene_at_{seg.start_time:.2f}"

        if max(track_start, seg.start_time) < min(track_end, seg.end_time):
            overlapping_ids.append(str(scene_id_to_add)) 
    return overlapping_ids


def _filter_frame_bboxes(
    all_frame_bboxes: List[Dict[str, Any]],
    segment_start_time: float,
    segment_end_time: float
) -> List[Dict[str, Any]]:
    """Filters frame_bboxes to include only those within the refined segment's timespan."""
    return [
        fb for fb in all_frame_bboxes
        if segment_start_time <= fb["timestamp"] <= segment_end_time
    ]


async def refine_yolo_tracks_across_scenes(
    manifest_segments: List[Segment], 
    manifest_raw_yolo_tags: List[Dict[str, Any]],
    env: CobraEnvironment,
    video_path: str,
    output_dir: str, 
    async_llm_client,
    skip_thumbnail_saving: bool = False
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Refines raw YOLO tracks, especially for 'person' class, by performing
    re-identification across scene cuts using an LLM.
    Returns a tuple: (refined_tracks_output, total_reid_prompt_tokens, total_reid_completion_tokens, total_reid_image_tokens)
    """
    print(f"Starting YOLO track refinement. Raw tracks: {len(manifest_raw_yolo_tags)}, Scenes: {len(manifest_segments)}")
    refined_tracks_output: List[Dict[str, Any]] = []
    total_reid_prompt_tokens = 0
    total_reid_completion_tokens = 0
    total_reid_image_tokens = 0
    next_refined_track_numeric_id = 1
    
    out_dir = Path(output_dir).resolve()                     # run-folder
    video_manifest_path = out_dir / "_video_manifest.json"   # file, kept for later

    refined_thumbs_dir = out_dir / "refined_track_thumbs"    # <-- NEW location
    if not skip_thumbnail_saving:
        _ensure_dir(refined_thumbs_dir)
    # else: print(f"[DEBUG] Skipping creation of refined_thumbs_dir due to flag.")

    for raw_yolo_track in manifest_raw_yolo_tags:
        track_class = raw_yolo_track["class"]
        original_yolo_id = raw_yolo_track["id"]
        all_track_frame_bboxes = raw_yolo_track["frame_bboxes"] 

        if not all_track_frame_bboxes: 
            continue
            
        if not raw_yolo_track["timecodes"]:
            continue
        track_original_start = raw_yolo_track["timecodes"][0]["start"]
        track_original_end = raw_yolo_track["timecodes"][0]["end"]

        if track_class != "person":
            new_id = f"{track_class}_refined_{next_refined_track_numeric_id}"
            next_refined_track_numeric_id += 1
            
            thumb_path = raw_yolo_track.get("representative_thumb_path") 
            if all_track_frame_bboxes:
                 new_thumb_path = generate_thumbnail_for_refined_track(
                    video_path, all_track_frame_bboxes, refined_thumbs_dir, new_id,
                    skip_thumbnail_saving=skip_thumbnail_saving
                 )
                 if new_thumb_path: thumb_path = new_thumb_path
            
            refined_tracks_output.append({
                "id": new_id,
                "class": track_class,
                "original_yolo_id": original_yolo_id,
                "thumb": thumb_path,
                "timecodes": [{"start": track_original_start, "end": track_original_end}],
                "bboxes": all_track_frame_bboxes,
                "source_scene_ids": _get_overlapping_scene_ids(track_original_start, track_original_end, manifest_segments)
            })
            continue

        overlapping_scenes = []
        for seg_idx, scene_segment in enumerate(manifest_segments):
            if max(track_original_start, scene_segment.start_time) < min(track_original_end, scene_segment.end_time):
                overlapping_scenes.append(scene_segment)
        
        overlapping_scenes.sort(key=lambda s: s.start_time)

        if len(overlapping_scenes) <= 1: 
            new_id = f"person_refined_{next_refined_track_numeric_id}"
            next_refined_track_numeric_id += 1
            
            # Use segment.id if available, otherwise segment_name, for source_scene_ids
            scene_ids_for_output = []
            if overlapping_scenes:
                scene_obj = overlapping_scenes[0]
                scene_id_val = getattr(scene_obj, 'id', None) or getattr(scene_obj, 'segment_name', f"scene_at_{scene_obj.start_time:.2f}")
                scene_ids_for_output.append(str(scene_id_val))
            
            refined_tracks_output.append({
                "id": new_id,
                "class": "person",
                "original_yolo_id": original_yolo_id,
                "thumb": generate_thumbnail_for_refined_track(video_path, all_track_frame_bboxes, refined_thumbs_dir, new_id, skip_thumbnail_saving=skip_thumbnail_saving),
                "timecodes": [{"start": track_original_start, "end": track_original_end}],
                "bboxes": all_track_frame_bboxes,
                "source_scene_ids": scene_ids_for_output
            })
            continue

        break_points = {track_original_start, track_original_end}

        for j in range(len(overlapping_scenes) - 1):
            current_scene_segment = overlapping_scenes[j]
            next_scene_segment = overlapping_scenes[j+1]

            cut_boundary_time_for_crop_before = current_scene_segment.end_time
            cut_boundary_time_for_crop_after = next_scene_segment.start_time
            split_decision_point = current_scene_segment.end_time

            crop_before_base64, _, _ = get_person_crop_at_boundary(
                video_path, all_track_frame_bboxes, cut_boundary_time_for_crop_before, "before"
            )
            crop_after_base64, _, _ = get_person_crop_at_boundary(
                video_path, all_track_frame_bboxes, cut_boundary_time_for_crop_after, "after"
            )

            p_tokens, c_tokens, img_tokens = 0,0,0 # Default if not called
            if not crop_before_base64 or not crop_after_base64:
                is_same_person = False 
            else:
                is_same_person, p_tokens, c_tokens, img_tokens = await _call_llm_for_reid(env, crop_before_base64, crop_after_base64, async_llm_client)
            
            total_reid_prompt_tokens += p_tokens
            total_reid_completion_tokens += c_tokens
            total_reid_image_tokens += img_tokens
            
            if not is_same_person:
                break_points.add(split_decision_point)
                if next_scene_segment.start_time > current_scene_segment.end_time + 0.1: 
                     break_points.add(next_scene_segment.start_time)

        sorted_break_points = sorted(list(break_points))
        
        for k in range(len(sorted_break_points) - 1):
            refined_segment_start_time = sorted_break_points[k]
            refined_segment_end_time = sorted_break_points[k+1]

            if refined_segment_end_time - refined_segment_start_time < 0.05: 
                continue

            segment_frame_bboxes = _filter_frame_bboxes(
                all_track_frame_bboxes, refined_segment_start_time, refined_segment_end_time
            )

            if not segment_frame_bboxes: 
                continue
            
            actual_start_time = segment_frame_bboxes[0]["timestamp"]
            actual_end_time = segment_frame_bboxes[-1]["timestamp"]
            
            if actual_end_time - actual_start_time < 0.05: 
                continue

            new_id = f"person_refined_{next_refined_track_numeric_id}"
            next_refined_track_numeric_id += 1
            
            refined_tracks_output.append({
                "id": new_id,
                "class": "person",
                "original_yolo_id": original_yolo_id,
                "thumb": generate_thumbnail_for_refined_track(video_path, segment_frame_bboxes, refined_thumbs_dir, new_id, skip_thumbnail_saving=skip_thumbnail_saving),
                "timecodes": [{"start": actual_start_time, "end": actual_end_time}],
                "bboxes": segment_frame_bboxes,
                "source_scene_ids": _get_overlapping_scene_ids(actual_start_time, actual_end_time, manifest_segments)
            })
    
    print(f"Track refinement finished. Produced {len(refined_tracks_output)} refined tracks.")
    print(f"Re-ID LLM Token Usage: Prompt={total_reid_prompt_tokens}, Completion={total_reid_completion_tokens}, ImageEst={total_reid_image_tokens}")
    return refined_tracks_output, total_reid_prompt_tokens, total_reid_completion_tokens, total_reid_image_tokens 