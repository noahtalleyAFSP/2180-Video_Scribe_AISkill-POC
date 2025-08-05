import json
import cv2
import os
import base64
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
# from face_recognition import face_encodings, load_image_file # type: ignore
# from scipy.spatial.distance import cosine # type: ignore
import logging # ADDED for logging
from openai import AsyncAzureOpenAI, AsyncOpenAI # ADDED for LLM client typing
import asyncio # ADDED for retry logic

from ..cobra_utils import estimate_image_tokens # MODIFIED: Import from cobra_utils

logger = logging.getLogger(__name__) # ADDED for logging

# --- User-defined Confidence Thresholds ---
CONF_FACE_EMBEDDING = 0.95  # Adjusted based on typical 1-cosine values (lower means more similar)
                            # User had 0.98 but for 1-cosine, higher is better.
                            # Let's re-evaluate: cosine distance: lower is better. 1-cosine: higher is better.
                            # User skeleton: cosine_distance(emb1, emb2) < 0.25. So 1 - cosine > 0.75
                            # Let's stick to user's provided logic: cosine_distance < THRESHOLD
FACE_EMBEDDING_SIMILARITY_THRESHOLD = 0.25 # Lower is more similar (max ~2.0 for completely different)
                                           # Original user example: < 0.25. So (1-cosine_distance) > 0.75 for "same"
CONF_VISION_LLM = 0.98      # GPT answer confidence for "yes"

# --- Helper: pad_bbox (from step0_yolo_track.py) ---
def pad_bbox(bbox: Tuple[int, int, int, int], shape: Tuple[int, int], pct: float = 0.10) -> Tuple[int, int, int, int]:
    """Pad bounding box by a percentage of its size.
    bbox: (x1, y1, x2, y2)
    shape: (height, width) of the image
    pct: percentage to pad (0.10 = 10%)
    """
    x1, y1, x2, y2 = bbox
    bh, bw = y2 - y1, x2 - x1 # Bbox height and width

    pad_w = int(bw * pct)
    pad_h = int(bh * pct)

    # Apply padding
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(shape[1] - 1, x2 + pad_w) # shape[1] is width
    y2 = min(shape[0] - 1, y2 + pad_h) # shape[0] is height
    return (x1, y1, x2, y2)

# --- Helper: Face Embedding ---
# def embed_face(img_bgr: Optional[cv2.Mat]) -> Optional[List[float]]:
#     """Generates face embedding for a BGR image. Returns None if no face or error."""
#     if img_bgr is None or img_bgr.size == 0:
#         logger.debug("Embed_face: Input image is None or empty.")
#         return None
#     try:
#         rgb_frame = img_bgr[:, :, ::-1]  # Convert BGR to RGB
#         # Ensure the image is writable as face_recognition might modify it internally
#         # or expect a contiguous array.
#         rgb_frame_writable = rgb_frame.copy()
#         encs = face_encodings(rgb_frame_writable, num_jitters=1, model='small')
#         if encs:
#             return encs[0].tolist() # Convert numpy array to list for JSON serialization
#         logger.debug("Embed_face: No face found in image.")
#         return None
#     except Exception as e:
#         logger.error(f"Embed_face: Error generating face embedding: {e}")
#         return None

# --- Helper: Extract Crops Around Cut ---
def get_crops_for_cut(
    track: Dict[str, Any], 
    cut_time: float,
    video_cap: cv2.VideoCapture, 
    pad_pct: float = 0.15 # User suggested 0.15s, but this is for bbox padding
) -> Tuple[Optional[cv2.Mat], Optional[cv2.Mat], Optional[Dict[str,Any]], Optional[Dict[str,Any]]]:
    """
    Finds frame records before and after a cut, and returns their cropped images.
    Also returns the frame_bbox dicts for potential thumbnail generation.
    """
    frame_bbox_before_cut: Optional[Dict[str, Any]] = None
    frame_bbox_after_cut: Optional[Dict[str, Any]] = None
    
    # Timestamps in frame_bboxes are floats
    frames_sorted = sorted(track.get("frame_bboxes", []), key=lambda x: x["timestamp"])

    # Find the last frame strictly BEFORE the cut
    for fb in reversed(frames_sorted):
        if fb["timestamp"] < cut_time:
            frame_bbox_before_cut = fb
            break
    
    # Find the first frame strictly AFTER the cut
    for fb in frames_sorted:
        if fb["timestamp"] > cut_time: # Strictly greater
            frame_bbox_after_cut = fb
            break

    if not frame_bbox_before_cut or not frame_bbox_after_cut:
        logger.debug(f"Track {track.get('id')}: Could not find frames immediately before and after cut {cut_time:.3f}s.")
        return None, None, None, None

    # Safety margin: ensure chosen frames are reasonably close to the cut
    # This might be too strict, depends on frame rate.
    # Max time diff from cut to frame (e.g. 1 second for typical FPS)
    MAX_TIME_DIFF_FROM_CUT = 1.0 
    if abs(frame_bbox_before_cut["timestamp"] - cut_time) > MAX_TIME_DIFF_FROM_CUT or \
       abs(frame_bbox_after_cut["timestamp"] - cut_time) > MAX_TIME_DIFF_FROM_CUT:
        logger.debug(f"Track {track.get('id')}: Frames around cut {cut_time:.3f}s are too far from the cut time. Before: {frame_bbox_before_cut['timestamp']:.3f}s, After: {frame_bbox_after_cut['timestamp']:.3f}s. Skipping comparison for this cut.")
        return None, None, None, None

    def _crop_frame(fb_data: Dict[str, Any]) -> Optional[cv2.Mat]:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, fb_data["frame_number"])
        ok, frame_img = video_cap.read()
        if not ok:
            logger.warning(f"Could not read frame number {fb_data['frame_number']} for track {track.get('id')}.")
            return None
        try:
            # Ensure bbox is a tuple of 4 ints
            bbox_tuple = tuple(map(int, fb_data["bbox"]))
            x1, y1, x2, y2 = pad_bbox(bbox_tuple, frame_img.shape[:2], pct=pad_pct) # Use the passed pad_pct for bbox
            return frame_img[y1:y2, x1:x2]
        except Exception as e:
            logger.error(f"Error padding/cropping frame {fb_data['frame_number']} for track {track.get('id')}: {e}. Bbox: {fb_data['bbox']}")
            return None

    crop_before = _crop_frame(frame_bbox_before_cut)
    crop_after = _crop_frame(frame_bbox_after_cut)
    
    if crop_before is None or crop_before.size == 0 or crop_after is None or crop_after.size == 0 :
        logger.warning(f"Track {track.get('id')}: Failed to generate valid crops around cut {cut_time:.3f}s.")
        return None, None, frame_bbox_before_cut, frame_bbox_after_cut # Return frame_bbox data even if crops failed

    return crop_before, crop_after, frame_bbox_before_cut, frame_bbox_after_cut


# --- Helper: Person Identity Comparison ---
async def is_same_person(
    crop_before: Optional[cv2.Mat], 
    crop_after: Optional[cv2.Mat],
    llm_client: Optional[AsyncAzureOpenAI | AsyncOpenAI],
    llm_deployment: Optional[str],
    prompt_log: Optional[list] = None
) -> Tuple[bool, float, int, int, int]:
    """
    Compares two image crops to determine if they are the same person using Vision LLM.
    If LLM is not available, defaults to False (different).
    Returns (is_same: bool, confidence: float, prompt_tokens: int, completion_tokens: int, image_tokens: int).
    Confidence for LLM: as reported by LLM.
    """
    if crop_before is None or crop_before.size == 0 or crop_after is None or crop_after.size == 0:
        logger.warning("is_same_person: One or both crops are invalid. Returning False.")
        return False, 0.0, 0, 0, 0 # MODIFIED: Return 0 for tokens

    # Prioritize Vision LLM for comparison
    if llm_client and llm_deployment:
        logger.debug("Using Vision LLM for person comparison.")
        try:
            def encode_crop_to_base64(crop_img: cv2.Mat) -> Optional[str]:
                if crop_img is None or crop_img.size == 0: return None
                # Attempt to encode, handle potential errors if crop_img is not a valid image format for imencode
                try:
                    is_success, buffer = cv2.imencode('.jpg', crop_img)
                    if not is_success:
                        logger.warning(f"cv2.imencode failed for a crop. Crop shape: {crop_img.shape if hasattr(crop_img, 'shape') else 'N/A'}")
                        return None
                    return base64.b64encode(buffer).decode('utf-8')
                except cv2.error as e:
                    logger.error(f"cv2.error during imencode: {e}. Crop shape: {crop_img.shape if hasattr(crop_img, 'shape') else 'N/A'}")
                    return None
                except Exception as e_gen:
                    logger.error(f"General exception during imencode: {e_gen}. Crop shape: {crop_img.shape if hasattr(crop_img, 'shape') else 'N/A'}")
                    return None

            img1_b64 = encode_crop_to_base64(crop_before)
            img2_b64 = encode_crop_to_base64(crop_after)

            if not img1_b64 or not img2_b64:
                logger.warning("Could not encode one or both crops for LLM. Returning False.")
                return False, 0.0, 0, 0, 0 # Default to different if encoding fails

            prompt_text = (
                "You are an expert in visual identity verification. Look at the two provided cropped images. "
                "Are these two images showing the exact same person? "
                "Consider facial features, clothing if visible and consistent, and any unique identifiers. "
                "Respond with only 'yes' or 'no', followed by a comma, then your confidence level as a decimal between 0.0 and 1.0 (e.g., 'yes, 0.98' or 'no, 0.95'). "
                "If you are completely unsure, respond 'no, 0.5'."
            )
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}", "detail": "high"}}
                ]}
            ]
            
            if prompt_log is not None:
                prompt_log.append({
                    "type": "reid_compare",
                    "context": {"crop_before": bool(crop_before is not None), "crop_after": bool(crop_after is not None)},
                    "messages": messages
                })
            
            # ADDED: Retry logic for LLM call
            max_retries = 3
            retry_delay_seconds = 5
            response_text = ""
            llm_response_obj = None # ADDED: To store the response object for token usage

            # --- ADDED: Token counting initialization ---
            prompt_tokens = 0
            completion_tokens = 0
            calculated_image_tokens = 0
            api_total_tokens = 0
            # --- END ADDED ---

            for attempt in range(max_retries):
                try:
                    # --- ADDED: Estimate image tokens before the call ---
                    calculated_image_tokens = estimate_image_tokens(messages) # type: ignore
                    # --- END ADDED ---

                    response = await llm_client.chat.completions.create(
                        model=llm_deployment,
                        messages=messages, # type: ignore
                        max_tokens=50,
                        temperature=0.1 
                    )
                    llm_response_obj = response # Store the response object
                    response_text = response.choices[0].message.content if response.choices and response.choices[0].message else ""
                    if response_text: # Successful call
                        break 
                except Exception as llm_call_e:
                    logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {llm_call_e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay_seconds) # Wait before retrying
                    else:
                        logger.error(f"LLM call failed after {max_retries} attempts for same_person check.")
                        raise # Re-raise the last exception if all retries fail
            
            # Ensure response_text is available before proceeding
            if not response_text or not llm_response_obj:
                logger.error("LLM did not return a response after retries. Defaulting to different.")
                return False, 0.0, 0, 0, 0 # MODIFIED: Return 0 for tokens

            logger.debug(f"LLM response for same_person: '{response_text}'") # Moved log line after retry logic

            # --- ADDED: Extract token usage from llm_response_obj --- 
            if hasattr(llm_response_obj, 'usage') and llm_response_obj.usage:
                prompt_tokens = getattr(llm_response_obj.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(llm_response_obj.usage, 'completion_tokens', 0)
                # Use API's image_tokens if available, else our estimate
                image_tokens_from_api = getattr(llm_response_obj.usage, 'image_tokens', None)
                if image_tokens_from_api is not None:
                    calculated_image_tokens = image_tokens_from_api
                api_total_tokens = getattr(llm_response_obj.usage, 'total_tokens', 0)
            # --- END ADDED ---

            parts = response_text.lower().strip().split(',')
            llm_says_same = parts[0].strip() == "yes"
            llm_confidence = 0.0
            if len(parts) > 1:
                try:
                    llm_confidence = float(parts[1].strip())
                except ValueError:
                    logger.warning(f"Could not parse confidence from LLM response: '{response_text}'")
            
            # Decision based on LLM confidence
            if llm_says_same and llm_confidence >= CONF_VISION_LLM:
                return True, llm_confidence, prompt_tokens, completion_tokens, calculated_image_tokens
            elif not llm_says_same and llm_confidence >= CONF_VISION_LLM:
                return False, llm_confidence, prompt_tokens, completion_tokens, calculated_image_tokens
            else: 
                logger.info(f"LLM was not confident enough (Same: {llm_says_same}, Conf: {llm_confidence:.2f}). Defaulting to 'different'.")
                return False, llm_confidence, prompt_tokens, completion_tokens, calculated_image_tokens # Default to different if LLM is unsure

        except Exception as e:
            logger.error(f"Error during LLM call for same_person: {e}", exc_info=True)
            return False, 0.0, 0, 0, 0 # MODIFIED: Default to different on LLM error, return 0 for tokens
    else:
        logger.warning("LLM client not available for person Re-ID. Defaulting to 'different'.")
        return False, 0.0, 0, 0, 0 # MODIFIED: Default to different if no LLM client, return 0 for tokens


# --- Helper: Generate Thumbnail for a Track Segment ---
def _generate_thumbnail_for_split_track(
    track_segment_frame_bboxes: List[Dict[str, Any]],
    video_cap: cv2.VideoCapture,
    thumb_dir: Path,
    new_track_id: str,
    pad_pct: float = 0.10,
    skip_thumbnail_saving: bool = False
) -> Optional[str]:
    """Generates a thumbnail for a representative frame of a track segment."""
    if not track_segment_frame_bboxes: # No frames in this segment
        logger.warning(f"Cannot generate thumbnail for {new_track_id}: no frame_bboxes provided.")
        return None
    
    if skip_thumbnail_saving: # ADDED check
        logger.debug(f"Skipping thumbnail generation for {new_track_id} due to skip_thumbnail_saving=True.")
        return None 

    # Ensure thumbs_dir exists (it should be created by the caller if saving is enabled)
    # However, if only this function is called directly and saving is intended, it needs to exist.
    # For safety, we can check and create it here if it's truly needed and skip_thumbnail_saving is False.
    # But generally, the caller (refine_tracks_over_cuts) should manage the directory creation based on the flag.
    if not thumb_dir.exists():
        logger.info(f"Thumbnail directory {thumb_dir} does not exist. Creating it for {new_track_id}.")
        try:
            thumb_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            logger.error(f"Failed to create thumbnail directory {thumb_dir}: {e_mkdir}. Cannot save thumbnail for {new_track_id}.")
            return None

    # Select a representative frame (e.g., middle frame of the segment)
    if len(track_segment_frame_bboxes) > 0:
        middle_index = len(track_segment_frame_bboxes) // 2
        middle_frame = track_segment_frame_bboxes[middle_index]
    else:
        logger.warning(f"Cannot generate thumbnail for {new_track_id}: no frame_bboxes provided.")
        return None

    video_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame["frame_number"])
    ret_thumb, thumb_frame_img = video_cap.read()
    if not ret_thumb:
        logger.warning(f"Failed to read frame {middle_frame['frame_number']} for new track {new_track_id} thumbnail.")
        return None

    try:
        bbox_tuple = tuple(map(int, middle_frame["bbox"]))
        x1_p, y1_p, x2_p, y2_p = pad_bbox(bbox_tuple, thumb_frame_img.shape[:2], pct=pad_pct)
        cropped_thumb = thumb_frame_img[y1_p:y2_p, x1_p:x2_p]

        if cropped_thumb.size > 0:
            thumb_filename = f"split_track_{new_track_id}.jpg"
            thumb_full_path = thumb_dir / thumb_filename
            cv2.imwrite(str(thumb_full_path), cropped_thumb)
            return str(thumb_full_path.resolve())
        else:
            logger.warning(f"Empty crop for thumbnail of new track {new_track_id} at frame {middle_frame['frame_number']}")
            return None
    except Exception as e:
        logger.error(f"Failed to generate thumbnail for new track {new_track_id}: {e}")
        return None

# --- Main Refinement Function ---
async def refine_tracks_over_cuts(
    raw_yolo_tracks_path: str,
    scene_cuts_path: str, # Path to JSON file with list of cut timestamps [float, ...]
    video_path_str: str,
    output_dir_str: str,
    llm_client: Optional[AsyncAzureOpenAI | AsyncOpenAI], # For LLM fallback
    llm_deployment: Optional[str],    # For LLM fallback
    skip_thumbnail_saving: bool = False # ADDED
) -> Tuple[List[Dict[str, Any]], int, int, int]: # MODIFIED: Added token counts to return type
    """
    Refines YOLO tracks by attempting to link track segments across scene cuts.
    Returns the refined list of tracks and total prompt, completion, and image tokens used.
    """
    if not llm_client or not llm_deployment:
        logger.warning("LLM client or deployment not provided to refine_tracks_over_cuts. Skipping Re-ID over cuts.")
        with open(raw_yolo_tracks_path, 'r', encoding='utf-8') as f_raw: # Ensure utf-8 encoding
            raw_tracks = json.load(f_raw)
        return raw_tracks, 0, 0, 0 # Return raw tracks and 0 tokens if no LLM

    # --- ADDED: Initialize token counters ---
    total_prompt_tokens_step = 0
    total_completion_tokens_step = 0
    total_image_tokens_step = 0
    # --- END ADDED ---

    output_dir = Path(output_dir_str)
    split_thumbs_dir = Path(output_dir_str) / "yolo_thumbs_reid_split"
    if not skip_thumbnail_saving: # ADDED: conditional creation
        logger.info(f"Thumbnails for split tracks will be saved to: {split_thumbs_dir}")
        split_thumbs_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Thumbnail saving for split tracks is SKIPPED.")

    # --- ADDED: Load scene_cuts from scene_cuts_path ---
    try:
        with open(scene_cuts_path, 'r', encoding='utf-8') as f_cuts:
            scene_cuts = json.load(f_cuts) # This will be the list of cut timestamps
        if not isinstance(scene_cuts, list):
            logger.error(f"Content of scene_cuts_path ({scene_cuts_path}) is not a list as expected. Got: {type(scene_cuts)}")
            # Decide on fallback: return raw_tracks or raise error
            return raw_tracks, 0, 0, 0 # Fallback: return raw tracks if cuts format is wrong
    except Exception as e_load_cuts:
        logger.error(f"Failed to load or parse scene_cuts_path {scene_cuts_path}: {e_load_cuts}")
        return raw_tracks, 0, 0, 0 # Fallback: return raw tracks if cuts file fails to load
    # --- END ADDED ---

    video_cap = cv2.VideoCapture(str(video_path_str))
    if not video_cap.isOpened():
        logger.error(f"Cannot open video: {video_path_str}")
        # Attempt to load raw_tracks here as well, so it's defined for the return
        try:
            with open(raw_yolo_tracks_path, 'r', encoding='utf-8') as f_raw_fallback: # Ensure utf-8 encoding
                raw_tracks_fallback = json.load(f_raw_fallback)
            return raw_tracks_fallback, 0, 0, 0
        except Exception as e_load_fallback:
            logger.error(f"Failed to load raw_tracks_path {raw_yolo_tracks_path} even for fallback return: {e_load_fallback}")
            return [], 0, 0, 0 # Return empty list and 0 tokens if video and raw_tracks file fail

    # --- ADDED: Load raw_tracks from raw_yolo_tracks_path ---
    try:
        with open(raw_yolo_tracks_path, 'r', encoding='utf-8') as f_raw_main: # Ensure utf-8 encoding
            raw_tracks = json.load(f_raw_main)
    except Exception as e_load_main:
        logger.error(f"Failed to load raw_yolo_tracks_path {raw_yolo_tracks_path}: {e_load_main}")
        video_cap.release() # Release video capture if open
        return [], 0, 0, 0 # Return empty list and 0 tokens if loading fails
    # --- END ADDED ---

    final_tracks_list: List[Dict[str, Any]] = []
    
    # Determine the starting point for new IDs
    max_existing_id = 0
    for t in raw_tracks:
        try:
            if t.get("id") is not None:
                 max_existing_id = max(max_existing_id, int(t["id"]))
        except ValueError: # Handle cases where ID might not be a simple integer string
            logger.warning(f"Track ID '{t.get('id')}' is not a simple integer. Will use a UUID-based scheme for new IDs if needed, or rely on a large offset.")
            # Fallback to a very large number or UUID if parsing all IDs as int is problematic
            # For now, we'll assume numeric IDs or add a large number to avoid collision.
    next_new_id_counter = max_existing_id + 1

    progress_total = len(raw_tracks)  # crude estimate: each track will be checked at most once per cut
    progress_done = 0

    for original_track in raw_tracks:
        try: # ADDED: Outer try-except for each original_track
            if original_track.get("class") != "person":
                final_tracks_list.append(original_track)
                continue

            current_track_segments = [original_track] # Start with the original track as the first segment

            # Iterate over cuts to potentially split the track multiple times
            # A track might be split, and then the latter part of that split might be split again by a subsequent cut.
            
            # This loop processes one original track and produces one or more track segments from it.
            # All these produced segments are added to final_tracks_list.
            
            processed_track_parts_for_this_original_track = []
            
            # 'track_to_process' holds the current portion of the original track being examined for splits.
            # Initially, it's the full original_track. If a split occurs, track_to_process becomes the
            # 'left' part, and the 'right' part is queued for further checks against subsequent cuts.
            
            queue_of_track_portions_to_check = [{
                "id": original_track["id"], # Original ID
                "class": original_track["class"],
                "timecodes": original_track.get("timecodes", []), # Operate on a copy
                "frame_bboxes": original_track.get("frame_bboxes", []),
                "thumb": original_track.get("thumb"),
                "original_yolo_ids": [original_track["id"]] # Track its origin
            }]
            
            temp_id_assignment_map = {} # For new tracks, to ensure consistent thumb name if new ID is temp

            while queue_of_track_portions_to_check:
                current_portion = queue_of_track_portions_to_check.pop(0)
                current_timecodes = current_portion["timecodes"][:] # Work with a copy
                current_frame_bboxes = current_portion["frame_bboxes"]
                
                track_was_split_by_a_cut = False

                # Check this portion against all relevant cuts
                for tc_idx, timecode_interval in enumerate(current_timecodes):
                    s_time = timecode_interval["start"]
                    e_time = timecode_interval["end"]
                    
                    # Find cuts that fall strictly within this timecode_interval
                    relevant_cuts_for_interval = sorted([c for c in scene_cuts if s_time < c < e_time])

                    if not relevant_cuts_for_interval:
                        continue # No cuts within this specific timecode of the current portion

                    # If we are here, timecode_interval spans at least one cut.
                    # We process cuts left-to-right for this interval.
                    # Once a split occurs, the 'right' part is added to the main queue,
                    # and the 'left' part (which is 'current_portion' but with modified timecode)
                    # continues to be checked against further cuts *within its new, shorter duration*.
                    
                    # Effective start and end for the current segment being processed before a potential split
                    # This allows iterative splitting of a single timecode_interval
                    current_segment_start = s_time 
                    
                    for cut_event_time in relevant_cuts_for_interval:
                        if not (current_segment_start < cut_event_time < e_time): # Ensure cut is still relevant for the (potentially now shorter) segment
                            continue

                        logger.debug(f"Track {current_portion['id']} (orig ID: {original_track['id']}): Checking cut at {cut_event_time:.3f}s within interval {current_segment_start:.3f}s - {e_time:.3f}s.")
                        
                        crop1, crop2, fb_before, fb_after = get_crops_for_cut(current_portion, cut_event_time, video_cap)

                        if crop1 is None or crop2 is None:
                            logger.debug(f"Could not get valid crops for track {current_portion['id']} at cut {cut_event_time:.3f}s. Assuming same person to avoid unnecessary split.")
                            # To be safe, if crops fail, assume it's the same person to avoid an incorrect split.
                            # Or, could default to different if strictness is preferred. For now, assume same.
                            current_segment_start = cut_event_time # Move past this cut for this interval processing
                            continue 

                        # --- Perform Re-ID ---
                        # Only do Re-ID for persons for now
                        is_same_entity = True # Default for non-persons or if re-id fails
                        similarity_confidence = 0.0

                        if current_portion["class"] == "person":
                            progress_done += 1
                            print(f"[ReID {progress_done}] Track {current_portion['id']} vs cut {cut_event_time:.2f}s … calling Vision LLM …")
                            # ADD LOGGING HERE
                            logger.info(f"Verifying Person ID '{current_portion['id']}' (original raw ID: {current_portion.get('original_yolo_ids', ['N/A'])[0]}) across cut at {cut_event_time:.3f}s.")
                            is_same_entity, similarity_confidence, p_tokens, c_tokens, i_tokens = await is_same_person(
                                crop1, crop2, 
                                llm_client=llm_client, llm_deployment=llm_deployment
                            )
                            # --- ADDED: Accumulate tokens ---
                            total_prompt_tokens_step += p_tokens
                            total_completion_tokens_step += c_tokens
                            total_image_tokens_step += i_tokens
                            # --- END ADDED ---
                            logger.info(f"Re-ID for Person ID '{current_portion['id']}' at cut {cut_event_time:.3f}s: Same Person = {is_same_entity}, Confidence = {similarity_confidence:.2f}")
                        elif crop1 is not None and crop2 is not None:
                            # Basic check for non-persons (e.g. if they are visually identical enough, like a static object)
                            # This is a placeholder for more sophisticated non-person Re-ID if needed.
                            # For now, assume same person
                            is_same_entity = True
                            similarity_confidence = 1.0

                        if not is_same_entity:
                            track_was_split_by_a_cut = True
                            
                            # --- Split the track ---
                            new_split_track_id_str = str(next_new_id_counter)
                            next_new_id_counter += 1
                            temp_id_assignment_map[id(current_portion)] = new_split_track_id_str


                            # Modify the current_portion (left part of the split)
                            # Its current timecode_interval is shortened
                            current_timecodes[tc_idx] = {"start": current_segment_start, "end": cut_event_time}
                            
                            # Create the new track (right part of the split)
                            new_track_timecodes = [{"start": cut_event_time, "end": e_time}]
                            
                            # If the original timecode_interval spanned MORE cuts after this one,
                            # those remaining parts of the interval need to be preserved for the new_track.
                            # Example: original interval [0-10], cuts at 3, 7.
                            # 1. Split at 3: left=[0-3], right_candidate=[3-10]
                            #    'current_portion' becomes the [0-3] part.
                            #    'new_track_after_split' gets [3-10]. This [3-10] will be added to queue.
                            
                            new_track_frame_bboxes = [fb for fb in current_frame_bboxes if fb["timestamp"] >= cut_event_time]
                            
                            new_track_thumb_path = _generate_thumbnail_for_split_track(
                                new_track_frame_bboxes, video_cap, split_thumbs_dir, new_split_track_id_str, skip_thumbnail_saving=skip_thumbnail_saving
                            )

                            new_track_after_split = {
                                "id": new_split_track_id_str,
                                "class": current_portion["class"],
                                "timecodes": new_track_timecodes, # Only the segment from cut to original e_time
                                "frame_bboxes": new_track_frame_bboxes,
                                "thumb": new_track_thumb_path,
                                "original_yolo_ids": current_portion.get("original_yolo_ids", [current_portion["id"]]), # Inherit lineage
                                "split_from_cut": cut_event_time
                            }
                            queue_of_track_portions_to_check.append(new_track_after_split) # Add new part to queue for further checks
                            
                            # The current_portion's active timecode_interval has been shortened.
                            # Its e_time for further processing of *this current_portion* is now cut_event_time.
                            # We break from iterating `relevant_cuts_for_interval` for *this* specific timecode_interval
                            # because it has been fundamentally altered. The new portion created from the split
                            # will handle cuts that were to its right.
                            e_time = cut_event_time # Update effective end for current_portion's timecode_interval
                            break # Break from relevant_cuts_for_interval loop for this tc_idx
                        else:
                            # Same person, continue checking this interval against the next cut
                            current_segment_start = cut_event_time
                    
                    # If this timecode_interval was split, current_timecodes[tc_idx] is already updated.
                    # If not split by any of its internal cuts, it remains as it was from the input current_portion.
                    # No special action needed here for that case.

                # After checking all timecode_intervals in current_portion against relevant cuts:
                # The current_portion might have had some of its timecode_intervals shortened.
                # It should be added to the final results.
                
                # Clean up: Remove timecodes that might have become empty or invalid due to splits
                valid_timecodes_for_current_portion = [
                    tc for tc in current_timecodes if tc["start"] < tc["end"]
                ]
                if not valid_timecodes_for_current_portion:
                     logger.debug(f"Portion of track {current_portion['id']} has no valid timecodes left after splits. Discarding.")
                     continue # Don't add this part if it has no duration

                current_portion["timecodes"] = valid_timecodes_for_current_portion
                
                # Update frame_bboxes for the current_portion to only include those within its final timecodes
                # This is important if a later part of its original bbox list was split off.
                final_bboxes_for_current_portion = []
                min_start_overall = min(tc["start"] for tc in current_portion["timecodes"])
                max_end_overall = max(tc["end"] for tc in current_portion["timecodes"])
                for fb in current_portion["frame_bboxes"]:
                    if min_start_overall <= fb["timestamp"] <= max_end_overall:
                        final_bboxes_for_current_portion.append(fb)
                current_portion["frame_bboxes"] = final_bboxes_for_current_portion

                # Update thumb for current_portion if it was the one split and its ID changes implicitly
                # or ensure its thumb is still relevant if it's the original_track object being modified.
                # If current_portion is the original track object and it was split, its thumb might need re-eval if first part is very short.
                # For simplicity, original track keeps its original thumb. Split-off parts get new thumbs.
                if current_portion["id"] == original_track["id"] and track_was_split_by_a_cut:
                     # The original track object (now representing the first part) might need its 'thumb'
                     # re-evaluated if its content significantly changed.
                     # For now, it keeps its original thumb.
                     pass


                processed_track_parts_for_this_original_track.append(current_portion)

            final_tracks_list.extend(processed_track_parts_for_this_original_track)
        
        except Exception as e_track_processing: # ADDED: Catch exceptions during processing of a single original_track
            track_id_for_log = original_track.get('id', 'UNKNOWN_ID')
            logger.error(f"Error processing original track ID {track_id_for_log} in refine_tracks_over_cuts: {e_track_processing}", exc_info=True)
            # Optionally, add the original_track to final_tracks_list if you want to keep it unprocessed on error
            # final_tracks_list.append(original_track) 
            logger.info(f"Skipping track ID {track_id_for_log} due to error and continuing with next tracks.")

    video_cap.release()

    # Save the refined tracks list
    final_refined_tracks_file = output_dir / f"{Path(video_path_str).stem}_reid_split_refined_yolo_tags.json"
    with open(final_refined_tracks_file, "w") as f:
        json.dump(final_tracks_list, f, indent=2)
    logger.info(f"Saved refined tracks (after cut splitting) to {final_refined_tracks_file}")

    return final_tracks_list, total_prompt_tokens_step, total_completion_tokens_step, total_image_tokens_step # MODIFIED: Return tokens

if __name__ == '__main__':
    # Example usage (requires dummy files and a test video)
    print("Testing step0b_reid_split_tracks.py")
    # This would require setting up a test video, running step0_yolo_track,
    # creating a dummy _cuts.json, and then calling refine_tracks_over_cuts.
    # Due to complexity, direct __main__ test is omitted here but recommended for development.
    pass 