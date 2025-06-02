from pathlib import Path
import json
import time
import cv2
import torch
import os
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

# Helper function to ensure output directories exist
def _ensure_dir(directory_path: Path):
    """Creates a directory if it doesn't exist."""
    directory_path.mkdir(parents=True, exist_ok=True)

def pad_bbox(bbox, shape, pct=0.10):
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

def run_yolo(video_path: Path, out_dir: Path,
             model_path: str = "yolo11x.pt",
             tracker_yaml: str = "bytetrack.yaml",
             conf: float = 0.20, iou: float = 0.80) -> List[Dict[str, Any]]:
    """
    Processes a video file, performs object tracking, and returns detailed track information.

    Output structure for each item in the list (manifest.raw_yolo_tags):
    {
      "id": "yolo_tracker_internal_id_XYZ", // Unique ID from the YOLO tracker (integer as string)
      "class": "person",
      "timecodes": [ // One continuous block per id
        { "start": 10.500, "end": 25.300 }
      ],
      "frame_bboxes": [
        { "frame_number": 315, "timestamp": 10.500, "bbox": [100, 150, 180, 400] }, // x1, y1, x2, y2
        // ... more frames ...
      ],
      "representative_thumb_path": "path/to/out_dir/thumbs/raw_track_XYZ.jpg" // Path to a single representative thumbnail
    }
    """
    print(f"Starting YOLO tracking for {video_path.name}...")
    t_start_yolo_processing = time.time()

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[WARNING] Video FPS is 0, defaulting to 30 FPS for timestamp calculations.")
        fps = 30.0 # Default FPS if not available or zero

    # --- Load YOLO Model ---
    model_file = Path(model_path)
    if not model_file.exists():
        # Try to resolve relative to script if not absolute
        model_file = Path(__file__).resolve().parent / model_path
        if not model_file.exists():
            print(f"[ERROR] YOLO model weights not found at {model_path} or {model_file}")
            cap.release()
            return []
    
    tracker_config_file = Path(__file__).resolve().parent / tracker_yaml # Assume tracker_yaml is in the same dir
    if not tracker_config_file.exists():
        print(f"[ERROR] Tracker config {tracker_yaml} not found at {tracker_config_file}")
        cap.release()
        return []

    yolo_model = YOLO(str(model_file))
    
    # --- Prepare output directories ---
    thumbs_dir = out_dir / "yolo_thumbs_raw" # Directory for raw track thumbnails
    _ensure_dir(thumbs_dir)

    # Intermediate structure to store track data:
    # { track_id: {"class": str, "frames": [{"frame_number": int, "timestamp": float, "bbox": list}]} }
    active_tracks_data: Dict[int, Dict[str, Any]] = {}
    
    frame_idx = -1 # Start at -1, so first frame is 0
    
    raw_yolo_output_list: List[Dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        current_timestamp = frame_idx / fps

        try:
            results = yolo_model.track(
                frame,
                imgsz=1280, # Consider making this configurable
                conf=conf,
                iou=iou,
                half=torch.cuda.is_available(),
                persist=True,
                tracker=str(tracker_config_file),
                verbose=False, # Reduce console spam
            )
        except Exception as e:
            print(f"[ERROR] YOLO model.track failed on frame {frame_idx}: {e}")
            continue # Skip frame on error

        res = results[0] # First result object

        if res.boxes.id is not None:
            for box_coords, track_id_tensor, cls_tensor in zip(
                res.boxes.xyxy.cpu().numpy(),      # Bounding boxes (x1, y1, x2, y2)
                res.boxes.id.int().cpu().numpy(),  # Track IDs
                res.boxes.cls.int().cpu().numpy(), # Class IDs
            ):
                track_id = int(track_id_tensor)
                class_name = yolo_model.names.get(int(cls_tensor), f"class_{cls_tensor}")
                bbox_int = [int(c) for c in box_coords] # Convert to int list [x1, y1, x2, y2]

                frame_data = {
                    "frame_number": frame_idx,
                    "timestamp": round(current_timestamp, 3),
                    "bbox": bbox_int
                }

                if track_id not in active_tracks_data:
                    active_tracks_data[track_id] = {
                        "class": class_name,
                        "frames_data": [frame_data],
                        "start_frame": frame_idx, # Keep track of first frame for thumbnail
                        "start_timestamp": round(current_timestamp, 3)
                    }
                else:
                    # Ensure class consistency for a track ID (YOLO might flicker)
                    # If class changes, could treat as new track or use first detected class
                    # For now, assume class is consistent or use first seen.
                    active_tracks_data[track_id]["frames_data"].append(frame_data)
        # else:
            # No tracks detected in this frame by res.boxes.id

    cap.release()

    # --- Consolidate active_tracks_data into raw_yolo_output_list ---
    for track_id, data in active_tracks_data.items():
        if not data["frames_data"]:
            continue # Should not happen if data was added

        first_frame_data = data["frames_data"][0]
        last_frame_data = data["frames_data"][-1]
        
        start_time = first_frame_data["timestamp"]
        end_time = last_frame_data["timestamp"]

        # Generate a representative thumbnail for the raw track (e.g., from its first frame)
        thumb_path_str = None
        try:
            # Re-open video to extract the specific frame for thumbnail
            thumb_cap = cv2.VideoCapture(str(video_path))
            if thumb_cap.isOpened():
                thumb_cap.set(cv2.CAP_PROP_POS_FRAMES, data["start_frame"])
                ret_thumb, thumb_frame_img = thumb_cap.read()
                if ret_thumb:
                    thumb_bbox = first_frame_data["bbox"] # Use bbox from its first appearance
                    # Pad the bounding box before cropping
                    padded_bbox = pad_bbox(thumb_bbox, thumb_frame_img.shape[:2])
                    x1, y1, x2, y2 = padded_bbox
                    cropped_thumb = thumb_frame_img[y1:y2, x1:x2]
                    
                    if cropped_thumb.size > 0: # Ensure crop is not empty
                        thumb_filename = f"raw_track_{track_id}.jpg"
                        thumb_full_path = thumbs_dir / thumb_filename
                        cv2.imwrite(str(thumb_full_path), cropped_thumb)
                        thumb_path_str = str(thumb_full_path.resolve())
                    else:
                        print(f"[WARNING] Empty crop for thumbnail of track {track_id} at frame {data['start_frame']}")
                thumb_cap.release()
        except Exception as e:
            print(f"[ERROR] Failed to generate thumbnail for raw track {track_id}: {e}")


        raw_yolo_output_list.append({
            "id": str(track_id),
            "class": data["class"],
            "timecodes": [{"start": start_time, "end": end_time}],
            "frame_bboxes": data["frames_data"],
            "thumb": thumb_path_str
        })

    processing_time = time.time() - t_start_yolo_processing
    print(f"YOLO tracking for {video_path.name} completed in {processing_time:.2f}s. Found {len(raw_yolo_output_list)} raw tracks.")
    
    # Optional: Save this detailed list as a JSON file for debugging
    # debug_json_path = out_dir / f"{video_path.stem}_yolo_raw_detailed.json"
    # with open(debug_json_path, "w") as f:
    #     json.dump(raw_yolo_output_list, f, indent=2)
    # print(f"Saved detailed raw YOLO tracking to {debug_json_path}")

    return raw_yolo_output_list

# --- Main execution block (if script is run directly) ---
# This part is usually for testing the script standalone.
# It can be kept or removed based on whether this script is only a module or also executable.
if __name__ == '__main__':
    print("Running step0_yolo_track.py directly for testing...")
    # Create dummy/test inputs
    test_video_dir = Path("test_videos")
    test_output_dir = Path("test_yolo_output")
    _ensure_dir(test_video_dir)
    _ensure_dir(test_output_dir)

    # Create a short dummy video for testing if none exists
    dummy_video_path = test_video_dir / "dummy_input.mp4"
    if not dummy_video_path.exists():
        print(f"Creating dummy video at {dummy_video_path} for testing...")
        w, h, fps_dummy, dur = 320, 240, 10, 3 # Short and small
        dummy_out_vid = cv2.VideoWriter(str(dummy_video_path),
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_dummy, (w, h))
        for i in range(int(fps_dummy * dur)):
            frm = cv2.UMat(h, w, cv2.CV_8UC3).get() # Initialize frame
            frm[:] = (i * 10 % 255, i * 5 % 255, i * 2 % 255) # Changing background
            # Add a moving circle to simulate an object
            center_x = w // 2 + int(30 * (i / (fps_dummy * dur) * 2 -1)) # Moves across
            center_y = h // 2
            cv2.circle(frm, (center_x, center_y), 15, (50, 150, 255), -1)
            cv2.putText(frm, f"F{i+1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            dummy_out_vid.write(frm)
        dummy_out_vid.release()
        print(f"Dummy video created. Please ensure you have 'yolo11x.pt' and 'bytetrack.yaml' accessible.")
    
    if dummy_video_path.exists():
        # Assuming yolo11x.pt and bytetrack.yaml are in the same directory as this script
        # or provide full paths.
        # For testing, you might need to download a small YOLO model if yolo11x.pt is large/proprietary
        # e.g., model_path="yolov8n.pt" (if 'yolov8n.pt' is available)
        
        # Ensure model and tracker files are present for the test
        # You might need to manually place them or adjust paths for __main__
        script_dir = Path(__file__).resolve().parent
        test_model_path = script_dir / "yolo11x.pt" # Adjust if your test model is different/elsewhere
        test_tracker_yaml = script_dir / "bytetrack.yaml"

        if not test_model_path.exists():
            print(f"TESTING SKIPPED: Model for testing not found at {test_model_path}")
            print("Please place yolo11x.pt (or a test model) and bytetrack.yaml in the same directory as this script, or modify paths.")
        else:
            print(f"Using model: {test_model_path}")
            print(f"Using tracker: {test_tracker_yaml}")
            detailed_tracks = run_yolo(
                video_path=dummy_video_path,
                out_dir=test_output_dir,
                model_path=str(test_model_path),
                tracker_yaml=str(test_tracker_yaml)
            )
            if detailed_tracks:
                print(f"\nSuccessfully processed dummy video. Got {len(detailed_tracks)} detailed tracks.")
                # Save the output for inspection
                output_json_path = test_output_dir / f"{dummy_video_path.stem}_yolo_raw_detailed_TEST.json"
                with open(output_json_path, "w") as f:
                    json.dump(detailed_tracks, f, indent=2)
                print(f"Test output saved to: {output_json_path}")
                print("Sample of first track:", json.dumps(detailed_tracks[0] if detailed_tracks else {}, indent=2))
            else:
                print("\nProcessing dummy video resulted in no detailed tracks.")
    else:
        print("Dummy video for testing does not exist. Skipping direct run test.") 