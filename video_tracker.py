"""

To hide the GUI that live previews it comment out the following lines:
# cv2.imshow("YOLOv8 Tracking", frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

How to adjust it
Disable the pop-up – leave SHOW_WINDOW = False.

Skip saving a video – set SAVE_VIDEO = False.

Change confidence threshold – tweak conf=0.25.

Just detect (no IDs) – replace model.track(...) with model.predict(...) and drop the tracking logic.

"""
# --- video_tracker.py -------------------------------------------------------
from pathlib import Path
import json, time, cv2, numpy as np, torch, os
from ultralytics import YOLO

# --- NEW: Select device globally ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] YOLO will run on device: {DEVICE}")

# ──────────────────────────── USER SETTINGS ────────────────────────────────
MODEL_PATH   = "yolo11x.pt"   # your detector weights
TRACKER_YAML = "bytetrack.yaml"    # or "deepsort.yaml" (must be in same folder)
VIDEO_SOURCE_DIR = Path("tracking_tests/") # directory containing source video files
OUTPUT_DIR       = Path("tracking_tests_results/")   # directory to save JSON and annotated videos

SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'] # Add/remove as needed

SAVE_VIDEO   = True    # save an annotated MP4
SHOW_WINDOW  = False   # set True to watch a live pop-up (cv2.imshow)
# ────────────────────────────────────────────────────────────────────────────

def ensure_dir(directory_path: Path):
    """Creates a directory if it doesn't exist."""
    directory_path.mkdir(parents=True, exist_ok=True)

def process_video(video_path: Path, model: YOLO, tracker_config: str, output_dir: Path):
    """
    Processes a single video file, performs object tracking, and saves results.
    """
    print(f"Processing {video_path.name}...")
    t_start_video_processing = time.time()
    use_half = DEVICE == "cuda"  # determine precision once per video

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] Cannot open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = None
    if SAVE_VIDEO:
        ensure_dir(output_dir)
        out_video_name = f"{video_path.stem}_tracked.mp4"
        out_video_path = output_dir / out_video_name
        video_writer = cv2.VideoWriter(str(out_video_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
        print(f"[info] Saving annotated video to {out_video_path}")

    # Book-keeping structure for current video
    # {tid: {"class": str, "frames": [(start_frame, end_frame), ...]}}
    # This intermediate structure helps in merging fragmented tracks of the same object later.
    # We will convert frames to timecodes at the end.
    raw_track_log: dict[int, dict[str, any]] = {} 
    
    frame_idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # model.to(DEVICE)  # removed redundant move each frame
        # --- ensure half precision when on CUDA ---
        # use_half is defined once before the loop
        results = model.track(
            frame,
            imgsz=1280,
            conf=0.20,
            iou=0.80,
            half=use_half,
            persist=True,
            tracker=tracker_config,
            verbose=False,
            device=DEVICE,
        )
        res = results[0]

        if res.boxes.id is not None:
            for box, tid, cls in zip(
                res.boxes.xyxy.int().cpu().numpy(),
                res.boxes.id.int().cpu().numpy(),
                res.boxes.cls.int().cpu().numpy(),
            ):
                tid_py = int(tid)
                label  = model.names.get(int(cls), f"class_{cls}")
                
                if tid_py not in raw_track_log:
                    raw_track_log[tid_py] = {"class": label, "last_seen_frame": frame_idx, "segments": [(frame_idx, frame_idx)]}
                else:
                    # If it's the same object and it's a continuation of the last segment
                    if raw_track_log[tid_py]["last_seen_frame"] == frame_idx - 1:
                        last_segment_start, _ = raw_track_log[tid_py]["segments"][-1]
                        raw_track_log[tid_py]["segments"][-1] = (last_segment_start, frame_idx)
                    else: # New segment for an existing track ID (re-appeared)
                        raw_track_log[tid_py]["segments"].append((frame_idx, frame_idx))
                    raw_track_log[tid_py]["last_seen_frame"] = frame_idx


                if SAVE_VIDEO or SHOW_WINDOW:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{tid_py}:{label}", (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if SAVE_VIDEO and video_writer:
            video_writer.write(frame)
        if SHOW_WINDOW:
            cv2.imshow(f"YOLOv8 Tracking - {video_path.name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[info] User pressed 'q', stopping video processing.")
                break
    
    cap.release()
    if SAVE_VIDEO and video_writer:
        video_writer.release()
    if SHOW_WINDOW:
        cv2.destroyWindow(f"YOLOv8 Tracking - {video_path.name}") # Destroy specific window

    processing_time = time.time() - t_start_video_processing
    print(f"Processed {frame_idx} frames from {video_path.name} in {processing_time:.1f}s.")

    # --- Convert raw_track_log to the desired JSON structure ---
    # Desired: {"actionSummary": {"object": [{"name": "obj_name", "timecodes": [{"start": s, "end": e}]}]}}
    
    # First, group by class name, collecting all segments
    # {"class_name": [{"start_frame": X, "end_frame": Y}, ...]}
    class_based_segments: dict[str, list[dict[str, int]]] = {}
    for tid_data in raw_track_log.values():
        class_name = tid_data["class"]
        if class_name not in class_based_segments:
            class_based_segments[class_name] = []
        for start_frame, end_frame in tid_data["segments"]:
            class_based_segments[class_name].append({"start_frame": start_frame, "end_frame": end_frame})

    # Now, convert to the final JSON structure with timecodes
    output_objects = []
    for class_name, segments in class_based_segments.items():
        timecodes = []
        for seg in segments:
            timecodes.append({
                "start": round(seg["start_frame"] / fps, 3),
                "end": round(seg["end_frame"] / fps, 3)
            })
        # Optional: Sort timecodes by start time if needed, though segments should already be somewhat ordered
        # timecodes.sort(key=lambda x: x["start"]) 
        output_objects.append({"name": class_name, "timecodes": timecodes})
        
    final_json_output = {"actionSummary": {"object": output_objects}}
    
    ensure_dir(output_dir)
    out_json_name = f"{video_path.stem}_yolo.json"
    out_json_path = output_dir / out_json_name
    with open(out_json_path, "w") as f:
        json.dump(final_json_output, f, indent=4) # Indent for readability
    print(f"Wrote per-object timings for {video_path.name} to {out_json_path}")


def main():
    """
    Main function to load model and process all videos in the source directory.
    """
    t_start_main = time.time()

    # 1) Model and tracker config
    if not Path(MODEL_PATH).exists():
        print(f"[error] Model weights not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    # --- ensure half precision when on CUDA ---
    use_half = DEVICE == "cuda"
    
    tracker_config_path = Path(__file__).resolve().parent / TRACKER_YAML
    if not tracker_config_path.exists():
        print(f"[error] Tracker config not found at {tracker_config_path} (expected in same directory as script)")
        print(f"       Please ensure {TRACKER_YAML} is present.")
        return
    tracker_config_str = str(tracker_config_path)

    # 2) Ensure output directory exists
    ensure_dir(OUTPUT_DIR)

    # 3) Find video files
    video_files = []
    if VIDEO_SOURCE_DIR.exists() and VIDEO_SOURCE_DIR.is_dir():
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            video_files.extend(list(VIDEO_SOURCE_DIR.glob(f"*{ext}")))
    else:
        print(f"[error] Video source directory not found or is not a directory: {VIDEO_SOURCE_DIR}")
        print(f"       Please create it and place video files inside.")
        return

    if not video_files:
        print(f"[info] No video files found in {VIDEO_SOURCE_DIR} with extensions {SUPPORTED_VIDEO_EXTENSIONS}.")
        # Create dummy video for testing if no videos are found (optional, can be removed)
        print(f"[info] Creating a dummy video 'dummy_input.mp4' in {VIDEO_SOURCE_DIR} for testing.")
        ensure_dir(VIDEO_SOURCE_DIR)
        dummy_video_path = VIDEO_SOURCE_DIR / "dummy_input.mp4"
        w, h, fps_dummy, dur = 640, 480, 20, 5
        dummy_out_vid = cv2.VideoWriter(str(dummy_video_path),
                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps_dummy, (w, h))
        for i in range(int(fps_dummy * dur)):
            frm = np.random.randint(0, 256, (h, w, 3), np.uint8)
            cv2.circle(frm, (w//2+int(80*np.sin(i*0.15)),
                             h//2+int(80*np.cos(i*0.15))), 25, (50,150,255), -1)
            cv2.putText(frm, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            dummy_out_vid.write(frm)
        dummy_out_vid.release()
        print(f"[info] Dummy video created at {dummy_video_path}. Please run the script again.")
        return # Exit so user can re-run with the dummy or their own videos

    print(f"Found {len(video_files)} video(s) to process in {VIDEO_SOURCE_DIR}.")

    # 4) Process each video
    for video_file_path in video_files:
        process_video(video_file_path, model, tracker_config_str, OUTPUT_DIR)
        if SHOW_WINDOW and cv2.waitKey(0) & 0xFF == ord('q'): # Allow 'q' to quit between videos if SHOW_WINDOW is on
             print("[info] User pressed 'q' during video display. Exiting batch processing.")
             break
    
    if SHOW_WINDOW: # Ensure all OpenCV windows are closed if any were opened
        cv2.destroyAllWindows()

    total_time = time.time() - t_start_main
    print(f"\nBatch processing completed in {total_time:.2f}s.")
    print(f"Output JSON files and annotated videos (if enabled) are in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
