import os
import time
import math
import numpy as np
import psutil
from typing import Union, Type, Optional, List, Tuple

import concurrent.futures
import requests
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
from PIL import Image, ImageFilter
from concurrent.futures import ThreadPoolExecutor

# --- ADD SCENEDETECT IMPORTS ---
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.frame_timecode import FrameTimecode # Needed for type hints
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: scenedetect library not found. Scene-based segmentation will not be available.")
# --- END IMPORTS ---

from .models.video import VideoManifest, Segment
from .models.environment import CobraEnvironment
from .cobra_utils import (
    get_elapsed_time,
    parse_transcript,
    write_video_manifest,
    extract_base_audio,
    segment_and_extract,
    parallelize_audio,
    parallelize_transcription,
    prepare_outputs_directory,
    upload_blob,
    generate_batch_transcript,
)

# --- ADDED: Helper function for dominant color extraction ---
def _get_dominant_colors(image_path: str, num_colors: int = 5, resize_width: int = 100) -> List[str]:
    """Analyzes an image to find dominant colors using Pillow.

    Args:
        image_path: Path to the image file.
        num_colors: Number of dominant colors to extract.
        resize_width: Width to resize image to before analysis (for speed).

    Returns:
        A list of hex color codes, or an empty list if analysis fails.
    """
    try:
        img = Image.open(image_path)

        # Resize for performance
        aspect_ratio = img.height / img.width
        resize_height = int(resize_width * aspect_ratio)
        img = img.resize((resize_width, resize_height))

        # Reduce colors using quantization
        # Convert to P mode with a limited palette based on the most frequent colors
        quantized_img = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)

        # Get the palette colors
        palette = quantized_img.getpalette() # Returns bytes: R1, G1, B1, R2, G2, B2, ...
        if palette is None:
            # If no palette (e.g., single color image), get the single color
            if img.mode == 'P': # Should already be P, but double-check
                 palette_indices = img.getcolors(1) # Get [(count, index)]
                 if palette_indices:
                      dominant_index = palette_indices[0][1]
                      # Need the original palette if possible, this fallback is imperfect
                      # Try getting color directly if not P mode originally
                      original_img = Image.open(image_path)
                      colors = original_img.getcolors(1_000_000) # Get all colors from original
                      if colors and len(colors) == 1:
                            # If original was single color
                            rgb = colors[0][1][:3] if isinstance(colors[0][1], tuple) else (255, 255, 255) # Default white if unknown
                            return [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"]
                      else:
                           # Fallback if complex image somehow resulted in no palette
                           print(f"Warning: Could not extract palette for single-color quantized image: {image_path}")
                           return []
            elif img.mode == 'RGB':
                colors = img.getcolors(1)
                if colors:
                    rgb = colors[0][1]
                    return [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"]
                else:
                     print(f"Warning: Could not extract single RGB color: {image_path}")
                     return [] # Should not happen for RGB
            else: # Other modes like RGBA
                 try:
                    rgb_img = img.convert('RGB')
                    colors = rgb_img.getcolors(1)
                    if colors:
                         rgb = colors[0][1]
                         return [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"]
                    else:
                         print(f"Warning: Could not extract single converted RGB color: {image_path}")
                         return []
                 except Exception as conv_err:
                      print(f"Warning: Could not convert image mode {img.mode} to RGB for color analysis: {conv_err}")
                      return []

        # Extract dominant RGB colors from the palette
        dominant_colors_rgb = []
        for i in range(num_colors):
            r_idx, g_idx, b_idx = i * 3, i * 3 + 1, i * 3 + 2
            if r_idx < len(palette) and g_idx < len(palette) and b_idx < len(palette):
                r, g, b = palette[r_idx], palette[g_idx], palette[b_idx]
                dominant_colors_rgb.append((r, g, b))
            else:
                break # Stop if palette is shorter than expected

        # Convert RGB to Hex
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in dominant_colors_rgb]
        return hex_colors

    except FileNotFoundError:
        print(f"Error: Image file not found for color analysis: {image_path}")
        return []
    except Exception as e:
        print(f"Error analyzing dominant colors for {image_path}: {e}")
        return []
# --- END ADDED ---

class VideoPreProcessor:
    # take either a video manifest object or a path to a video manifest file
    def __init__(
        self, video_manifest: Union[str, VideoManifest], env: CobraEnvironment
    ):
        self.manifest = video_manifest
        self.env = env

    def preprocess_video(
        self,
        output_directory: str = None,
        segment_length: int = 10,
        fps: float = 0.33,
        generate_transcripts_flag: bool = True,
        max_workers: int = None,
        trim_to_nearest_second=False,
        allow_partial_segments=True,
        overwrite_output=False,
        use_speech_based_segments=False,
        use_scene_detection: bool = False,
        scene_detection_threshold: float = 30.0,
        downscale_to_max_width: int = None,
        downscale_to_max_height: int = None,
    ) -> str:
        start_time = time.time()
        print(
            f"({get_elapsed_time(start_time)}s) Preprocessing video {self.manifest.name}"
        )

        if not isinstance(self.manifest, VideoManifest):
            raise ValueError("Video manifest is not defined.")

        if fps is None or fps <= 0:
            raise ValueError("'fps' must be a positive number")

        # Only validate segment_length if not using scene detection
        if not use_scene_detection:
            if segment_length is None or segment_length <= 0:
                raise ValueError("'segment_length' must be a positive number")
            elif self.manifest.source_video.duration is not None and segment_length > self.manifest.source_video.duration:
                print("Segment length > video duration. Setting segment length to video duration.")
                segment_length = self.manifest.source_video.duration

        # Set processing parameters
        print(f"({get_elapsed_time(start_time)}) Setting processing parameters...")
        self.manifest.processing_params.fps = fps
        # Store both, but prioritize scene detection if used
        self.manifest.processing_params.segment_length = segment_length
        self.manifest.processing_params.use_speech_based_segments = use_speech_based_segments

        # --- Determine segmentation method ---
        segmentation_method = "time" # Default
        if use_scene_detection:
            if SCENEDETECT_AVAILABLE:
                segmentation_method = "scene"
                print(f"({get_elapsed_time(start_time)}s) Using Scene Detection (threshold={scene_detection_threshold}).")
            else:
                print(f"({get_elapsed_time(start_time)}s) Warning: use_scene_detection=True but scenedetect library not found. Falling back to time-based segmentation.")
        elif use_speech_based_segments:
             # Existing speech-based logic (if implemented) could go here or stay as is
             # For now, assume time-based is the primary fallback
             print(f"({get_elapsed_time(start_time)}s) Using Speech-Based Segmentation (if audio and transcription available).")
             segmentation_method = "speech" # Or keep your existing logic flow
        else:
             print(f"({get_elapsed_time(start_time)}s) Using Time-Based Segmentation (interval={segment_length}s).")
        # --- End Determination ---

        # Set other params
        if self.manifest.source_video.audio_found is False:
            self.manifest.processing_params.generate_transcript_flag = False
            if use_speech_based_segments:
                print("Warning: No audio found in video, falling back to time/scene-based segmentation")
                self.manifest.processing_params.use_speech_based_segments = False
                if segmentation_method == "speech": segmentation_method = "time" # Force fallback
        else:
            self.manifest.processing_params.generate_transcript_flag = generate_transcripts_flag

        # These might be less useful with scene detection, but keep them for now
        self.manifest.processing_params.trim_to_nearest_second = trim_to_nearest_second
        self.manifest.processing_params.allow_partial_segments = allow_partial_segments
        self.manifest.processing_params.downscaled_resolution = None

        # Prepare the output directory
        print(f"({get_elapsed_time(start_time)}s) Preparing output directory")
        output_dir_prefix = "sceneDetect_" if segmentation_method == "scene" else "" # Add prefix if using scenes
        if output_directory is not None:
            self.manifest.processing_params.output_directory = (
                prepare_outputs_directory(
                    file_name=self.manifest.name,
                    output_directory=output_directory,
                    # Include prefix in default name generation logic if output_directory is None
                    output_directory_prefix=output_dir_prefix,
                    frames_per_second=fps,
                    segment_length=segment_length if segmentation_method != "scene" else scene_detection_threshold, # Use threshold in name for scenes
                    overwrite_output=overwrite_output,
                )
            )
        else:
             self.manifest.processing_params.output_directory = (
                 prepare_outputs_directory(
                    file_name=self.manifest.name,
                    output_directory_prefix=output_dir_prefix,
                    segment_length=segment_length if segmentation_method != "scene" else scene_detection_threshold,
                    frames_per_second=fps,
                    overwrite_output=overwrite_output,
                 )
             )

        # Extract audio and transcribe (remains the same logic)
        if (
            self.manifest.source_video.audio_found
            and self.manifest.processing_params.generate_transcript_flag
        ):
            print(f"({get_elapsed_time(start_time)}s) Extracting audio and initiating Batch Transcription...")
            # Ensure max_workers is defined or defaulted before passing
            if max_workers is None:
                cpu_count = psutil.cpu_count(logical=False) or 1
                memory = psutil.virtual_memory().total / (1024**3)  # Total memory in GB
                max_workers = min(cpu_count, int(memory // 2))

            try:
                self._extract_audio_and_transcribe(max_workers) # Ensure this function exists and works
                if not self.manifest.audio_transcription:
                     print(f"({get_elapsed_time(start_time)}s) Warning: Batch transcription did not produce results.")
                else:
                     print(f"({get_elapsed_time(start_time)}s) Batch transcription completed.")
            except Exception as e:
                print(f"({get_elapsed_time(start_time)}s) ERROR during audio extraction or transcription: {e}")
                print(f"({get_elapsed_time(start_time)}s) Warning: Continuing preprocessing without transcription.")
                self.manifest.processing_params.generate_transcript_flag = False

        # --- Run Scene Detection if requested ---
        scene_list_seconds: Optional[List[Tuple[float, float]]] = None
        if segmentation_method == "scene":
            try:
                print(f"({get_elapsed_time(start_time)}s) Running PySceneDetect...")
                # Ensure video path exists before opening
                if not self.manifest.source_video.path or not os.path.exists(self.manifest.source_video.path):
                    raise FileNotFoundError(f"Source video path not found: {self.manifest.source_video.path}")

                video = open_video(self.manifest.source_video.path)
                manager = SceneManager()
                manager.add_detector(ContentDetector(threshold=scene_detection_threshold))
                manager.detect_scenes(video)
                scene_list_timecodes = manager.get_scene_list() # Pairs of FrameTimecode
                scene_list_seconds = [
                    (start.get_seconds(), end.get_seconds())
                    for start, end in scene_list_timecodes
                ]
                print(f"({get_elapsed_time(start_time)}s) Detected {len(scene_list_seconds)} scenes.")
                if not scene_list_seconds:
                     print(f"({get_elapsed_time(start_time)}s) Warning: No scenes detected. Falling back to time-based segmentation.")
                     segmentation_method = "time" # Fallback if no scenes found
                     scene_list_seconds = None
            except Exception as e:
                 print(f"({get_elapsed_time(start_time)}s) ERROR during scene detection: {e}. Falling back to time-based segmentation.")
                 segmentation_method = "time" # Fallback on error
                 scene_list_seconds = None
        # --- End Scene Detection ---

        # Generate the segments (now accepts scene_list)
        print(f"({get_elapsed_time(start_time)}s) Generating segments using {segmentation_method} method...")
        self._generate_segments(scene_list=scene_list_seconds) # Pass the detected scenes

        # Configure thread pool
        if max_workers is None:
            cpu_count = psutil.cpu_count(logical=False) or 1
            memory = psutil.virtual_memory().total / (1024**3)  # Total memory in GB
            max_workers = min(cpu_count, int(memory // 2))

        # Process the segments (remains mostly the same)
        print(f"({get_elapsed_time(start_time)}s) Processing segments (extracting frames)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, segment in enumerate(self.manifest.segments):
                # Skip segments that have already been processed (unlikely with overwrite=True, but good practice)
                if segment.processed:
                    continue
                segment_transcription_text = None
                if self.manifest.audio_transcription:
                     segment_transcription_text = parse_transcript(
                          self.manifest.audio_transcription,
                          segment.start_time,
                          segment.end_time
                     )
                futures.append(
                     executor.submit(self._preprocess_segment,
                                      segment=segment,
                                      index=i,
                                      transcription_text=segment_transcription_text,
                                      downscale_to_max_width=downscale_to_max_width,
                                      downscale_to_max_height=downscale_to_max_height
                                      )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    i, updated_segment, res, downscaled_res = future.result()
                    self.manifest.segments[i] = updated_segment
                    if self.manifest.processing_params.downscaled_resolution is None and downscaled_res is not None:
                        self.manifest.processing_params.downscaled_resolution = downscaled_res
                except Exception as exc:
                    print(f'Segment processing generated an exception: {exc}')


        print(f"({get_elapsed_time(start_time)}s) All segments pre-processed")

        # Validation check
        for segment in self.manifest.segments:
            frame_file_paths = segment.segment_frames_file_path
            frame_intervals = segment.segment_frame_time_intervals
            if len(frame_file_paths) != len(frame_intervals):
                print(
                    f"Warning: Segment {segment.segment_name}: Frame file paths ({len(frame_file_paths)}) and frame intervals ({len(frame_intervals)}) lengths do not match. Adjusting..."
                )
                min_list_length = min(
                    len(frame_file_paths), len(frame_intervals))
                segment.segment_frames_file_path = frame_file_paths[:min_list_length]
                segment.segment_frame_time_intervals = frame_intervals[:min_list_length]
                segment.number_of_frames = min_list_length # Update frame count


        write_video_manifest(self.manifest)
        return self.manifest.video_manifest_path

    def _generate_segments(self, scene_list: Optional[List[Tuple[float, float]]] = None):
        # Ensure source video duration is available
        if self.manifest.source_video.duration is None:
            raise ValueError("Cannot generate segments: Source video duration is unknown.")

        video_duration = self.manifest.source_video.duration
        analysis_fps = self.manifest.processing_params.fps
        self.manifest.segments = [] # Clear any existing segments first

        # Explicitly check if scene_list is provided (not None)
        if scene_list is not None:
            # --- Use Scene Detection Results ---
            print("Generating segments based on detected scenes...")
            effective_duration = video_duration # Use full duration
            num_segments = len(scene_list)

            self.manifest.segment_metadata = self.manifest.segment_metadata.model_copy(
                update={
                    "effective_duration": effective_duration,
                    "num_segments": num_segments, # Initial count
                }
            )

            skipped_count = 0
            actual_segments_created = 0 # Track segments added to manifest
            # Use the fallback segment length from processing_params as the split threshold
            max_scene_duration = self.manifest.processing_params.segment_length
            if max_scene_duration is None or max_scene_duration <= 0:
                print(f"Warning: Invalid segment_length ({max_scene_duration}) for scene splitting threshold. Defaulting to 30.0s.")
                max_scene_duration = 30.0 # Fallback splitting threshold

            # --- ADDED: Define max analysis unit duration --- 
            ANALYSIS_UNIT_MAX_SECONDS = 30.0
            print(f"DEBUG: Enforcing max analysis unit duration of {ANALYSIS_UNIT_MAX_SECONDS}s.")

            for i, (scene_start_time, scene_end_time) in enumerate(scene_list):
                # Ensure end_time doesn't exceed video duration slightly due to frame boundaries
                scene_end_time = min(scene_end_time, video_duration)
                # Ensure start_time is non-negative
                scene_start_time = max(0.0, scene_start_time)
                scene_duration = scene_end_time - scene_start_time

                # Ensure original scene has positive duration
                if scene_duration <= 0.001: # Allow for tiny rounding diffs
                    print(f"Skipping scene {i+1} due to zero or negative duration ({scene_start_time}s -> {scene_end_time}s).")
                    skipped_count += 1
                    continue

                # Check if original scene is long enough for any frame extraction
                min_duration_for_frame = 1.0 / analysis_fps if analysis_fps > 0 else float('inf')
                if scene_duration < min_duration_for_frame - 0.001:
                    print(f"Skipping scene {i+1} ({scene_start_time:.3f}s to {scene_end_time:.3f}s): duration {scene_duration:.3f}s is too short for any frame extraction at {analysis_fps:.2f} fps (min {min_duration_for_frame:.3f}s).")
                    skipped_count += 1
                    continue

                # --- Split long scenes ---
                if scene_duration > ANALYSIS_UNIT_MAX_SECONDS:
                    print(f"Scene {i+1} ({scene_start_time:.3f}s to {scene_end_time:.3f}s) is longer than {ANALYSIS_UNIT_MAX_SECONDS:.1f}s. Splitting into sub-segments.")
                    num_sub_segments = math.ceil(scene_duration / ANALYSIS_UNIT_MAX_SECONDS)

                    for part_idx in range(num_sub_segments):
                        sub_start_time = scene_start_time + (part_idx * ANALYSIS_UNIT_MAX_SECONDS)
                        sub_end_time = min(sub_start_time + ANALYSIS_UNIT_MAX_SECONDS, scene_end_time)
                        sub_duration = sub_end_time - sub_start_time

                        # Check if sub-segment is long enough for frames
                        if sub_duration < min_duration_for_frame - 0.001:
                            print(f"  Skipping sub-segment {part_idx+1} of scene {i+1} ({sub_start_time:.3f}s to {sub_end_time:.3f}s): duration {sub_duration:.3f}s too short for frame extraction.")
                            continue # Skip this sub-segment

                        number_of_frames_in_sub_segment = math.ceil(sub_duration * analysis_fps)
                        number_of_frames_in_sub_segment = max(1, number_of_frames_in_sub_segment)
                        use_endpoint_sub = number_of_frames_in_sub_segment > 1
                        sub_segment_frames_times = np.linspace(
                            sub_start_time, sub_end_time, number_of_frames_in_sub_segment, endpoint=use_endpoint_sub
                        )
                        if number_of_frames_in_sub_segment == 1: sub_segment_frames_times = [sub_start_time]
                        sub_segment_frames_times = [round(x, 3) for x in sub_segment_frames_times]

                        sub_segment_name = f"scene{i+1}_part{part_idx+1}_start{sub_start_time:.3f}s_end{sub_end_time:.3f}s"
                        output_directory = self.manifest.processing_params.output_directory
                        sub_segment_folder_path = os.path.join(output_directory, sub_segment_name)
                        os.makedirs(sub_segment_folder_path, exist_ok=True)

                        self.manifest.segments.append(
                            Segment(
                                segment_name=sub_segment_name,
                                segment_folder_path=sub_segment_folder_path,
                                start_time=round(sub_start_time, 3),
                                end_time=round(sub_end_time, 3),
                                segment_duration=round(sub_duration, 3),
                                number_of_frames=number_of_frames_in_sub_segment,
                                segment_frame_time_intervals=sub_segment_frames_times,
                                processed=False,
                            )
                        )
                        actual_segments_created += 1
                else:
                    # --- Process scenes shorter than or equal to ANALYSIS_UNIT_MAX_SECONDS ---
                    # (This is the original logic, slightly adapted)
                    start_time = scene_start_time
                    end_time = scene_end_time
                    segment_duration = scene_duration # Use original scene duration

                    # Frame calculation (already checked minimum duration above)
                    number_of_frames_in_segment = math.ceil(segment_duration * analysis_fps)
                    number_of_frames_in_segment = max(1, number_of_frames_in_segment)
                    use_endpoint = number_of_frames_in_segment > 1
                    segment_frames_times = np.linspace(
                        start_time, end_time, number_of_frames_in_segment, endpoint=use_endpoint
                    )
                    if number_of_frames_in_segment == 1: segment_frames_times = [start_time]
                    segment_frames_times = [round(x, 3) for x in segment_frames_times]

                    segment_name = f"scene{i+1}_start{start_time:.3f}s_end{end_time:.3f}s"
                    output_directory = self.manifest.processing_params.output_directory
                    segment_folder_path = os.path.join(output_directory, segment_name)
                    os.makedirs(segment_folder_path, exist_ok=True)

                    self.manifest.segments.append(
                        Segment(
                            segment_name=segment_name,
                            segment_folder_path=segment_folder_path,
                            start_time=round(start_time, 3),
                            end_time=round(end_time, 3),
                            segment_duration=round(segment_duration, 3),
                            number_of_frames=number_of_frames_in_segment,
                            segment_frame_time_intervals=segment_frames_times,
                            processed=False,
                        )
                    )
                    actual_segments_created += 1

            # Update num_segments based on actual segments created
            final_num_segments = actual_segments_created
            self.manifest.segment_metadata.num_segments = final_num_segments
            print(f"Created {final_num_segments} segments from {len(scene_list) - skipped_count} valid scenes (skipped {skipped_count} scenes/sub-segments).")


        else:
            # --- Use Time-Based Segmentation (Existing Logic, with added check) ---
            print("Generating segments based on fixed time intervals...")
            segment_length = self.manifest.processing_params.segment_length
            # --- ADDED: Define max analysis unit duration --- 
            ANALYSIS_UNIT_MAX_SECONDS = 30.0
            print(f"DEBUG: Enforcing max analysis unit duration of {ANALYSIS_UNIT_MAX_SECONDS}s for time-based segments.")

            trim_to_nearest_second = self.manifest.processing_params.trim_to_nearest_second
            allow_partial_segments = self.manifest.processing_params.allow_partial_segments

            if trim_to_nearest_second:
                effective_duration = math.floor(video_duration)
            else:
                effective_duration = video_duration

            # Avoid division by zero if segment_length is somehow 0
            if segment_length <= 0:
                 print("Warning: segment_length is zero or negative. Defaulting to full video as one segment.")
                 num_segments = 1
                 segment_length = effective_duration
            elif allow_partial_segments:
                num_segments = math.ceil(effective_duration / segment_length)
            else:
                num_segments = math.floor(effective_duration / segment_length)
            num_segments = max(1, int(num_segments)) # Ensure at least one segment

            self.manifest.segment_metadata = self.manifest.segment_metadata.model_copy(
                update={
                    "effective_duration": effective_duration,
                    "num_segments": num_segments, # Initial count
                }
            )

            skipped_count = 0
            actual_segments_created = 0 # Track segments added to manifest

            for i in range(num_segments):
                orig_start_time = i * segment_length
                orig_end_time = min((i + 1) * segment_length, effective_duration)
                orig_segment_duration = orig_end_time - orig_start_time

                # --- Split long time-based segments based on ANALYSIS_UNIT_MAX_SECONDS --- 
                if orig_segment_duration > ANALYSIS_UNIT_MAX_SECONDS:
                     print(f"Time segment {i+1} ({orig_start_time:.3f}s to {orig_end_time:.3f}s) is longer than {ANALYSIS_UNIT_MAX_SECONDS:.1f}s. Splitting.")
                     num_sub_segments_time = math.ceil(orig_segment_duration / ANALYSIS_UNIT_MAX_SECONDS)
                     for part_idx_time in range(num_sub_segments_time):
                          sub_start_time = orig_start_time + (part_idx_time * ANALYSIS_UNIT_MAX_SECONDS)
                          sub_end_time = min(sub_start_time + ANALYSIS_UNIT_MAX_SECONDS, orig_end_time)
                          sub_duration = sub_end_time - sub_start_time
                          # Call helper to create the sub-segment
                          self._create_segment_object(f"seg{i+1}_part{part_idx_time+1}", sub_start_time, sub_end_time, sub_duration, analysis_fps)
                          actual_segments_created += 1
                else:
                     # --- Process time segments shorter than or equal to ANALYSIS_UNIT_MAX_SECONDS ---
                     # Check if segment is long enough for frame extraction
                     min_duration_for_frame = 1.0 / analysis_fps if analysis_fps > 0 else float('inf')
                     if orig_segment_duration < min_duration_for_frame - 0.001:
                          print(f"Skipping time segment {i+1} ({orig_start_time:.3f}s to {orig_end_time:.3f}s): duration {orig_segment_duration:.3f}s too short for frame extraction.")
                          skipped_count += 1
                          continue
                     # Call helper to create the segment object
                     self._create_segment_object(f"seg{i+1}", orig_start_time, orig_end_time, orig_segment_duration, analysis_fps)
                     actual_segments_created += 1

            # Update num_segments based on actual segments created
            final_num_segments = actual_segments_created
            self.manifest.segment_metadata.num_segments = final_num_segments
            print(f"Created {final_num_segments} final analysis segments based on time interval (skipped {skipped_count} original segments). Split {num_segments - skipped_count - actual_segments_created} long segments.")


    def _create_segment_object(self, base_name: str, start_time: float, end_time: float, segment_duration: float, analysis_fps: float): 
        """Helper function to calculate frames and create a Segment object."""
        number_of_frames = max(1, math.ceil(segment_duration * analysis_fps))
        use_endpoint = number_of_frames > 1
        frame_times = np.linspace(start_time, end_time, number_of_frames, endpoint=use_endpoint)
        if number_of_frames == 1: frame_times = [start_time]
        frame_times = [round(t, 3) for t in frame_times]

        segment_name = f"{base_name}_start{start_time:.3f}s_end{end_time:.3f}s"
        output_directory = self.manifest.processing_params.output_directory
        segment_folder_path = os.path.join(output_directory, segment_name)
        os.makedirs(segment_folder_path, exist_ok=True)

        self.manifest.segments.append(
            Segment(
                segment_name=segment_name,
                segment_folder_path=segment_folder_path,
                start_time=round(start_time, 3),
                end_time=round(end_time, 3),
                segment_duration=round(segment_duration, 3),
                number_of_frames=number_of_frames,
                segment_frame_time_intervals=frame_times,
                processed=False,
            )
        )

    def _preprocess_segment(self, segment: Segment, index: int, transcription_text: Optional[str], downscale_to_max_width=None, downscale_to_max_height=None):
        stop_watch_time = time.time()
        print(
            f"**Processing {segment.segment_name} - beginning frame extraction."
        )
        try:
            input_video_path = self.manifest.source_video.path
            segment_path = segment.segment_folder_path
            start_time = segment.start_time # Use segment's start time
            end_time = segment.end_time   # Use segment's end time
            fps = self.manifest.processing_params.fps # Use processing fps

            # Ensure start/end times are valid before passing to ffmpeg
            if start_time is None or end_time is None or start_time >= end_time:
                 print(f"Warning: Invalid start/end times for {segment.segment_name} ({start_time} -> {end_time}). Skipping frame extraction.")
                 segment.processed = False # Mark as failed if times invalid
                 return index, segment, False, None

            frames_dir = os.path.join(segment_path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            segment_video_path = os.path.join(segment_path, "segment.mp4") # Temp segment file

            # Call ffmpeg wrapper
            segment_and_extract(start_time, end_time, input_video_path, segment_path, frames_dir, fps,
                               downscale_to_max_width=downscale_to_max_width,
                               downscale_to_max_height=downscale_to_max_height)

            # --- Update frame renaming logic to handle potentially different number of frames ---
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")])
            number_of_extracted_frames = len(frame_files)

            # Recalculate frame times based on actual extracted frames and segment duration
            segment_duration = segment.segment_duration
            if number_of_extracted_frames > 0:
                 # Use endpoint=True for linspace if only one frame to get the start time
                 use_endpoint = number_of_extracted_frames > 1
                 frame_times = np.linspace(start_time, end_time, number_of_extracted_frames, endpoint=use_endpoint)
                 # Correct single-frame time to be start_time
                 if number_of_extracted_frames == 1:
                     frame_times = [start_time]
                 frame_times = [round(t, 3) for t in frame_times]
            else:
                 frame_times = [] # No frames extracted
                 print(f"Warning: No frames extracted for {segment.segment_name}. Check segment duration and video.")


            # Check if extracted frames match expected (can happen with edge cases)
            if number_of_extracted_frames != segment.number_of_frames:
                 print(f"Warning: Segment {segment.segment_name} - Expected {segment.number_of_frames} frames, extracted {number_of_extracted_frames}. Using extracted count.")
                 segment.number_of_frames = number_of_extracted_frames


            segment.segment_frame_time_intervals = frame_times # Use recalculated times
            segment.segment_frames_file_path = [] # Reset and populate

            # Safety check for list length mismatch before zipping
            if len(frame_files) != len(frame_times):
                 print(f"Error: Mismatch between listed frame files ({len(frame_files)}) and calculated frame times ({len(frame_times)}) for {segment.segment_name}. Adjusting...")
                 min_len = min(len(frame_files), len(frame_times))
                 frame_files = frame_files[:min_len]
                 frame_times = frame_times[:min_len]
                 segment.number_of_frames = min_len # Update frame count again

            for i, (frame_file, frame_time) in enumerate(zip(frame_files, frame_times)):
                 old_frame_path = os.path.join(frames_dir, frame_file)
                 # Use 3 decimal places in filename
                 new_frame_filename = f"frame_{i}_{frame_time:.3f}s.jpg"
                 new_frame_path = os.path.join(frames_dir, new_frame_filename)
                 try:
                     # Avoid renaming if names are identical (can happen with rounding)
                     if old_frame_path != new_frame_path:
                         os.rename(old_frame_path, new_frame_path)
                     segment.segment_frames_file_path.append(new_frame_path)
                 except OSError as e:
                     print(f"Error renaming frame {old_frame_path} to {new_frame_path}: {e}")
                     # Decide how to handle - skip frame? For now, log and continue
            # --- End frame renaming update ---


            # Clean up temp segment video
            if os.path.exists(segment_video_path):
                 try: os.remove(segment_video_path)
                 except OSError as e: print(f"Warning: Could not remove temp segment file {segment_video_path}: {e}")

            print(f"**Processed {segment.segment_name} - extracted {segment.number_of_frames} frames in {get_elapsed_time(stop_watch_time)}s")

            segment.transcription = transcription_text if transcription_text else "No transcription for this segment."
            segment.processed = True

            downscaled_res = None
            if segment.segment_frames_file_path:
                 try:
                      with Image.open(segment.segment_frames_file_path[0]) as img:
                           width, height = img.size
                           print(f"Frame resolution for {segment.segment_name}: {width}x{height}")
                           downscaled_res = [width, height]
                 except Exception as e:
                      print(f"Warning: Could not determine frame resolution for {segment.segment_name}: {e}")

            # --- ADDED: Analyze dominant colors from middle frame ---
            segment.dominant_colors_hex = [] # Default to empty list
            if segment.segment_frames_file_path:
                middle_frame_index = len(segment.segment_frames_file_path) // 2
                middle_frame_path = segment.segment_frames_file_path[middle_frame_index]
                print(f"Analyzing dominant colors for {segment.segment_name} using frame: {os.path.basename(middle_frame_path)}")
                segment.dominant_colors_hex = _get_dominant_colors(middle_frame_path, num_colors=5) # Store hex list
                print(f" -> Dominant colors found: {segment.dominant_colors_hex}")
            else:
                 print(f"Warning: No frames found for {segment.segment_name}, skipping color analysis.")
            # --- END ADDED ---

            # --- ADDED: Upload frames after extraction and renaming ---
            self._upload_frames_get_urls(segment)
            # --- END ADDED ---

            return index, segment, True, downscaled_res

        except Exception as e:
            import traceback
            print(f"Error processing segment {segment.segment_name} frames: {e}")
            traceback.print_exc() # Print full traceback for ffmpeg errors
            segment.processed = False
            return index, segment, False, None

    def _extract_audio_and_transcribe(self, max_workers: int):
         # This method extracts the *full* audio for transcription,
         # which is independent of video segmentation method. No changes needed here.
         # Ensure this method is correctly implemented based on previous discussions.
         start_t = time.time()
         base_audio_name = f"{os.path.splitext(self.manifest.name)[0]}.wav"
         local_audio_path = os.path.join(
              self.manifest.processing_params.output_directory,
              base_audio_name
         )
         try:
             # --- 1. Extract Full Audio Locally ---
             print(f"({get_elapsed_time(start_t)}s) Extracting full audio to {local_audio_path} (Format: WAV)...")
             extract_base_audio(self.manifest.source_video.path, local_audio_path)
             self.manifest.source_audio.path = local_audio_path
             # Add check for file existence after extraction
             if not os.path.exists(local_audio_path):
                 raise FileNotFoundError(f"Extracted audio file not found: {local_audio_path}")
             self.manifest.source_audio.file_size_mb = os.path.getsize(local_audio_path) / (1024 * 1024)
             print(f"({get_elapsed_time(start_t)}s) Audio extracted ({self.manifest.source_audio.file_size_mb:.2f} MB).")

             # --- 2. Upload Audio to Azure Blob Storage (if configured) ---
             blob_config = self.env.blob_storage
             if not blob_config or not blob_config.account_name or not blob_config.container_name or \
                (not blob_config.connection_string and not blob_config.sas_token):
                 print(f"({get_elapsed_time(start_t)}s) Blob Storage not configured. Skipping upload and Batch Transcription.")
                 self.manifest.audio_transcription = None
                 return # Exit if no blob config

             print(f"({get_elapsed_time(start_t)}s) Uploading audio to Azure Blob Storage...")
             blob_name_upload = f"audio_uploads/{base_audio_name}" # Use a subfolder
             audio_blob_sas_url = upload_blob(
                 local_file_path=local_audio_path,
                 blob_name=blob_name_upload,
                 env=self.env,
                 overwrite=True,
                 read_permission_hours=48 # Sufficient time for transcription
             )

             if not audio_blob_sas_url:
                 print(f"({get_elapsed_time(start_t)}s) Failed to upload audio to Blob Storage. Skipping transcription.")
                 self.manifest.audio_transcription = None
                 return # Exit if upload failed

             print(f"({get_elapsed_time(start_t)}s) Audio uploaded. SAS URL generated.")

             # --- 3. Call Batch Transcription ---
             print(f"({get_elapsed_time(start_t)}s) Submitting Batch Transcription job...")
             # Specify candidate locales if known, otherwise default
             candidate_locales = ["en-US"] # Or load from config/params
             transcript_result = generate_batch_transcript(
                  audio_blob_sas_url=audio_blob_sas_url,
                  env=self.env,
                  candidate_locales=candidate_locales
             )
             self.manifest.audio_transcription = transcript_result
             if transcript_result:
                  print(f"({get_elapsed_time(start_t)}s) Batch transcription job finished successfully.")
             else:
                  print(f"({get_elapsed_time(start_t)}s) Batch transcription job failed or returned no result.")

         except Exception as e:
             print(f"({get_elapsed_time(start_t)}s) ERROR during audio extraction or transcription pipeline: {e}")
             self.manifest.audio_transcription = None # Ensure transcription is None on error
             # Optionally re-raise or handle differently depending on desired behavior

         # --- 4. Optional: Clean up local audio ---
         finally:
             if os.path.exists(local_audio_path):
                  try:
                       print(f"({get_elapsed_time(time.time())}s) Removing local audio file: {local_audio_path}") # Use current time
                       os.remove(local_audio_path)
                  except Exception as e:
                       print(f"({get_elapsed_time(time.time())}s) Warning: Could not remove local audio file {local_audio_path}: {e}")

    # --- ADDED: Helper to upload a single frame ---
    def _upload_one_frame(self, local_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Uploads a single frame and returns (sas_url, blob_name)."""
        try:
            vid_prefix = os.path.splitext(self.manifest.name)[0]
            blob_name = f"temp-frames/{vid_prefix}/{os.path.basename(local_path)}"
            sas_url = upload_blob(
                local_file_path=local_path,
                blob_name=blob_name,
                env=self.env,
                overwrite=True,
                read_permission_hours=4 # SAS URL valid for 4 hours
            )
            if sas_url:
                return sas_url, blob_name
            else:
                print(f"Warning: Failed to upload frame {local_path}")
                return None, None
        except Exception as e:
            print(f"Error uploading frame {local_path}: {e}")
            return None, None
    # --- END ADDED ---

    # --- ADDED: Upload all segment frames concurrently ---
    def _upload_frames_get_urls(self, segment: Segment) -> None:
        """
        Uploads each local JPG in segment.segment_frames_file_path concurrently
        to <container>/temp-frames/<video-id>/<filename>
        and stores the SAS URL list in segment.frame_urls and blob names in segment._blob_names.
        """
        if not segment.segment_frames_file_path:
            segment.frame_urls = []
            segment._blob_names = []
            print(f"No frames to upload for {segment.segment_name}.")
            return

        print(f"Uploading {len(segment.segment_frames_file_path)} frames for {segment.segment_name} concurrently...")
        start_upload_time = time.time()
        frame_urls = []
        blob_names = []

        # Use ThreadPoolExecutor for concurrent uploads (I/O bound)
        # Adjust max_workers based on your environment/network capacity
        with ThreadPoolExecutor(max_workers=16) as pool:
            # Map the upload function to the list of local paths
            results = list(pool.map(self._upload_one_frame, segment.segment_frames_file_path))

        # Process results, filtering out failures
        for sas_url, blob_name in results:
            if sas_url and blob_name:
                frame_urls.append(sas_url)
                blob_names.append(blob_name)
            # Failures are logged within _upload_one_frame

        segment.frame_urls = frame_urls # Store successful URLs
        segment._blob_names = blob_names # Store names of successfully uploaded blobs
        upload_duration = time.time() - start_upload_time
        print(f"Finished uploading {len(frame_urls)} frames for {segment.segment_name} in {upload_duration:.2f}s.")
        if len(frame_urls) != len(segment.segment_frames_file_path):
             print(f"Warning: Failed to upload {len(segment.segment_frames_file_path) - len(frame_urls)} frames for {segment.segment_name}.")

    # --- END ADDED ---
